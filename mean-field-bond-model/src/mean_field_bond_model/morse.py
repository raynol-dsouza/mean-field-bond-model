"""
Morse bonding potentials: evaluation, calibration against Hessians, parameter
I/O and LAMMPS bond coefficients.

One Hessian fixes only the curvature k = 2 D_e a^2. Hessians at several
isotropic volumes fix both parameters through the slope of k(r): at r_e,
dk/dr = -6 D_e a^3, so a = -(dk/dr) / (3 k) and D_e = k / (2 a^2), refined by
a least-squares fit of k(r).
"""

import json
import warnings

import numpy as np
from scipy.optimize import least_squares

from .symmetry import NeighborBondSymmetryAnalysis, generate_bond_vectors

__all__ = [
    "MorsePotential",
    "build_bond_inputs",
    "build_bonding_potentials",
    "compute_morse_stiffness",
    "load_morse_params",
    "save_morse_params",
]


def _morse_curvature(r, D_e, a, r_e):
    e = np.exp(-a * (r - r_e))
    return 2.0 * D_e * a**2 * (2.0 * e**2 - e)


class MorsePotential:
    """V(b) = D_e (1 - exp(-a (|b| - r_e)))^2, on arrays of bond vectors of
    shape (..., 3)."""

    def __init__(self, D_e, a, r_e):
        self.D_e = float(D_e)
        self.a = float(a)
        self.r_e = float(r_e)

    def __repr__(self):
        return f"MorsePotential(D_e={self.D_e}, a={self.a}, r_e={self.r_e})"

    def __call__(self, b):
        r = np.sqrt(np.einsum("...i,...i->...", b, b))
        return self.D_e * (1.0 - np.exp(-self.a * (r - self.r_e))) ** 2

    def gradient(self, b):
        r = np.sqrt(np.einsum("...i,...i->...", b, b))
        e = np.exp(-self.a * (r - self.r_e))
        return (2.0 * self.D_e * self.a * e * (1.0 - e) / r)[..., None] * b


def _spring_constants(structure, analysis, hessian):
    """Mean longitudinal spring constant -b_hat . Phi . b_hat per (Wyckoff
    site, shell) of each representative."""
    k = {}
    for w, wd in analysis["Wyckoff_sites"].items():
        bs = wd["basis_sites"][0]
        data = analysis["basis_sites"][bs]
        for s, sd in data["shells"].items():
            b = generate_bond_vectors(analysis, bs, s)
            b /= np.linalg.norm(b, axis=-1, keepdims=True)
            blocks = hessian[data["representative_atom"], sd["neighbor_ids"]]
            k[(w, s)] = float(-np.einsum("ni,nij,nj->n", b, blocks, b).mean())
    return k


def compute_morse_stiffness(
    structure,
    hessian,
    n_shells,
    D_e=1e-3,
    strained_structures=None,
    strained_hessians=None,
    k_min=1e-3,
    a_min=1e-3,
    a_max=10.0,
    D_e_max=50.0,
    decimals=4,
    on_unphysical="fallback",
):
    """
    Morse parameters for every shell of every Wyckoff site, with r_e the
    reference bond length.

    Without strained_structures, D_e is fixed (a float or {Wyckoff_site:
    {shell: D_e}}) and a = sqrt(k / (2 D_e)). With strained_structures
    (isotropic scalings of structure) and their Hessians, D_e and a are fitted
    to k(r); a shell whose fit is unphysical raises (on_unphysical="raise") or
    falls back to the fixed D_e ("fallback"). A shell with negative k is made
    inert (D_e = a = 0).

    Returns
    -------
    morse_params : dict
        {Wyckoff_site: {"species", "basis_sites", "shells": {shell: {"D_e",
        "a", "r_e", "target_species"}}}}
    diagnostics : dict
        Same layout; each shell holds "k", "r_e", "mode" ("fixed_D_e",
        "joint_fit", "fallback_fixed_D_e" or "negative_k_null") and, for a
        joint fit, "residual_rms", "r_samples" and "k_samples".
    """
    if on_unphysical not in ("raise", "fallback"):
        raise ValueError("on_unphysical must be 'raise' or 'fallback'.")
    analysis = NeighborBondSymmetryAnalysis(
        structure, n_shells=n_shells, decimals=decimals
    ).analyze()
    if len(strained_structures or []) != len(strained_hessians or []):
        raise ValueError("Give one Hessian per strained structure.")
    volumes = [(1.0, hessian)]
    for strained, H in zip(strained_structures or [], strained_hessians or []):
        c = (strained.get_volume() / structure.get_volume()) ** (1 / 3)
        if not np.allclose(
            strained.get_cell().array,
            c * structure.get_cell().array,
            atol=1e-6,
        ):
            raise ValueError(
                "Strained structures must be isotropic scalings of the "
                "reference."
            )
        volumes.append((c, H))
    k_of = [(c, _spring_constants(structure, analysis, H)) for c, H in volumes]

    def fixed(w, s, k):
        D = D_e[w][s] if isinstance(D_e, dict) else D_e
        a = np.sqrt(max(k, k_min) / (2.0 * D))
        return D, (a if a_min <= a <= a_max else a_min)

    params = {}
    diagnostics = {}
    for w, wd in analysis["Wyckoff_sites"].items():
        shells = analysis["basis_sites"][wd["basis_sites"][0]]["shells"]
        params[w] = {
            "species": wd["species"],
            "basis_sites": list(wd["basis_sites"]),
            "shells": {},
        }
        diagnostics[w] = {
            "species": wd["species"],
            "basis_sites": list(wd["basis_sites"]),
            "shells": {},
        }
        for s, sd in shells.items():
            r_e = float(sd["reference_bond_length"])
            r = np.array([c * r_e for c, _ in k_of])
            k = np.array([ks[(w, s)] for _, ks in k_of])
            entry = {
                "k": float(k[0]),
                "r_e": r_e,
                "target_species": sd["target_species"],
            }
            if k[0] < 0:
                warnings.warn(
                    f"(Wyckoff site {w}, shell {s}) has negative k = "
                    f"{k[0]:.4g}; set inert.",
                    stacklevel=2,
                )
                D = 0.0
                a = 0.0
                entry["mode"] = "negative_k_null"
            elif len(k) == 1:
                D, a = fixed(w, s, k[0])
                entry["mode"] = "fixed_D_e"
            else:
                entry.update(r_samples=r.tolist(), k_samples=k.tolist())
                slope = np.polyfit(r - r_e, k, 1)[0]
                a0 = -slope / (3.0 * max(k[0], k_min))
                D0 = max(k[0], k_min) / (2.0 * a0**2) if a0 > 0 else np.inf
                if a_min <= a0 <= a_max and D0 <= D_e_max:
                    fit = least_squares(
                        lambda x: _morse_curvature(r, x[0], np.exp(x[1]), r_e)
                        - k,
                        [D0, np.log(a0)],
                        bounds=(
                            [0.0, np.log(a_min)],
                            [D_e_max, np.log(a_max)],
                        ),
                    )
                    D = fit.x[0]
                    a = np.exp(fit.x[1])
                    entry.update(
                        mode="joint_fit",
                        residual_rms=float(np.sqrt(np.mean(fit.fun**2))),
                    )
                elif on_unphysical == "raise":
                    raise ValueError(
                        f"(Wyckoff site {w}, shell {s}): unphysical Morse fit "
                        f"(k = {k[0]:.4g}, dk/dr = {slope:.4g})."
                    )
                else:
                    warnings.warn(
                        f"(Wyckoff site {w}, shell {s}): unphysical Morse "
                        "fit; using fixed D_e.",
                        stacklevel=2,
                    )
                    D, a = fixed(w, s, k[0])
                    entry["mode"] = "fallback_fixed_D_e"
                if np.isclose(a, a_max):
                    a = a_min
            params[w]["shells"][s] = {
                "D_e": float(np.round(D, decimals)),
                "a": float(np.round(a, decimals)),
                "r_e": r_e,
                "target_species": sd["target_species"],
            }
            diagnostics[w]["shells"][s] = entry
    return params, diagnostics


def save_morse_params(morse_params, path):
    data = {
        str(w): {**d, "shells": {str(s): p for s, p in d["shells"].items()}}
        for w, d in morse_params.items()
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_morse_params(path):
    with open(path) as f:
        data = json.load(f)
    return {
        int(w): {**d, "shells": {int(s): p for s, p in d["shells"].items()}}
        for w, d in data.items()
    }


def build_bonding_potentials(morse_params):
    """
    {Wyckoff_site: {"species", "basis_sites", "shells": {shell:
    MorsePotential}}}, with one shared object per distinct parameter set, so
    both ends of a bond carry the same potential.
    """
    shared = {}
    out = {}
    for w, d in morse_params.items():
        shells = {}
        for s, p in d["shells"].items():
            key = (p["D_e"], p["a"], p["r_e"])
            shells[s] = shared.setdefault(key, MorsePotential(*key))
        out[w] = {
            "species": d["species"],
            "basis_sites": list(d["basis_sites"]),
            "shells": shells,
        }
    return out


def build_bond_inputs(structure, n_shells, morse_params):
    """
    LAMMPS inputs for a Morse bond lattice: the structure with its bond
    topology attached, and {"bond_style", "bond_coeff_ids", "bond_coeff_args",
    "species"} with one bond_coeff per bond type.
    """
    analyzer = NeighborBondSymmetryAnalysis(structure, n_shells=n_shells)
    analysis = analyzer.analyze()
    _, bond_type_shells = analyzer.assign_bond_types()
    ids = []
    args = []
    for bond_type, pair in sorted(bond_type_shells.items()):
        p, q = (morse_params[w]["shells"][s] for w, s in pair)
        if any(not np.isclose(p[k], q[k]) for k in ("D_e", "a", "r_e")):
            raise ValueError(
                f"Bond type {bond_type}: the two ends {pair} carry different "
                "Morse parameters."
            )
        ids.append(str(bond_type))
        args.append(f"{p['D_e']} {p['a']} {p['r_e']}")
    return analyzer.get_structure_with_bonds(), {
        "bond_style": "morse",
        "bond_coeff_ids": ids,
        "bond_coeff_args": args,
        "species": analysis["species"],
    }

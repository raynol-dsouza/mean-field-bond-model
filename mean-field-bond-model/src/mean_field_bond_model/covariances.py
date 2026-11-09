"""
Normalized bond-bond covariances alpha = Cov(delta b_target, delta b_ref)
Cov(delta b_ref, delta b_ref)^-1 from a real-space Hessian, by a Bloch sum over
the q-points commensurate with the supercell.

The dynamical matrix is not mass-weighted: classical displacement covariances
are [Phi(q)^-1], so the alphas do not depend on the masses.

The bonds summed in a reference shell's effective potential leave from a finite
cluster of atoms. For a reference bond joining two Wyckoff sites the cluster is
the union of the neighborhoods of both of its ends, so both ends sum over the
same bonds and give the same effective potential. Otherwise it is the
neighborhood of the origin atom.
"""

import warnings

import numpy as np
from scipy.spatial import cKDTree

from .symmetry import get_primitive_cell

__all__ = [
    "BondPairCovarianceCalculator",
    "commensurate_qpoints",
    "monkhorst_pack_qpoints",
]


def _fold(x):
    return x - np.ceil(x - 0.5)


def commensurate_qpoints(supercell_cell, primitive_cell):
    """
    The q-points (Cartesian) commensurate with the supercell, one per +-q pair,
    with weights 1 for self-paired points and 2 otherwise, and the total number
    of q-points.
    """
    N = np.round(supercell_cell @ np.linalg.inv(primitive_cell))
    n_q = round(abs(np.linalg.det(N)))
    corners = (
        np.array(np.meshgrid([0, 1], [0, 1], [0, 1], indexing="ij"))
        .reshape(3, -1)
        .T
        @ N.T
    )
    lo = np.floor(corners.min(0)).astype(int) - 2
    hi = np.ceil(corners.max(0)).astype(int) + 2
    m = np.stack(
        np.meshgrid(
            *[np.arange(a, b + 1) for a, b in zip(lo, hi)], indexing="ij"
        ),
        -1,
    ).reshape(-1, 3)
    raw = m @ np.linalg.inv(N).T
    q = _fold(np.unique(np.round(raw - np.floor(raw + 1e-8), 8), axis=0))
    if len(q) != n_q:
        raise ValueError(
            "The supercell is not an integer multiple of the primitive cell."
        )

    return _reduce_by_inversion(q, primitive_cell)


def monkhorst_pack_qpoints(primitive_cell, mesh_size):
    """Gamma-centred Monkhorst-Pack mesh of mesh_size (int or 3 ints), in the
    same form as commensurate_qpoints."""
    n = np.broadcast_to(mesh_size, 3)
    q = np.stack(
        np.meshgrid(*[np.arange(k) / k for k in n], indexing="ij"), -1
    ).reshape(-1, 3)
    return _reduce_by_inversion(_fold(q), primitive_cell)


def _reduce_by_inversion(q, primitive_cell):
    """One q per +-q pair (weight 2, or 1 if q is its own partner), in
    Cartesian coordinates."""
    q = np.round(q, 8) + 0.0
    index = {tuple(x): i for i, x in enumerate(q)}
    partner = [index[tuple(np.round(_fold(-x), 8) + 0.0)] for x in q]
    keep = [i for i in range(len(q)) if i <= partner[i]]
    weights = np.array([1 if partner[i] == i else 2 for i in keep])
    return (
        q[keep] @ (2 * np.pi * np.linalg.inv(primitive_cell).T),
        weights,
        len(q),
    )


class BondPairCovarianceCalculator:
    """
    Parameters
    ----------
    structure : ase.Atoms
        The supercell the symmetry analysis and the Hessian were built on.
    symmetry_analysis : dict
        From NeighborBondSymmetryAnalysis.analyze().
    hessian : np.ndarray, shape (N, N, 3, 3)
    mesh_size : int or 3 ints, optional
        Gamma-centred Monkhorst-Pack mesh. Defaults to None: the q-points
        commensurate with the supercell, which is exact for a supercell
        Hessian.
    rcond : float, optional
        Eigenvalues of D(q) at or below rcond * max are treated as null in the
        pseudo-inverse.
    """

    def __init__(
        self,
        structure,
        symmetry_analysis,
        hessian,
        mesh_size=None,
        rcond=1e-10,
        qchunk=512,
    ):
        self.analysis = symmetry_analysis
        self.basis_sites = symmetry_analysis["basis_sites"]
        self.representative = {
            w: d["basis_sites"][0]
            for w, d in symmetry_analysis["Wyckoff_sites"].items()
        }
        primitive_cell = get_primitive_cell(structure)[0]
        if mesh_size is None:
            self.qpoints, self.weights, self.n_q = commensurate_qpoints(
                structure.get_cell().array, primitive_cell
            )
        else:
            self.qpoints, self.weights, self.n_q = monkhorst_pack_qpoints(
                primitive_cell, mesh_size
            )
        self.Dinv = np.concatenate(
            [
                self._pseudo_inverse(
                    self._dynamical_matrices(
                        structure, hessian, self.qpoints[i : i + qchunk]
                    ),
                    rcond,
                )
                for i in range(0, len(self.qpoints), qchunk)
            ]
        )

    def _dynamical_matrices(self, structure, hessian, qpoints):
        n = len(self.basis_sites)
        D = np.zeros((len(qpoints), 3 * n, 3 * n), dtype=complex)
        positions = structure.get_positions()
        cell = structure.get_cell().array
        basis_of_atom = self.analysis["atom_to_basis_site_map"]
        for bs, data in self.basis_sites.items():
            atom = data["representative_atom"]
            coupled = np.abs(hessian[atom]).sum(axis=(1, 2)) > 0
            coupled[atom] = False
            neighbors = np.flatnonzero(coupled)
            blocks = hessian[atom, neighbors]
            frac = (positions[neighbors] - positions[atom]) @ np.linalg.inv(
                cell
            )
            phases = np.exp(1j * ((frac - np.round(frac)) @ cell @ qpoints.T))
            D[:, 3 * bs : 3 * bs + 3, 3 * bs : 3 * bs + 3] -= blocks.sum(
                axis=0
            )
            for t in np.unique(basis_of_atom[neighbors]):
                mask = basis_of_atom[neighbors] == t
                D[:, 3 * bs : 3 * bs + 3, 3 * t : 3 * t + 3] += np.einsum(
                    "nq,nij->qij", phases[mask], blocks[mask]
                )
        return D

    @staticmethod
    def _pseudo_inverse(D, rcond):
        values, vectors = np.linalg.eigh(0.5 * (D + D.conj().swapaxes(-1, -2)))
        scale = np.abs(values).max(axis=-1, keepdims=True)
        if values.min() < -1e-8 * scale.max():
            warnings.warn(
                "D(q) has negative eigenvalues: the structure is "
                "dynamically unstable. "
                "Those modes are excluded.",
                RuntimeWarning,
                stacklevel=3,
            )
        inverse = np.where(
            values > rcond * scale,
            1.0 / np.where(values == 0, 1.0, values),
            0.0,
        )
        return (vectors * inverse[:, None, :]) @ vectors.conj().swapaxes(
            -1, -2
        )

    def _covariances(self, ref, bonds):
        """Cov(delta b, delta b_ref), shape (len(bonds), 3, 3), for bonds given
        as dicts with origin/target/vector/offset."""
        a = 3 * ref["origin"]
        a_t = 3 * ref["target"]
        column = (
            self.Dinv[:, :, a_t : a_t + 3]
            * np.exp(-1j * self.qpoints @ ref["vector"])[:, None, None]
            - self.Dinv[:, :, a : a + 3]
        )
        groups = {}
        for i, bond in enumerate(bonds):
            groups.setdefault((bond["origin"], bond["target"]), []).append(i)
        covariances = np.zeros((len(bonds), 3, 3))
        for (o, t), index in groups.items():
            offsets = np.array([bonds[i]["offset"] for i in index])
            vectors = np.array([bonds[i]["vector"] for i in index])
            to_target = (
                np.exp(1j * (offsets + vectors) @ self.qpoints.T)
                * self.weights
            )
            to_origin = np.exp(1j * offsets @ self.qpoints.T) * self.weights
            term = to_target @ column[:, 3 * t : 3 * t + 3].reshape(
                -1, 9
            ) - to_origin @ column[:, 3 * o : 3 * o + 3].reshape(-1, 9)
            covariances[index] = np.real(term).reshape(-1, 3, 3) / self.n_q
        return covariances

    def _resolve(self, scaled, target_wyckoff, vector, tol=1e-3):
        """Basis site of target_wyckoff at fractional position scaled + vector,
        and its own scaled position."""
        lattice = self.analysis["primitive_lattice"]
        position = scaled + vector @ np.linalg.inv(lattice)
        candidates = self.analysis["Wyckoff_sites"][target_wyckoff][
            "basis_sites"
        ]
        delta = position - np.array(
            [self.basis_sites[bs]["primitive_scaled"] for bs in candidates]
        )
        distances = np.linalg.norm(
            (delta - np.round(delta)) @ lattice, axis=-1
        )
        if distances.min() > tol:
            raise RuntimeError(
                "A bond lands on no basis site; structure and analysis do "
                "not match."
            )
        bs = candidates[int(distances.argmin())]
        return bs, self.basis_sites[bs]["primitive_scaled"]

    def _bonds_from(self, bs, position, scaled, shells="shells"):
        """Every bond leaving one atom, generated from its Wyckoff
        representative's shells and its coset rotation."""
        data = self.basis_sites[bs]
        w = data["Wyckoff_site"]
        rep = self.basis_sites[self.representative[w]]
        for shell, shell_data in sorted(rep[shells].items()):
            target_wyckoff = self.basis_sites[shell_data["target_basis_site"]][
                "Wyckoff_site"
            ]
            for k in shell_data["rotation_ids"]:
                rotation = data["coset_rotation"] @ rep["rotations"][k]
                vector = rotation @ shell_data["reference_bond_vector"]
                target, target_scaled = self._resolve(
                    scaled, target_wyckoff, vector
                )
                yield {
                    "origin": bs,
                    "target": target,
                    "vector": vector,
                    "offset": position,
                    "rotation": rotation,
                    "potential_key": (w, shell),
                    "target_species": shell_data["target_species"],
                    "target_scaled": target_scaled,
                }

    def _neighborhood(self, bs, position, scaled):
        """The atom itself and its neighbors in every bonded and origin-only
        shell, as (basis_site, position, scaled)."""
        atoms = [(bs, np.asarray(position, float), scaled)]
        for part in ("shells", "origin_shells"):
            atoms += [
                (b["target"], position + b["vector"], b["target_scaled"])
                for b in self._bonds_from(bs, position, scaled, part)
            ]
        return atoms

    @staticmethod
    def _bond_weights(bonds, tol=1e-3):
        """
        1/2 for a bond whose reverse is also enumerated, 1 otherwise, so every
        physical bond totals 1; and a shared pair id for the two halves of a
        reversed pair.
        """
        forward = np.array(
            [np.concatenate([b["offset"], b["vector"]]) for b in bonds]
        )
        reverse = np.array(
            [
                np.concatenate([b["offset"] + b["vector"], -b["vector"]])
                for b in bonds
            ]
        )
        residuals, partners = cKDTree(forward).query(reverse)
        matched = residuals < tol
        index = np.arange(len(bonds))
        return np.where(matched, 0.5, 1.0), np.where(
            matched, np.minimum(index, partners), index
        )

    @staticmethod
    def _collapse(targets, rtol=1e-6):
        """
        Merge targets entering V_eff identically, weight * V(M^T d + a) with
        M = alpha^T rotation, summing their weights: reversed pairs, and any
        other targets with the same M.
        """
        merged = []
        by_key = {}
        for i, t in enumerate(targets):
            by_key.setdefault(t["potential_key"], []).append(i)
        for index in by_key.values():
            M = np.array(
                [
                    (targets[i]["alpha"].T @ targets[i]["rotation"]).ravel()
                    for i in index
                ]
            )
            parent = list(range(len(index)))

            def root(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            first = {}
            for n, i in enumerate(index):
                parent[root(n)] = root(
                    first.setdefault(targets[i]["pair_id"], n)
                )
            tree = cKDTree(M)
            for n in range(len(index)):
                for m in tree.query_ball_point(
                    M[n], rtol * max(np.linalg.norm(M[n]), 1e-30)
                ):
                    parent[root(m)] = root(n)
            clusters = {}
            for n, i in enumerate(index):
                clusters.setdefault(root(n), []).append(i)
            for members in clusters.values():
                keep = min(
                    members,
                    key=lambda i: (np.linalg.norm(targets[i]["offset"]), i),
                )
                merged.append(
                    (
                        keep,
                        {
                            **targets[keep],
                            "weight": sum(
                                targets[i]["weight"] for i in members
                            ),
                        },
                    )
                )
        return [t for _, t in sorted(merged, key=lambda x: x[0])]

    def compute_alphas(self, alpha_threshold=1e-3, tol=1e-3):
        """
        Returns
        -------
        alphas : dict
            {Wyckoff_site: {"species", "basis_sites", "shells": {shell:
            {"reference_bond_vector", "targets": [{"target_bond_vector",
            "target_species", "alpha", "rotation", "potential_key", "weight"},
            ...]}}}}

            A target enters the effective potential as weight * V_key(alpha^T
            rotation d + a_key), where potential_key = (origin Wyckoff site,
            shell) and d is the reference bond's displacement.
        """
        alphas = {}
        for w, bs in self.representative.items():
            data = self.basis_sites[bs]
            origin_atoms = self._neighborhood(
                bs, np.zeros(3), data["primitive_scaled"]
            )
            alphas[w] = {
                "species": self.analysis["Wyckoff_sites"][w]["species"],
                "basis_sites": list(
                    self.analysis["Wyckoff_sites"][w]["basis_sites"]
                ),
                "shells": {},
            }
            for shell, shell_data in sorted(data["shells"].items()):
                a = shell_data["reference_bond_vector"]
                far_wyckoff = self.basis_sites[
                    shell_data["target_basis_site"]
                ]["Wyckoff_site"]
                far, far_scaled = self._resolve(
                    data["primitive_scaled"], far_wyckoff, a
                )
                atoms = list(origin_atoms)
                if far_wyckoff != w:
                    for b, p, s in self._neighborhood(far, a, far_scaled):
                        if not any(
                            b == b2 and np.linalg.norm(p - p2) < tol
                            for b2, p2, _ in atoms
                        ):
                            atoms.append((b, p, s))
                bonds = [
                    bond
                    for b, p, s in atoms
                    for bond in self._bonds_from(b, p, s)
                ]
                weights, pair_ids = self._bond_weights(bonds, tol)

                ref = {
                    "origin": bs,
                    "target": far,
                    "vector": a,
                    "offset": np.zeros(3),
                }
                alpha = self._covariances(ref, bonds) @ np.linalg.inv(
                    self._covariances(ref, [ref])[0]
                )
                targets = [
                    {
                        **bond,
                        "alpha": alpha[i],
                        "weight": weights[i],
                        "pair_id": pair_ids[i],
                    }
                    for i, bond in enumerate(bonds)
                    if np.linalg.norm(alpha[i]) >= alpha_threshold
                ]
                own = [
                    t
                    for t in targets
                    if t["origin"] == bs
                    and not t["offset"].any()
                    and np.allclose(t["vector"], a)
                ]
                if len(own) != 1 or not np.allclose(
                    own[0]["alpha"], np.eye(3), atol=1e-6
                ):
                    raise RuntimeError(
                        f"(Wyckoff site {w}, shell {shell}): the reference "
                        "bond's own alpha is "
                        "not the identity; the q-mesh or the Hessian is "
                        "inconsistent."
                    )
                alphas[w]["shells"][shell] = {
                    "reference_bond_vector": a,
                    "targets": [
                        {
                            "target_bond_vector": t["vector"],
                            "target_species": t["target_species"],
                            "alpha": t["alpha"],
                            "rotation": t["rotation"],
                            "potential_key": t["potential_key"],
                            "weight": float(t["weight"]),
                        }
                        for t in self._collapse(targets)
                    ],
                }
        return alphas

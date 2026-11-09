"""
The generalized mean-field bond model: effective potentials, bond densities and
the NVT/NPT self-consistency conditions, for crystals with any number of
Wyckoff sites and species.

Shells that carry one and the same bond density form a class and are solved
once: site-symmetry orbits within a Wyckoff site, and the two ends of a bond
joining two Wyckoff sites (whose bond-centred effective potentials coincide,
V_eff,m(d) = V_eff,rep(Q d)).

Each class is solved on a grid fixed in thermal units, d = L xi with
xi in [-1, 1]^3, whose axes are the principal axes of the Hessian of V_eff at
the strained reference bond and whose half-widths are
buffer * n_sigma * sqrt(k_B T / curvature). V_eff is evaluated on a coarse
grid and splined onto a fine one.

One effective temperature T_eff holds for the whole crystal, fixed by
sum_kappa c_kappa T_vir,kappa(T_eff) = T.
"""

import warnings

import numpy as np
from scipy.constants import eV, physical_constants
from scipy.interpolate import make_interp_spline
from scipy.optimize import root_scalar

__all__ = [
    "BondClasses",
    "MeanFieldBondModel",
    "build_bonding_potentials_by_bond_type",
    "generate_bond_grids",
]

KB = physical_constants["Boltzmann constant in eV/K"][0]
BAR = eV / 1e-30 / 1e5


def build_bonding_potentials_by_bond_type(
    symmetry_analysis, bond_type_map, params_by_bond_type
):
    """
    {Wyckoff_site: {"species", "basis_sites", "shells": {shell: potential}}}
    from one potential per bond type (a callable, or Morse parameters {"D_e",
    "a", "r_e"}), as in NeighborBondSymmetryAnalysis.assign_bond_types().
    """
    from .morse import MorsePotential

    shared = {
        t: p if callable(p) else MorsePotential(**p)
        for t, p in params_by_bond_type.items()
    }
    out = {}
    for w, wd in symmetry_analysis["Wyckoff_sites"].items():
        bs = wd["basis_sites"][0]
        out[w] = {
            "species": wd["species"],
            "basis_sites": list(wd["basis_sites"]),
            "shells": {
                s: shared[bond_type_map[(bs, s)]]
                for s in symmetry_analysis["basis_sites"][bs]["shells"]
            },
        }
    return out


def generate_bond_grids(symmetry_analysis, displacement, n_fine):
    """
    Fixed output grids: {Wyckoff_site: {shell: {"reference_bond_vector",
    "fine", "fine_displacements", "fine_axes"}}}, a cube of displacements in
    [-displacement, displacement] about the unstrained reference bond, shared
    by the model's bond densities and the MD histograms.
    """
    axis = np.linspace(-displacement, displacement, n_fine)
    cube = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)
    grids = {}
    for w, wd in symmetry_analysis["Wyckoff_sites"].items():
        grids[w] = {}
        for s, sd in symmetry_analysis["basis_sites"][wd["basis_sites"][0]][
            "shells"
        ].items():
            a = np.asarray(sd["reference_bond_vector"], float)
            grids[w][s] = {
                "reference_bond_vector": a,
                "fine": a + cube,
                "fine_displacements": cube,
                "fine_axes": (axis, axis, axis),
            }
    return grids


def _unit_cube(n):
    x = np.linspace(-1.0, 1.0, n)
    return np.stack(np.meshgrid(x, x, x, indexing="ij"), axis=-1).reshape(
        -1, 3
    )


def _spline_matrix(n_coarse, n_fine):
    """Cubic spline interpolation from n_coarse to n_fine evenly spaced points
    on [-1, 1], as a matrix."""
    return make_interp_spline(
        np.linspace(-1, 1, n_coarse), np.eye(n_coarse), k=3
    )(np.linspace(-1, 1, n_fine))


def _interpolate(A, coarse):
    n = A.shape[1]
    return np.einsum(
        "ai,bj,ck,ijk->abc", A, A, A, coarse.reshape(n, n, n), optimize=True
    ).ravel()


class _Converged(Exception):
    def __init__(self, x):
        self.x = x


def _find_root(
    f,
    x0,
    step,
    domain=(None, None),
    rtol=1e-8,
    f_atol=0.0,
    factor=1.5,
    max_iter=60,
    overshoot=1.25,
    max_secant=8,
):
    """
    Bracket the root by secant steps that overshoot it, falling back to
    expanding symmetrically about x0, then refine with toms748.
    """
    lo = -np.inf if domain[0] is None else domain[0]
    hi = np.inf if domain[1] is None else domain[1]

    def refine(a, b):
        def g(x):
            fx = f(x)
            if abs(fx) <= f_atol:
                raise _Converged(x)
            return fx

        try:
            return root_scalar(
                g, bracket=sorted((a, b)), method="toms748", rtol=rtol
            ).root
        except _Converged as done:
            return done.x

    f0 = f(x0)
    if abs(f0) <= f_atol:
        return x0
    xa = x0
    fa = f0
    xb = x0 + step if x0 + step <= hi else max(x0 - step, lo)
    for _ in range(max_secant):
        fb = f(xb)
        if abs(fb) <= f_atol:
            return xb
        if fa * fb < 0:
            return refine(xa, xb)
        slope = (fb - fa) / (xb - xa)
        if not np.isfinite(slope) or slope == 0.0:
            break
        x = min(max(xb - overshoot * fb / slope, lo), hi)
        if x == xb:
            break
        xa = xb
        fa = fb
        xb = x
    for _ in range(max_iter):
        for x in (min(x0 + step, hi), max(x0 - step, lo)):
            fx = f(x)
            if abs(fx) <= f_atol:
                return x
            if f0 * fx < 0:
                return refine(x0, x)
        step *= factor
    raise RuntimeError(f"No sign change of the residual found around {x0}.")


class BondClasses:
    """
    Partition of the representative shells (Wyckoff_site, shell) into classes
    sharing one bond density.

    Attributes
    ----------
    members : dict
        {class: [(Wyckoff_site, shell), ...]}, a class named by its first
        member.
    of : dict
        {(Wyckoff_site, shell): (class, Q)} with V_eff,member(d) =
        V_eff,class(Q d) and Q a_member = a_class.
    """

    def __init__(self, symmetry_analysis, tol=1e-3):
        basis_sites = symmetry_analysis["basis_sites"]
        representative = {
            w: d["basis_sites"][0]
            for w, d in symmetry_analysis["Wyckoff_sites"].items()
        }
        edges = {}

        def link(m, n, Q):
            edges.setdefault(m, []).append((n, Q))
            edges.setdefault(n, []).append((m, Q.T))

        for w, bs in representative.items():
            data = basis_sites[bs]
            rotations = np.asarray(data["rotations"], float)
            for s, sd in data["shells"].items():
                a = np.asarray(sd["reference_bond_vector"], float)
                for t, td in data["shells"].items():
                    hit = np.flatnonzero(
                        np.linalg.norm(
                            rotations @ a - td["reference_bond_vector"],
                            axis=-1,
                        )
                        < tol
                    )
                    if t > s and hit.size:
                        link((w, t), (w, s), rotations[hit[0]].T)
                far = basis_sites[sd["target_basis_site"]]
                if far["Wyckoff_site"] == w:
                    continue
                far_rep = basis_sites[representative[far["Wyckoff_site"]]]
                far_rotations = far["coset_rotation"] @ np.asarray(
                    far_rep["rotations"], float
                )
                for t, td in far_rep["shells"].items():
                    hit = np.flatnonzero(
                        np.linalg.norm(
                            far_rotations @ td["reference_bond_vector"] + a,
                            axis=-1,
                        )
                        < tol
                    )
                    if hit.size:
                        link(
                            (far["Wyckoff_site"], t),
                            (w, s),
                            -far_rotations[hit[0]],
                        )
                        break

        self.members = {}
        self.of = {}
        for w, bs in sorted(representative.items()):
            for s in sorted(basis_sites[bs]["shells"]):
                if (w, s) in self.of:
                    continue
                self.of[(w, s)] = ((w, s), np.eye(3))
                self.members[(w, s)] = [(w, s)]
                stack = [(w, s)]
                while stack:
                    node = stack.pop()
                    for other, Q in edges.get(node, []):
                        if other not in self.of:
                            self.of[other] = ((w, s), self.of[node][1] @ Q.T)
                            self.members[(w, s)].append(other)
                            stack.append(other)


class MeanFieldBondModel:
    """
    Parameters
    ----------
    structure : ase.Atoms
        The supercell the symmetry analysis was built on.
    bonding_potentials : dict
        {Wyckoff_site: {"shells": {shell: potential}}}; a potential takes bond
        vectors (..., 3) and has a .gradient().
    symmetry_analysis, alphas : dict
        From NeighborBondSymmetryAnalysis.analyze() and
        BondPairCovarianceCalculator.compute_alphas().
    n_coarse, n_fine : int, optional
        Points per axis on which V_eff is evaluated and onto which it is
        splined. Defaults to 15, 41.
    use_coarse : bool, optional
        Evaluate V_eff on the coarse grid and spline it (default), or directly
        on the fine grid.
    n_sigma, buffer : float, optional
        Grid half-width in thermal widths, and the extra factor on it. Defaults
        to 6, 1.1.
    max_displacement : float, optional
        Cap on the grid half-width in Angstrom. Defaults to half the reference
        bond length.
    displacement, n_output : float, int, optional
        Fixed output grid (see generate_bond_grids) on which bond densities are
        reported when store_shell_details is set. Defaults to None (no
        densities) and 51.
    """

    def __init__(
        self,
        structure,
        bonding_potentials,
        symmetry_analysis,
        alphas,
        n_coarse=15,
        n_fine=41,
        use_coarse=True,
        n_sigma=6.0,
        buffer=1.1,
        max_displacement=None,
        displacement=None,
        n_output=51,
        max_batch=4_000_000,
    ):
        self.potentials = bonding_potentials
        self.n_atoms = len(structure)
        self.V_0 = structure.get_volume()
        self.n_sigma = n_sigma
        self.buffer = buffer
        self.max_displacement = max_displacement
        self.max_batch = max_batch

        wyckoff_sites = symmetry_analysis["Wyckoff_sites"]
        self.sites = {
            w: {
                "species": d["species"],
                "atomic_fraction": d["atomic_fraction"],
            }
            for w, d in wyckoff_sites.items()
        }
        self.shells = {}
        for w, d in wyckoff_sites.items():
            for s, sd in symmetry_analysis["basis_sites"][d["basis_sites"][0]][
                "shells"
            ].items():
                self.shells[(w, s)] = {
                    "a": np.asarray(sd["reference_bond_vector"], float),
                    "n_bonds": int(sd["n_bonds"]),
                    "target_species": sd["target_species"],
                }
        self.classes = BondClasses(symmetry_analysis)
        self._check_class_potentials()
        self.groups = {
            c: self._target_groups(alphas, c) for c in self.classes.members
        }

        self.use_coarse = use_coarse
        self.coarse = _unit_cube(n_coarse)
        self.fine = _unit_cube(n_fine)
        self.A = _spline_matrix(n_coarse, n_fine)
        self.edge = (np.abs(self.fine) == 1.0).any(axis=1)

        self.bond_grids = None
        if displacement is not None:
            self.bond_grids = generate_bond_grids(
                symmetry_analysis, displacement, n_output
            )
            self.displacement = displacement
            self.A_output = _spline_matrix(n_coarse, n_output)

        self._hessians = {}
        self._grids = {}
        self._grid_key = None

    def _check_class_potentials(self):
        for (w, s), (c, Q) in self.classes.of.items():
            b = np.linspace(0.6, 1.6, 32)[:, None] * self.shells[(w, s)]["a"]
            if not np.allclose(
                self.potentials[w]["shells"][s](b),
                self.potentials[c[0]]["shells"][c[1]](b @ Q.T),
            ):
                raise ValueError(
                    f"Shell {(w, s)} and shell {c} hold the same bonds but "
                    "carry different potentials."
                )

    def _target_groups(self, alphas, c):
        """Targets grouped by potential: [(potential, a, M with M_t = alpha_t^T
        R_t, weights)]."""
        groups = {}
        for t in alphas[c[0]]["shells"][c[1]]["targets"]:
            M, weights = groups.setdefault(t["potential_key"], ([], []))
            M.append(
                np.asarray(t["alpha"], float).T
                @ np.asarray(t["rotation"], float)
            )
            weights.append(t["weight"])
        return [
            (
                self.potentials[w]["shells"][s],
                self.shells[(w, s)]["a"],
                np.array(M),
                np.array(weights),
            )
            for (w, s), (M, weights) in groups.items()
        ]

    def effective_potential(self, c, d, eps):
        """V_eff of class c at displacements d (n, 3) about the strained
        reference bond eps * a."""
        v = np.zeros(len(d))
        chunk = max(1, self.max_batch // max(len(d), 1))
        for potential, a, M, weights in self.groups[c]:
            for i in range(0, len(M), chunk):
                v += weights[i : i + chunk] @ potential(
                    d @ M[i : i + chunk] + eps * a
                )
        return v

    def _hessian(self, c, eps, h=2e-3):
        """Hessian of V_eff at d = 0 by central differences, cached per eps."""
        if (c, eps) not in self._hessians:
            e = np.eye(3) * h
            pairs = [(i, j) for i in range(3) for j in range(i + 1, 3)]
            points = [np.zeros(3)] + [
                e[i] * sign for i in range(3) for sign in (1, -1)
            ]
            points += [
                e[i] * si + e[j] * sj
                for i, j in pairs
                for si in (1, -1)
                for sj in (1, -1)
            ]
            v = self.effective_potential(c, np.array(points), eps)
            H = np.diag(
                [
                    (v[1 + 2 * i] + v[2 + 2 * i] - 2 * v[0]) / h**2
                    for i in range(3)
                ]
            )
            for n, (i, j) in enumerate(pairs):
                pp, pm, mp, mm = v[7 + 4 * n : 11 + 4 * n]
                H[i, j] = (pp - pm - mp + mm) / (4 * h**2)
                H[j, i] = H[i, j]
            self._hessians[(c, eps)] = H
        return self._hessians[(c, eps)]

    def grid_axes(self, c, temperature, eps, curvature_floor=1e-3):
        """L, whose columns are the half-width vectors of class c's thermal
        grid."""
        curvature, axes = np.linalg.eigh(self._hessian(c, eps))
        half = (
            self.buffer
            * self.n_sigma
            * np.sqrt(
                KB * temperature / np.maximum(curvature, curvature_floor)
            )
        )
        cap = self.max_displacement or 0.5 * np.linalg.norm(
            self.shells[c]["a"]
        )
        return axes * np.minimum(half, cap)

    def _grid(self, c, temperature, eps):
        """Integrands of class c on its fine grid, cached for the current
        (temperature, eps)."""
        if self._grid_key != (temperature, eps):
            self._grids = {}
            self._grid_key = (temperature, eps)
        if c not in self._grids:
            L = self.grid_axes(c, temperature, eps)
            d = self.fine @ L.T
            veff = (
                _interpolate(
                    self.A, self.effective_potential(c, self.coarse @ L.T, eps)
                )
                if self.use_coarse
                else self.effective_potential(c, d, eps)
            )
            a = self.shells[c]["a"]
            b = eps * a + d
            potential = self.potentials[c[0]]["shells"][c[1]]
            gradient = potential.gradient(b)
            self._grids[c] = {
                "veff": veff - veff.min(),
                "u": b @ (a / np.linalg.norm(a)),
                "d_grad": np.einsum("pi,pi->p", d, gradient),
                "b_grad": np.einsum("pi,pi->p", b, gradient),
                "v": potential(b),
            }
        return self._grids[c]

    def _solve_lambda(self, c, T_eff, temperature, eps, lm, max_iter=50):
        """Lagrange multiplier fixing <b . a_hat> = eps |a|, by Newton with a
        bracketed fallback."""
        g = self._grid(c, temperature, eps)
        beta = 1.0 / (KB * T_eff)
        target = np.linalg.norm(self.shells[c]["a"]) * eps

        def density(x):
            e = -beta * (g["veff"] + x * g["u"])
            p = np.exp(e - e.max())
            return p / p.sum()

        x = lm
        for _ in range(max_iter):
            p = density(x)
            mean = p @ g["u"]
            if abs(mean - target) <= 1e-12 * target:
                return x, p
            variance = p @ g["u"] ** 2 - mean**2
            if not variance > 0:
                break
            x += (mean - target) / (beta * variance)
        x = _find_root(
            lambda y: density(y) @ g["u"] - target, lm, 1e-2, rtol=1e-12
        )
        return x, density(x)

    def _solve(self, T_eff, temperature, eps, state):
        """Per-class Lagrange multiplier and bond averages at one T_eff."""
        lms = state.setdefault("lms", {})
        out = {}
        for c in self.classes.members:
            lms[c], p = self._solve_lambda(
                c, T_eff, temperature, eps, lms.get(c, 0.0)
            )
            g = self._grid(c, temperature, eps)
            out[c] = {
                "lagrange_multiplier": lms[c],
                "d_grad": p @ g["d_grad"],
                "b_grad": p @ g["b_grad"],
                "v": p @ g["v"],
                "edge_mass": p[self.edge].sum(),
            }
        return out

    def _crystal_T_vir(self, averages):
        return sum(
            self.sites[w]["atomic_fraction"]
            * sh["n_bonds"]
            * averages[self.classes.of[(w, s)][0]]["d_grad"]
            for (w, s), sh in self.shells.items()
        ) / (6 * KB)

    def bond_density(self, w, s, T_eff, lm, eps):
        """Bond density of one shell on its output grid, normalized to sum to
        1."""
        c, Q = self.classes.of[(w, s)]
        a = self.shells[(w, s)]["a"]
        grid = self.bond_grids[w][s]
        if self.use_coarse:
            D = self.displacement * self.coarse - (eps - 1.0) * a
            veff = _interpolate(
                self.A_output, self.effective_potential(c, D @ Q.T, eps)
            )
        else:
            D = grid["fine_displacements"].reshape(-1, 3) - (eps - 1.0) * a
            veff = self.effective_potential(c, D @ Q.T, eps)
        u = grid["fine"].reshape(-1, 3) @ (a / np.linalg.norm(a))
        e = -(veff + lm * u) / (KB * T_eff)
        p = np.exp(e - e.max())
        return (p / p.sum()).reshape(grid["fine"].shape[:-1])

    def _nvt(self, temperature, eps, state, store_shell_details):
        cache = {}

        def residual(T_eff):
            if T_eff not in cache:
                cache[T_eff] = (
                    self._crystal_T_vir(
                        self._solve(T_eff, temperature, eps, state)
                    )
                    - temperature
                )
            return cache[T_eff]

        x0 = state.get("T_eff_ratio", 1.0) * temperature
        T_eff = _find_root(
            residual,
            x0,
            0.05 * temperature,
            domain=(max(1e-6, 1e-4 * temperature), None),
            rtol=1e-6,
            f_atol=1e-6 * max(temperature, 1.0),
        )
        state["T_eff_ratio"] = T_eff / temperature
        averages = self._solve(T_eff, temperature, eps, state)

        N_by_V = self.n_atoms / (self.V_0 * eps**3)
        sites = {
            w: {**d, "T_vir": 0.0, "P": 0.0, "U_ah": 0.0, "shells": {}}
            for w, d in self.sites.items()
        }
        edge_mass = 0.0
        for (w, s), sh in self.shells.items():
            r = averages[self.classes.of[(w, s)][0]]
            m = sh["n_bonds"]
            T_vir = m * r["d_grad"] / (6 * KB)
            shell = {
                "T_vir": T_vir,
                "P": N_by_V * (KB * T_vir - m / 6 * r["b_grad"]) * BAR,
                "U_ah": m / 2 * r["v"] - 1.5 * KB * T_vir,
            }
            for key in shell:
                sites[w][key] += shell[key]
            edge_mass = max(edge_mass, r["edge_mass"])
            if store_shell_details:
                shell.update(
                    lagrange_multiplier=r["lagrange_multiplier"],
                    edge_mass=r["edge_mass"],
                    target_species=sh["target_species"],
                    bond_class=self.classes.of[(w, s)][0],
                )
                if self.bond_grids is not None:
                    shell["bond_density"] = self.bond_density(
                        w, s, T_eff, r["lagrange_multiplier"], eps
                    )
                sites[w]["shells"][s] = shell
        if edge_mass > 1e-6:
            warnings.warn(
                f"Up to {edge_mass:.1e} of a bond density lies on its grid "
                f"edge at T={temperature} K; "
                "raise n_sigma or max_displacement.",
                RuntimeWarning,
                stacklevel=3,
            )
        for site in sites.values():
            if not store_shell_details:
                del site["shells"]

        def crystal(key):
            return sum(d["atomic_fraction"] * d[key] for d in sites.values())

        return {
            "temperature": temperature,
            "eps": eps,
            "T_eff": T_eff,
            "T_vir": crystal("T_vir"),
            "P": crystal("P"),
            "U_ah": crystal("U_ah"),
            "Wyckoff_sites": sites,
        }

    def _npt(
        self, temperature, pressure, state, store_shell_details, eps_bounds
    ):
        results = {}

        def residual(eps):
            if eps not in results:
                results[eps] = self._nvt(
                    temperature, eps, state, store_shell_details=False
                )
            return results[eps]["P"] - pressure

        history = state.setdefault("eps_history", [])
        x0 = history[-1][1] if history else 1.0
        if len(history) > 1 and history[-1][0] != history[-2][0]:
            (T1, e1), (T2, e2) = history[-2:]
            x0 = float(
                np.clip(
                    e2 + (e2 - e1) * (temperature - T2) / (T2 - T1),
                    *eps_bounds,
                )
            )
        step = min(2e-3, 1e-4 * np.sqrt(max(temperature, 1.0)))
        eps = _find_root(
            residual, x0, step, domain=eps_bounds, rtol=1e-6, f_atol=1e-2
        )
        history.append((temperature, eps))
        if store_shell_details or eps not in results:
            return self._nvt(temperature, eps, state, store_shell_details)
        return results[eps]

    def _static_energy(self, eps):
        return sum(
            self.sites[w]["atomic_fraction"]
            * sh["n_bonds"]
            / 2
            * float(self.potentials[w]["shells"][s](eps * sh["a"]))
            for (w, s), sh in self.shells.items()
        )

    @staticmethod
    def _add_free_energy(results, offset=0.0):
        """
        F_ah = -T int_0^T U_ah(T') / T'^2 dT' (Eq. 9 of the manuscript) over
        the swept temperatures: trapezoids between them, and U_ah / T^2
        extrapolated linearly from the lowest two down to T = 0.
        """
        order = np.argsort([r["temperature"] for r in results])
        T = np.array([results[i]["temperature"] for i in order])
        g = np.array([(results[i]["U_ah"] - offset) for i in order]) / T**2
        g0 = g[0] - T[0] * (g[1] - g[0]) / (T[1] - T[0])
        integral = np.cumsum(
            np.concatenate(
                [
                    [0.5 * T[0] * (g0 + g[0])],
                    0.5 * np.diff(T) * (g[1:] + g[:-1]),
                ]
            )
        )
        for i, value in zip(order, integral):
            results[i]["F_ah"] = float(-results[i]["temperature"] * value)
        return results

    def bond_data(self, result):
        """
        (bond_grids, bond_densities) of a single-temperature run_nvt / run_npt
        result, in the same layout as MDBondAnalysis.compute_bond_densities.
        """
        if self.bond_grids is None:
            raise ValueError(
                "Build MeanFieldBondModel with displacement= and n_output= "
                "to get bond densities."
            )
        densities = {
            w: {
                s: shell["bond_density"] for s, shell in site["shells"].items()
            }
            for w, site in result["Wyckoff_sites"].items()
        }
        return self.bond_grids, densities

    def run_nvt(self, temperatures, eps=1.0, store_shell_details=True):
        """
        Self-consistent NVT solve at fixed isotropic strain eps, for one
        temperature (K) or a sweep.

        Returns
        -------
        result : dict, or a list of them for a sweep
            {"temperature", "eps", "T_eff", "T_vir", "P" (bar), "U_ah"
            (eV/atom), "F_ah" (eV/atom, sweeps only), "Wyckoff_sites": {w:
            {"species", "atomic_fraction", "T_vir", "P", "U_ah", "shells": {s:
            {"T_vir", "P", "U_ah", "lagrange_multiplier", "edge_mass",
            "target_species", "bond_class", "bond_density"}}}}}. Crystal values
            are atomic-fraction-weighted; site values are per atom of that
            site; "shells" only with store_shell_details, "bond_density" only
            with an output grid. "U_ah" includes the static lattice energy at
            eps, which F_ah, as in Eq. 9, excludes.
        """
        state = {}
        if np.isscalar(temperatures):
            return self._nvt(
                float(temperatures), eps, state, store_shell_details
            )
        results = [
            self._nvt(float(T), eps, state, store_shell_details)
            for T in temperatures
        ]
        return (
            self._add_free_energy(results, self._static_energy(eps))
            if len(results) > 1
            else results
        )

    def run_npt(
        self,
        temperatures,
        pressure=0.0,
        store_shell_details=True,
        eps_bounds=(0.8, 1.3),
    ):
        """
        Self-consistent NPT solve at pressure (bar); the result of run_nvt at
        the strain found. For a sweep, "F_ah" integrates "U_ah" along the swept
        path, eps(T) included.
        """
        state = {}
        if np.isscalar(temperatures):
            return self._npt(
                float(temperatures),
                pressure,
                state,
                store_shell_details,
                eps_bounds,
            )
        results = [
            self._npt(
                float(T), pressure, state, store_shell_details, eps_bounds
            )
            for T in temperatures
        ]
        return self._add_free_energy(results) if len(results) > 1 else results

"""
Bond-vector statistics and densities from MD trajectories, per Wyckoff site and
shell, on the same output grids as the mean-field bond densities
(generate_bond_grids).

Every bond is centred on its unstrained reference direction and rotated into
the frame of its Wyckoff representative's reference bond, so all
symmetry-equivalent bonds pool into one sample.
"""

import numpy as np

from .mean_field import generate_bond_grids
from .symmetry import NeighborBondSymmetryAnalysis, wyckoff_shell_map

__all__ = ["MDBondAnalysis", "load_md_bond_data", "save_md_bond_data"]


class MDBondAnalysis:
    """
    Parameters
    ----------
    reference_structure : ase.Atoms
        The relaxed structure the MD run started from.
    n_shells, cutoff_radius : optional
        As in NeighborBondSymmetryAnalysis.
    """

    def __init__(
        self, reference_structure, n_shells=None, cutoff_radius=None, tol=1e-3
    ):
        self.analyzer = NeighborBondSymmetryAnalysis(
            reference_structure, n_shells, cutoff_radius, tol=tol
        )
        self.analysis = self.analyzer.analyze()
        self.directed_bonds = self.analyzer.classify_directed_bonds()
        self.reference_structure = reference_structure
        self.tol = tol

    def compute_bond_vectors(self, positions, cells, pool_basis_sites=False):
        """
        Parameters
        ----------
        positions : np.ndarray, shape (n_snapshots, n_atoms, 3)
            Cartesian positions, same atoms and order as reference_structure.
        cells : np.ndarray, shape (n_snapshots, 3, 3)
        pool_basis_sites : bool, optional
            Pool the bonds of every basis site of a Wyckoff site, not only of
            its representative.

        Returns
        -------
        bond_vectors : dict
            {Wyckoff_site: {shell: np.ndarray (n_samples, 3)}}, fluctuations
            b - a in the representative's frame.
        """
        positions = np.asarray(positions, float)
        cells = np.asarray(cells, float)
        basis_sites = self.analysis["basis_sites"]
        wyckoff_of = {bs: d["Wyckoff_site"] for bs, d in basis_sites.items()}
        representative = {
            w: d["basis_sites"][0]
            for w, d in self.analysis["Wyckoff_sites"].items()
        }
        shell_maps = {
            bs: wyckoff_shell_map(d, wyckoff_of)
            for bs, d in basis_sites.items()
        }
        key_of = {
            bs: {s: k for k, s in m.items()} for bs, m in shell_maps.items()
        }
        reference_positions = self.reference_structure.get_positions()
        reference_cell = self.reference_structure.get_cell().array

        pairs = {}
        for origin, target, bs, s in self.directed_bonds:
            pairs.setdefault((bs, s), []).append((origin, target))

        out = {}
        for (bs, s), pair in pairs.items():
            w = wyckoff_of[bs]
            if not pool_basis_sites and bs != representative[w]:
                continue
            i, j = np.array(pair).T
            rep = basis_sites[representative[w]]
            s_rep = shell_maps[representative[w]][key_of[bs][s]]
            shell = rep["shells"][s_rep]
            rotations = (
                basis_sites[bs]["coset_rotation"]
                @ rep["rotations"][shell["rotation_ids"]]
            )
            directions = rotations @ shell["reference_bond_vector"]

            frac = (
                reference_positions[j] - reference_positions[i]
            ) @ np.linalg.inv(reference_cell)
            equilibrium = (frac - np.round(frac)) @ reference_cell
            distances = np.linalg.norm(
                equilibrium[:, None] - directions[None], axis=-1
            )
            match = distances.argmin(axis=1)
            if distances[np.arange(len(match)), match].max() > self.tol:
                raise RuntimeError(
                    f"A bond of (basis site {bs}, shell {s}) matches no "
                    "reference direction."
                )

            frac = np.einsum(
                "tpi,tij->tpj",
                positions[:, j] - positions[:, i],
                np.linalg.inv(cells),
            )
            bonds = np.einsum("tpi,tij->tpj", frac - np.round(frac), cells)
            fluctuation = np.einsum(
                "pji,tpj->tpi", rotations[match], bonds - directions[match]
            )
            bucket = out.setdefault(w, {})
            bucket[s_rep] = np.concatenate(
                [
                    bucket.get(s_rep, np.empty((0, 3))),
                    fluctuation.reshape(-1, 3),
                ]
            )
        return out

    def compute_bond_densities(
        self, positions, cells, displacement, n_fine, pool_basis_sites=True
    ):
        """
        Histograms of compute_bond_vectors on generate_bond_grids(displacement,
        n_fine), normalized to sum to 1. Returns (bond_grids, {Wyckoff_site:
        {shell: density}}).
        """
        vectors = self.compute_bond_vectors(positions, cells, pool_basis_sites)
        grids = generate_bond_grids(self.analysis, displacement, n_fine)
        centers = np.linspace(-displacement, displacement, n_fine)
        half = (centers[1] - centers[0]) / 2
        edges = np.append(centers - half, centers[-1] + half)
        densities = {}
        for w, shells in grids.items():
            densities[w] = {}
            for s in shells:
                counts, _ = np.histogramdd(
                    vectors.get(w, {}).get(s, np.empty((0, 3))),
                    bins=(edges,) * 3,
                )
                densities[w][s] = counts / max(counts.sum(), 1)
        return grids, densities


def save_md_bond_data(bond_grids, bond_densities, path):
    arrays = {
        "pairs": np.array([(w, s) for w in bond_grids for s in bond_grids[w]])
    }
    for w, s in arrays["pairs"].tolist():
        arrays[f"reference_bond_vector_{w}_{s}"] = bond_grids[w][s][
            "reference_bond_vector"
        ]
        arrays[f"fine_displacements_{w}_{s}"] = bond_grids[w][s][
            "fine_displacements"
        ]
        arrays[f"density_{w}_{s}"] = bond_densities[w][s]
    np.savez_compressed(path, **arrays)


def load_md_bond_data(path):
    """(bond_grids, bond_densities) as written by save_md_bond_data."""
    grids = {}
    densities = {}
    with np.load(path) as data:
        for w, s in data["pairs"].tolist():
            a = data[f"reference_bond_vector_{w}_{s}"]
            cube = data[f"fine_displacements_{w}_{s}"]
            axis = cube[:, 0, 0, 0]
            grids.setdefault(w, {})[s] = {
                "reference_bond_vector": a,
                "fine": a + cube,
                "fine_displacements": cube,
                "fine_axes": (axis, axis, axis),
            }
            densities.setdefault(w, {})[s] = data[f"density_{w}_{s}"]
    return grids, densities

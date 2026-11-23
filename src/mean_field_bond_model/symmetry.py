"""
Neighbor-bond shells of a periodic crystal and the site-symmetry operations
that generate every bond of a shell from one reference bond.

A basis site is one atom of the primitive cell; a Wyckoff site groups the basis
sites spglib finds equivalent. A shell is one site-symmetry orbit of bonds from
a basis site to one target basis site.
"""

import numpy as np
from matscipy.neighbours import neighbour_list
from spglib import get_symmetry_dataset, standardize_cell

__all__ = [
    "NeighborBondSymmetryAnalysis",
    "generate_bond_vectors",
    "get_primitive_cell",
    "pair_shells_by_reverse_bond",
    "wyckoff_shell_map",
]


def get_primitive_cell(structure, symprec=1e-5):
    """Primitive (lattice, scaled_positions, numbers), kept in the input
    Cartesian frame."""
    cell = (
        structure.get_cell().array,
        structure.get_scaled_positions(),
        structure.get_atomic_numbers(),
    )
    primitive = standardize_cell(
        cell, to_primitive=True, no_idealize=True, symprec=symprec
    )
    if primitive is None:
        raise RuntimeError("spglib found no primitive cell; loosen symprec.")
    return primitive


def generate_bond_vectors(symmetry_analysis, basis_site, shell):
    """All bond vectors of one shell, R_k @ reference_bond_vector."""
    data = symmetry_analysis["basis_sites"][basis_site]
    shell_data = data["shells"][shell]
    return (
        data["rotations"][shell_data["rotation_ids"]]
        @ shell_data["reference_bond_vector"]
    )


def pair_shells_by_reverse_bond(symmetry_analysis, tol=1e-3):
    """
    {(basis_site, shell): (target_basis_site, target_shell)}, the shell holding
    the same physical bonds read in the opposite direction.
    """
    basis_sites = symmetry_analysis["basis_sites"]
    paired = {}
    for basis_site, data in basis_sites.items():
        for shell, shell_data in data["shells"].items():
            target = shell_data["target_basis_site"]
            reverse = -generate_bond_vectors(
                symmetry_analysis, basis_site, shell
            )
            best = tol
            match = None
            for candidate, candidate_data in basis_sites[target][
                "shells"
            ].items():
                if candidate_data["target_basis_site"] != basis_site:
                    continue
                bonds = generate_bond_vectors(
                    symmetry_analysis, target, candidate
                )
                distance = np.linalg.norm(
                    bonds[None] - reverse[:, None], axis=-1
                ).min()
                if distance < best:
                    best = distance
                    match = candidate
            if match is None:
                raise RuntimeError(
                    f"No reverse shell for (basis site {basis_site}, shell "
                    f"{shell}); the cutoff cuts "
                    "a shell asymmetrically."
                )
            paired[(basis_site, shell)] = (target, match)
    return paired


def wyckoff_shell_map(basis_site_data, wyckoff_of_basis_site, decimals=4):
    """
    {(bond length, target Wyckoff site, ordinal): shell} for one basis site, so
    that a sibling basis site's shell can be matched to its Wyckoff
    representative's shell numbering.
    """
    shell_of_key = {}
    seen = {}
    for shell in sorted(basis_site_data["shells"]):
        shell_data = basis_site_data["shells"][shell]
        key = (
            round(float(shell_data["reference_bond_length"]), decimals),
            wyckoff_of_basis_site[shell_data["target_basis_site"]],
        )
        seen[key] = seen.get(key, -1) + 1
        shell_of_key[key + (seen[key],)] = shell
    return shell_of_key


def _wrap(x):
    return x - np.floor(x + 1e-8)


def _cartesian_operations(prim_lattice, rotations):
    return prim_lattice.T @ rotations @ np.linalg.inv(prim_lattice).T


class NeighborBondSymmetryAnalysis:
    """
    Parameters
    ----------
    structure : ase.Atoms
        Relaxed, fully periodic structure.
    n_shells : int, optional
        Number of distance shells; sets the cutoff.
    cutoff_radius : float, optional
        Takes precedence over n_shells.
    n_extra_origin_shells : int, optional
        Extra distance shells resolved only to widen the clusters of the
        covariance calculation; they carry no bonds. Defaults to 0.
    """

    def __init__(
        self,
        structure,
        n_shells=None,
        cutoff_radius=None,
        symprec=1e-5,
        decimals=4,
        tol=1e-3,
        n_extra_origin_shells=0,
    ):
        if n_shells is None and cutoff_radius is None:
            raise ValueError("Give n_shells or cutoff_radius.")
        self.structure = structure
        self.symprec = symprec
        self.decimals = decimals
        self.tol = tol
        self.analysis = {}
        self.paired_shells = None
        self.bond_type_map = None
        self.lammps_bonds = None
        self._neighbors = None
        self._assign_sites()

        epsilon = 5 * 10 ** (-decimals)
        if cutoff_radius is not None:
            lengths, _ = self._bond_lengths(float(cutoff_radius) + epsilon)
            n_shells = len(lengths)
            self.cutoff_radius = float(cutoff_radius)
        else:
            self.cutoff_radius = self._cutoff_for(n_shells)
        self.n_shells = n_shells
        self.bonded_cutoff_radius = self.cutoff_radius
        if n_extra_origin_shells:
            self.cutoff_radius = self._cutoff_for(
                n_shells + n_extra_origin_shells
            )

    def _assign_sites(self):
        structure = self.structure
        self.primitive_cell = get_primitive_cell(structure, self.symprec)
        prim_lattice, prim_scaled, prim_numbers = self.primitive_cell
        prim_scaled = _wrap(prim_scaled)
        self.primitive_dataset = get_symmetry_dataset(
            self.primitive_cell, symprec=self.symprec
        )
        site_tol = max(10 * self.symprec, 1e-4)

        # Basis sites are assigned by minimum-image distance, not spglib's
        # mapping_to_primitive, which refers to the idealized cell.
        numbers = structure.get_atomic_numbers()
        scaled = _wrap(structure.get_positions() @ np.linalg.inv(prim_lattice))
        delta = scaled[:, None] - prim_scaled[None]
        delta -= np.round(delta)
        distances = np.linalg.norm(delta @ prim_lattice, axis=-1)
        distances[numbers[:, None] != prim_numbers[None]] = np.inf
        prim_index = np.argmin(distances, axis=1)
        if distances[np.arange(len(scaled)), prim_index].max() > site_tol:
            raise RuntimeError(
                "An atom matches no primitive basis atom; check the "
                "structure or symprec."
            )

        dense = {}
        for p in prim_index.tolist():
            dense.setdefault(p, len(dense))
        basis_site_of_atom = np.array([dense[p] for p in prim_index])
        wyckoff_dense = {}
        wyckoff_of_basis = []
        for p in sorted(dense, key=dense.get):
            raw = int(self.primitive_dataset.equivalent_atoms[p])
            wyckoff_of_basis.append(
                wyckoff_dense.setdefault(raw, len(wyckoff_dense))
            )
        wyckoff_of_basis = np.array(wyckoff_of_basis)

        basis_sites = {}
        for p, basis_site in dense.items():
            basis_sites[basis_site] = {
                "Wyckoff_site": int(wyckoff_of_basis[basis_site]),
                "representative_atom": int(np.flatnonzero(prim_index == p)[0]),
                "primitive_scaled": prim_scaled[p].copy(),
            }
        representative = {}
        for basis_site in sorted(basis_sites):
            representative.setdefault(
                basis_sites[basis_site]["Wyckoff_site"], basis_site
            )
        for basis_site, data in basis_sites.items():
            rep = representative[data["Wyckoff_site"]]
            data["coset_rotation"] = (
                np.eye(3)
                if basis_site == rep
                else self._coset_rotation(
                    basis_sites[rep]["primitive_scaled"],
                    data["primitive_scaled"],
                    site_tol,
                )
            )

        symbols = structure.get_chemical_symbols()
        atom_to_wyckoff = wyckoff_of_basis[basis_site_of_atom]
        wyckoff_sites = {}
        for w in range(len(wyckoff_dense)):
            members = sorted(
                bs for bs, d in basis_sites.items() if d["Wyckoff_site"] == w
            )
            atom = basis_sites[members[0]]["representative_atom"]
            n_here = int(np.sum(atom_to_wyckoff == w))
            wyckoff_sites[w] = {
                "species": symbols[atom],
                "atomic_number": int(numbers[atom]),
                "mass": float(structure.get_masses()[atom]),
                "n_atoms": n_here,
                "atomic_fraction": n_here / len(structure),
                "basis_sites": members,
            }
        by_element = {}
        for w, d in wyckoff_sites.items():
            by_element.setdefault(d["species"], []).append(w)
        for element, members in by_element.items():
            if len(members) > 1:
                for i, w in enumerate(sorted(members), start=1):
                    wyckoff_sites[w]["species"] = f"{element}{i}"

        self.analysis.update(
            {
                "n_atoms": len(structure),
                "species": sorted(set(symbols)),
                "atom_to_basis_site_map": basis_site_of_atom,
                "atom_to_Wyckoff_site_map": atom_to_wyckoff,
                "primitive_lattice": prim_lattice,
                "Wyckoff_sites": wyckoff_sites,
                "basis_sites": basis_sites,
            }
        )

    def _operations_mapping(self, tau_from, tau_to, tol):
        """Mask of primitive space-group operations taking fractional position
        tau_from onto tau_to."""
        dataset = self.primitive_dataset
        lattice = self.primitive_cell[0]
        images = _wrap(dataset.rotations @ tau_from + dataset.translations)
        diff = _wrap(images - tau_to)
        diff = np.minimum(diff, 1.0 - diff)
        return np.linalg.norm(diff @ lattice, axis=-1) < tol

    def _coset_rotation(self, tau_from, tau_to, tol):
        mask = self._operations_mapping(tau_from, tau_to, tol)
        if not mask.any():
            raise RuntimeError(
                f"No space-group operation maps {tau_from} onto {tau_to}."
            )
        return _cartesian_operations(
            self.primitive_cell[0],
            self.primitive_dataset.rotations[np.argmax(mask)],
        )

    def _site_rotations(self, atom):
        lattice = self.primitive_cell[0]
        tau = _wrap(
            self.structure.get_positions()[atom] @ np.linalg.inv(lattice)
        )
        mask = self._operations_mapping(tau, tau, max(10 * self.symprec, 1e-4))
        return _cartesian_operations(
            lattice, self.primitive_dataset.rotations[mask]
        )

    def _bond_lengths(self, cutoff):
        i, d = neighbour_list("id", self.structure, cutoff)
        per_site = [
            np.unique(
                np.round(d[i == data["representative_atom"]], self.decimals)
            )
            for data in self.analysis["basis_sites"].values()
        ]
        return np.unique(np.concatenate(per_site)), per_site

    def _cutoff_for(self, n_shells):
        """Smallest cutoff giving every basis site at least n_shells distinct
        bond lengths."""
        trial = 2.0 * float(min(self.structure.cell.lengths()))
        for _ in range(20):
            _, per_site = self._bond_lengths(trial)
            if min(len(p) for p in per_site) > n_shells:
                return float(
                    max(p[n_shells - 1] for p in per_site)
                    + 5 * 10 ** (-self.decimals)
                )
            trial *= 1.2
        raise RuntimeError(
            f"Could not find a cutoff holding {n_shells} shells."
        )

    def _shell_orbits(self, atom, rotations):
        """Bonds of one atom split by (distance, target basis site) and then
        into site-symmetry orbits."""
        if self._neighbors is None:
            self._neighbors = neighbour_list(
                "ijdD", self.structure, self.cutoff_radius
            )
        i, j, d, D = self._neighbors
        mask = i == atom
        neighbors = j[mask]
        vectors = D[mask]
        lengths = np.round(d[mask], self.decimals)
        targets = self.analysis["atom_to_basis_site_map"][neighbors]
        orbits = []
        for length, target in sorted(
            set(zip(lengths.tolist(), targets.tolist()))
        ):
            select = (lengths == length) & (targets == target)
            order = np.lexsort(vectors[select].T[::-1])
            bonds = vectors[select][order]
            ids = neighbors[select][order]
            images = np.einsum("kij,nj->kni", rotations, bonds)
            remaining = set(range(len(bonds)))
            split = []
            while remaining:
                seed = min(remaining)
                hit = (
                    np.linalg.norm(
                        images[:, seed, None] - bonds[None], axis=-1
                    )
                    < self.tol
                )
                members = sorted(set(np.flatnonzero(hit.any(axis=0)).tolist()))
                split.append((target, length, bonds[members], ids[members]))
                remaining -= set(members)
            # deterministic order, so sibling basis sites number their shells
            # alike
            orbits += sorted(split, key=lambda o: (-len(o[2]), o[2].tolist()))
        return orbits

    def _resolve_orbit(self, rotations, target, length, bonds, ids):
        ref = int(np.lexsort(bonds.T[::-1])[-1])
        diffs = np.linalg.norm(
            (rotations @ bonds[ref])[:, None] - bonds[None], axis=-1
        )
        rotation_ids = [None] * len(bonds)
        for k, b in zip(
            *np.unravel_index(np.argsort(diffs, axis=None), diffs.shape)
        ):
            if diffs[k, b] >= self.tol:
                break
            if rotation_ids[b] is None:
                rotation_ids[b] = int(k)
        if None in rotation_ids:
            raise RuntimeError(
                "A bond of a shell orbit has no site-symmetry operation; "
                "loosen tol."
            )
        target_wyckoff = self.analysis["basis_sites"][target]["Wyckoff_site"]
        return {
            "reference_bond_vector": bonds[ref],
            "reference_bond_length": length,
            "reference_neighbor_id": int(ids[ref]),
            "target_basis_site": int(target),
            "target_species": self.analysis["Wyckoff_sites"][target_wyckoff][
                "species"
            ],
            "n_bonds": len(bonds),
            "neighbor_ids": ids,
            "rotation_ids": rotation_ids,
            "bond_vectors": bonds,
        }

    def analyze(self):
        """
        Returns
        -------
        analysis : dict
            Top level: n_atoms, species, atom_to_basis_site_map,
            atom_to_Wyckoff_site_map, primitive_lattice, Wyckoff_sites,
            basis_sites.
            basis_sites[bs]: Wyckoff_site, representative_atom,
            primitive_scaled, coset_rotation, rotations (site-symmetry,
            Cartesian), shells and origin_shells.
            shells[s]: reference_bond_vector, reference_bond_length,
            target_basis_site, target_species, n_bonds, neighbor_ids,
            rotation_ids, bond_vectors.
        """
        epsilon = 5 * 10 ** (-self.decimals)
        for data in self.analysis["basis_sites"].values():
            rotations = self._site_rotations(data["representative_atom"])
            orbits = self._shell_orbits(data["representative_atom"], rotations)
            shells = {
                s: self._resolve_orbit(rotations, *o)
                for s, o in enumerate(orbits, start=1)
            }
            data["rotations"] = rotations
            data["shells"] = {
                s: v
                for s, v in shells.items()
                if v["reference_bond_length"]
                <= self.bonded_cutoff_radius + epsilon
            }
            data["origin_shells"] = {
                s: v for s, v in shells.items() if s not in data["shells"]
            }
        self.paired_shells = pair_shells_by_reverse_bond(
            self.analysis, self.tol
        )
        return self.analysis

    def assign_bond_types(self):
        """
        One bond type per reciprocal pair of (Wyckoff site, shell).

        Returns
        -------
        bond_type_map : dict
            {(basis_site, shell): bond_type}
        bond_type_shells : dict
            {bond_type: ((Wyckoff_site_a, shell_a), (Wyckoff_site_b, shell_b))}
        """
        wyckoff_of = {
            bs: d["Wyckoff_site"]
            for bs, d in self.analysis["basis_sites"].items()
        }

        def lift(key):
            return wyckoff_of[key[0]], key[1]

        pairs = sorted(
            {
                tuple(sorted((lift(k), lift(v))))
                for k, v in self.paired_shells.items()
            }
        )
        bond_type_shells = dict(enumerate(pairs, start=1))
        type_of = {
            shell: t for t, pair in bond_type_shells.items() for shell in pair
        }
        bond_type_map = {key: type_of[lift(key)] for key in self.paired_shells}
        return bond_type_map, bond_type_shells

    def classify_directed_bonds(self):
        """(origin_atom, target_atom, basis_site, shell) for both directions of
        every bond."""
        i, j, d, D = neighbour_list(
            "ijdD",
            self.structure,
            self.bonded_cutoff_radius + 5 * 10 ** (-self.decimals),
        )
        basis_of = self.analysis["atom_to_basis_site_map"]
        directed = []
        for basis_site in self.analysis["basis_sites"]:
            shells = sorted(self.analysis["basis_sites"][basis_site]["shells"])
            directions = np.concatenate(
                [
                    generate_bond_vectors(self.analysis, basis_site, s)
                    for s in shells
                ]
            )
            labels = np.concatenate(
                [
                    [s]
                    * self.analysis["basis_sites"][basis_site]["shells"][s][
                        "n_bonds"
                    ]
                    for s in shells
                ]
            )
            mask = basis_of[i] == basis_site
            diffs = np.linalg.norm(
                D[mask][:, None] - directions[None], axis=-1
            )
            best = diffs.argmin(axis=1)
            if diffs[np.arange(len(best)), best].max() > self.tol:
                raise ValueError(
                    "A bond matches no shell; the cutoff includes part of a "
                    "shell."
                )
            directed += zip(
                i[mask].tolist(),
                j[mask].tolist(),
                [basis_site] * int(mask.sum()),
                labels[best].tolist(),
            )
        return directed

    def get_lammps_bonds(self):
        """[(atom_i, atom_j, bond_type)] with atom_i < atom_j, and the
        bond_type_map."""
        if self.lammps_bonds is None:
            self.bond_type_map, _ = self.assign_bond_types()
            bonds = {}
            for (
                origin,
                target,
                basis_site,
                shell,
            ) in self.classify_directed_bonds():
                if origin < target:
                    if (origin, target) in bonds:
                        raise ValueError(
                            "An atom pair appears twice; use a larger "
                            "supercell."
                        )
                    bonds[(origin, target)] = self.bond_type_map[
                        (basis_site, shell)
                    ]
            self.lammps_bonds = [(a, b, t) for (a, b), t in bonds.items()]
        return self.lammps_bonds, self.bond_type_map

    def get_structure_with_bonds(self, structure=None):
        """Copy of the structure with a "bonds" array, for ase.io.write(...,
        format="lammps-data", atom_style="full")."""
        structure = (
            structure if structure is not None else self.structure
        ).copy()
        per_atom = [[] for _ in range(len(structure))]
        for a, b, t in self.get_lammps_bonds()[0]:
            per_atom[a].append(f"{b}({t})")
        structure.new_array(
            "bonds",
            np.array(
                [",".join(e) if e else "_" for e in per_atom], dtype=object
            ),
            dtype=object,
        )
        return structure

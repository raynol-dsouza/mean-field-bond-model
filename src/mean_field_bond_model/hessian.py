"""
Displaced supercells for finite-displacement Hessians (phonopy), and
isotropically strained copies of a structure for multi-volume Hessians. Forces
are computed externally and passed back to compute_hessian.
"""

import numpy as np
from ase import Atoms
from phonopy import Phonopy
from phonopy.structure.atoms import PhonopyAtoms

__all__ = [
    "compute_hessian",
    "compute_hessians_for_structures",
    "generate_displacements_for_structures",
    "generate_strained_structures",
    "generate_supercells_with_displacements",
]


def _to_phonopy(atoms):
    return PhonopyAtoms(
        symbols=atoms.get_chemical_symbols(),
        cell=atoms.get_cell().array,
        scaled_positions=atoms.get_scaled_positions(),
        masses=atoms.get_masses(),
    )


def _to_ase(atoms):
    return Atoms(
        symbols=atoms.symbols,
        cell=atoms.cell,
        scaled_positions=atoms.scaled_positions,
        masses=atoms.masses,
        pbc=True,
    )


def _phonopy(structure, supercell_matrix, symprec):
    return Phonopy(
        _to_phonopy(structure),
        supercell_matrix=supercell_matrix,
        primitive_matrix="P",
        symprec=symprec,
    )


def generate_supercells_with_displacements(
    structure,
    supercell=1,
    distance=0.01,
    symprec=1e-5,
    is_plusminus="auto",
    return_phonopy=False,
):
    """
    Returns
    -------
    displaced : list of ase.Atoms
        Supercells to compute forces on, in this order.
    displacement_dict : dict
        Everything compute_hessian needs. Its "supercell_structure" is the
        structure the Hessian is indexed against, and the one to analyze.
    phonon : phonopy.Phonopy
        Only with return_phonopy; can be passed to compute_hessian and reused
        for phonon properties.
    """
    supercell = np.asarray(supercell)
    matrix = (
        np.diag(np.broadcast_to(supercell, 3))
        if supercell.ndim < 2
        else supercell
    )
    phonon = _phonopy(structure, matrix, symprec)
    phonon.generate_displacements(distance=distance, is_plusminus=is_plusminus)
    displacement_dict = {
        "input_structure": structure,
        "supercell_matrix": matrix,
        "symprec": symprec,
        "supercell_structure": _to_ase(phonon.supercell),
        "displacement_dataset": phonon.dataset,
    }
    displaced = [_to_ase(s) for s in phonon.supercells_with_displacements]
    if return_phonopy:
        return displaced, displacement_dict, phonon
    return displaced, displacement_dict


def compute_hessian(forces, displacement_dict, phonopy=None):
    """
    Mass-unweighted Hessian, shape (N, N, 3, 3), of the supercell from the
    forces on the displaced supercells. A Phonopy object from
    generate_supercells_with_displacements is reused if given.
    """
    d = displacement_dict
    phonon = (
        phonopy
        if phonopy is not None
        else _phonopy(
            d["input_structure"], d["supercell_matrix"], d["symprec"]
        )
    )
    phonon.dataset = d["displacement_dataset"]
    phonon.forces = np.asarray(forces)
    phonon.produce_force_constants()
    return np.asarray(phonon.force_constants)


def generate_strained_structures(structure, strain, n_strains=None):
    """
    Strained copies of structure, cell @ F.T with fractional coordinates kept.

    strain is one of: a scalar with n_strains (isotropic strains evenly spaced
    in [-strain, strain]); [minus, plus] with n_strains (evenly spaced between
    them); a 1-D array of isotropic strains; or an (n, 3, 3) array of
    deformation gradients F.
    """
    strain = np.asarray(strain, float)
    if strain.ndim == 3:
        gradients = strain
    else:
        if n_strains is not None:
            if strain.ndim == 0:
                lo = -strain
                hi = strain
            else:
                lo = strain[0]
                hi = strain[1]
            strain = np.linspace(lo, hi, n_strains)
        gradients = [(1.0 + e) * np.eye(3) for e in np.atleast_1d(strain)]
    out = []
    for F in gradients:
        strained = structure.copy()
        strained.set_cell(structure.get_cell().array @ F.T, scale_atoms=True)
        out.append(strained)
    return out


def generate_displacements_for_structures(structures, **kwargs):
    """generate_supercells_with_displacements(structure, **kwargs) for every
    structure."""
    return [
        generate_supercells_with_displacements(structure, **kwargs)
        for structure in structures
    ]


def compute_hessians_for_structures(forces, displacement_dicts, phonopys=None):
    """compute_hessian for every (forces, displacement_dict[, phonopy])
    triple."""
    phonopys = phonopys if phonopys is not None else [None] * len(forces)
    return [
        compute_hessian(f, d, p)
        for f, d, p in zip(forces, displacement_dicts, phonopys)
    ]

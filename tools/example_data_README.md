# Example data for the mean-field bond model

Data for the example notebooks of the `mean-field-bond-model` code
(https://github.com/raynol-dsouza/mean-field-bond-model), for six crystals:
`Al_fcc`, `Fe_bcc`, `Mg_hcp`, `AlFe_B2`, `AlFe3_D0_3` and `Al2FeNi_L2_1`.

This dataset accompanies

R. Dsouza, T. D. Swinburne, and J. Neugebauer, "Thermodynamics of complex
crystals from a generalized mean-field bond model", arXiv:<arXiv ID> (2026).

The code used to generate and analyze it is release <version> of
mean-field-bond-model, DOI: <code DOI>.

Each archive holds folders of the repository's `examples/` directory and
unpacks into it, e.g. with

```python
import mean_field_bond_model as mfbm

mfbm.example_data.fetch_example_data(
    ["inputs", "morse_hessians", "md_data"], "examples"
)
```

which downloads the archives from this record, checks their SHA-256 and
unpacks them.

## Archives

### `inputs.zip`: inputs of both notebooks

- `structures/<system>/<system>.data`: relaxed structures (LAMMPS data
  files).
- `potentials/`: the reference interatomic potentials the Morse parameters
  were fitted to, redistributed from their original sources:
  - `Al/Al1.eam.fs` (EAM, Al): M. I. Mendelev, M. J. Kramer, C. A. Becker,
    and M. Asta, Philos. Mag. 88, 1723 (2008);
  - `AlFe/Al-Fe.eam.fs` (EAM, AlFe and AlFe3): M. I. Mendelev,
    D. J. Srolovitz, G. J. Ackland, and S. Han, J. Mater. Res. 20, 208
    (2005);
  - `AlFeNi/FeNiAl_Hayakawa_2024.eam.alloy` (EAM, Al2FeNi): S. Hayakawa and
    H. Xu, Comput. Mater. Sci. 232, 112614 (2024);
  - `Fe/Fe_nmFe.2021.mod.adp` (ADP, Fe): S. Starikov, D. Smirnova,
    T. Pradhan, Y. Lysogorskiy, H. Chapman, M. Mrovec, and R. Drautz,
    Phys. Rev. Materials 5, 063607 (2021);
  - `Mg/Mg.eam.fs` (EAM, Mg): D. Y. Sun, M. I. Mendelev, C. A. Becker,
    K. Kudin, T. Haxhimali, M. Asta, J. J. Hoyt, A. Karma, and
    D. J. Srolovitz, Phys. Rev. B 73, 024116 (2006).
- `morse_parameters/<system>.json`: the Morse parameters, per Wyckoff
  position and shell, fitted by `hessian_to_morse_parametrization.ipynb` to
  the force constants of the reference potential at several isotropic
  strains. The Hessians and MD data below were generated from them.

### `morse_hessians.zip`: inputs of the model notebook

`morse_hessians/<system>/<system>.npz`, key `hessian`: the mass-unweighted
Hessian (eV/A^2), shape (N, N, 3, 3), of the perfect supercell at the
relaxed volume, computed with the Morse bonding potentials of
`examples/morse_parameters/<system>.json`. The supercells are the same as for
the MD simulations:

| System | Neighbor shells | Simulation cell | Atoms |
| --- | --- | --- | --- |
| `Al_fcc` | 3 | 5x5x5 | 500 |
| `Fe_bcc` | 3 | 6x6x6 | 432 |
| `Mg_hcp` | 3 | 5x5x5 | 500 |
| `AlFe_B2` | 2 | 6x6x6 | 432 |
| `AlFe3_D0_3` | 2 | 4x4x4 | 1024 |
| `Al2FeNi_L2_1` | 2 | 4x4x4 | 1024 |

Neighbor shells are counted as site-symmetry orbits. For Mg, the c/a ratio is
fixed at its zero-temperature value.

### `md_data.zip`: MD reference data

- `md_data/<system>/<system>_1000K_bond.npz`: bond densities from NVT MD at
  1000 K, per (Wyckoff site, shell), on a cube of 51x51x51 displacements
  about the reference bond; read with
  `mean_field_bond_model.md.load_md_bond_data`.
- `md_data/<system>/<system>_thermo_nvt.csv` and `..._thermo_npt.csv`: NVT and
  NPT temperature sweeps, one row per temperature (K), with the anharmonic
  internal and free energies `U_ah`, `F_ah` (meV/atom), the pressure `P`
  (bar), the lattice scaling `eps` and their standard errors (`*_sem`).

MD settings. All simulations were run with LAMMPS (A. P. Thompson et al.,
Comput. Phys. Commun. 271, 108171 (2022)), version 22 Jul 2025 - Update 2, on
the supercells of the table above. They used the same Morse bonding
potentials as the model, not the reference potentials: each bond carries the
potential of its shell through a fixed bond topology, so the MD samples
exactly the model's Hamiltonian. Ten temperatures, evenly spaced from 100 K to
1000 K, were each sampled with five independent runs of 10 000 equilibration
and 25 000 production steps at a 1 fs time step. NVT runs used a Langevin
thermostat at the volume of the relaxed crystal. NPT runs used an additional
isotropic Nose-Hoover barostat at zero pressure.

Analysis. The anharmonic internal energy per atom is
U_ah = [U_MD - (3N - 3) k_B T_kin / 2] / N, with T_kin the instantaneous
kinetic temperature and -3 the center-of-mass correction. In NPT, the static
energy of the lattice at the sampled strain is subtracted as well. Means and
standard errors come from block averages over each run, combined over the
five runs. The anharmonic free energy is obtained from a weighted
least-squares fit U_ah(T) = sum_{n=2}^{4} a_n T^n, with no constant or
linear term. This integrates analytically to
F_ah(T) = -sum_{n=2}^{4} a_n T^n / (n - 1), with standard errors propagated
from the covariance of the fitted coefficients.

### `mean_field_bond_model_results.zip`: results of the model notebook

- `results/<system>/<system>_1000K_bond.npz`: mean-field bond densities at
  1000 K, on the same grids as the MD ones.
- `results/<system>/<system>_thermo_nvt.csv` and `..._thermo_npt.csv`:
  mean-field NVT and NPT sweeps at the same ten temperatures: `U_ah`, `F_ah`
  (meV/atom), `P` (bar), `eps` and the virial temperature `T_vir` (K). `F_ah`
  is obtained by trapezoidal integration over temperature, extrapolated
  linearly from the two lowest temperatures to T = 0.
- `plots/<system>/`: the figures of the notebook.

These are regenerated by running `mean_field_bond_model.ipynb`, and are
archived as the results reported in the paper.

## License

CC BY 4.0, except for the files in `potentials/`, which are redistributed
under the terms of their original sources, cited above.

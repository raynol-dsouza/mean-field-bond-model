# Code pertaining to the mean-field bond model 

Site-symmetry-aware nearest-neighbor bond shell analysis and the mean-field
bond model for complex atomic crystals.

The package takes a relaxed crystal structure and:

1. finds its neighbor-bond shells per Wyckoff site and the site-symmetry
   operations that generate every bond of a shell from one reference bond
   (`symmetry`);
2. calibrates a Morse bonding potential per shell from finite-displacement
   Hessians at one or several volumes (`hessian`, `morse`);
3. computes normalized bond-bond covariances from a real-space Hessian by a
   Bloch sum over commensurate q-points (`covariances`);
4. solves the self-consistent mean-field bond model in the NVT and NPT
   ensembles, giving bond densities, the virial temperature, pressure,
   anharmonic internal and free energies, and thermal expansion
   (`mean_field`);
5. builds the matching bond densities from MD trajectories for comparison
   (`md`), and plots both (`plot`, `plot_data`).

The model works for crystals with any number of Wyckoff sites and species.

## Installation

```bash
git clone https://github.com/raynol-dsouza/mean-field-bond-model.git
cd mean-field-bond-model
pip install -e ".[examples]"
```

Python 3.10 or newer is required. The core dependencies are `ase`,
`matscipy`, `numpy`, `phonopy`, `scipy` and `spglib`; the `examples` extra
adds `matplotlib`, `pandas` and `jupyter`.

## Examples

Two notebooks under `examples/` run the full pipeline. They ship without
outputs; each runs in about a minute on a laptop, and the paper's results and
figures are archived on Zenodo (`mean_field_bond_model_results`, below).

| Notebook | What it does |
| --- | --- |
| `hessian_to_morse_parametrization.ipynb` | Hessians of an EAM potential at seven volumes, then a joint Morse fit per shell. |
| `mean_field_bond_model.ipynb` | Covariances, the mean-field bond model at 1000 K compared with MD bond densities, and NVT/NPT temperature sweeps compared with MD thermodynamics. |

Both default to fcc Al. Change `prefix` in the model notebook to run any of
the six bundled systems: `Al_fcc`, `Fe_bcc`, `Mg_hcp`, `AlFe_B2`,
`AlFe3_D0_3` and `Al2FeNi_L2_1`.

## Example data (Zenodo)

All example data, inputs included, is archived on Zenodo (https://doi.org/10.5281/zenodo.24466123) rather
than in the repository, and unpacks into `examples/`:

| Archive | Folders | Contents |
| --- | --- | --- |
| `inputs` | `structures/`, `potentials/`, `morse_parameters/` | Relaxed structures, the EAM/ADP potentials and the fitted Morse parameters. |
| `morse_hessians` | `morse_hessians/` | Hessians computed with the bundled Morse parameters. |
| `md_data` | `md_data/` | MD reference bond densities at 1000 K and NVT/NPT thermodynamics. |
| `mean_field_bond_model_results` | `results/`, `plots/` | Model results and figures, as reported in the paper. |

The notebooks download what they need on their first run: the parametrization
notebook `inputs`, the model notebook `inputs`, `morse_hessians` and
`md_data`. To fetch any of them yourself:

```python
import mean_field_bond_model as mfbm

mfbm.example_data.fetch_example_data(
    ["inputs", "morse_hessians", "md_data", "mean_field_bond_model_results"],
    "examples",
)
```

Each archive is checked against its SHA-256 (`example_data.json`) before it is
unpacked. The Hessians and MD data were generated from the archived Morse
parameter files, which the parametrization notebook doesn't overwrite unless
you set `overwrite = True`.

## Minimal usage

```python
import mean_field_bond_model as mfbm
import numpy as np

from ase.io import read

mfbm.example_data.fetch_example_data(["inputs", "morse_hessians"], "examples")
structure = read(
    "examples/structures/Al_fcc/Al_fcc.data", format="lammps-data"
)
_, displacement_dict = mfbm.hessian.generate_supercells_with_displacements(
    structure, supercell=5
)
supercell = displacement_dict["supercell_structure"]

morse_params = mfbm.load_morse_params("examples/morse_parameters/Al_fcc.json")
hessian = np.load("examples/morse_hessians/Al_fcc/Al_fcc.npz")["hessian"]
analysis = mfbm.NeighborBondSymmetryAnalysis(supercell, n_shells=3).analyze()
alphas = mfbm.BondPairCovarianceCalculator(
    supercell, analysis, hessian
).compute_alphas()

model = mfbm.MeanFieldBondModel(
    supercell, mfbm.build_bonding_potentials(morse_params), analysis, alphas
)

result = model.run_npt([300.0, 600.0, 900.0], pressure=0.0)
print([r["eps"] for r in result])
```

## License

BSD 3-Clause; see `LICENSE`.

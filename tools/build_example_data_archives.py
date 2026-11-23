"""
Builds the example-data archives for Zenodo from the folders under examples/,
and writes their checksums to src/mean_field_bond_model/example_data.json.

The archives are reproducible: files in sorted order, fixed timestamps and
permissions, no compression (the .npz and .png files are compressed already),
so the same files always give the same SHA-256.

    python tools/build_example_data_archives.py OUTPUT_DIR

Upload every .zip in OUTPUT_DIR and its README.md to the Zenodo record, then
set ZENODO_RECORD in src/mean_field_bond_model/example_data.py.
"""

import hashlib
import json
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
REGISTRY = ROOT / "src" / "mean_field_bond_model" / "example_data.json"

# archive name: folders under examples/ it holds
ARCHIVES = {
    "inputs": [
        "structures",
        "potentials",
        "morse_parameters",
    ],
    "morse_hessians": ["morse_hessians"],
    "md_data": ["md_data"],
    "mean_field_bond_model_results": ["results", "plots"],
}


def build(name, folders, output_dir):
    path = output_dir / f"{name}.zip"
    files = sorted(
        f
        for folder in folders
        for f in (EXAMPLES / folder).rglob("*")
        if f.is_file()
    )
    if not files:
        raise FileNotFoundError(f"No files under {folders} in {EXAMPLES}.")
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as archive:
        for f in files:
            info = zipfile.ZipInfo(
                f.relative_to(EXAMPLES).as_posix(), (1980, 1, 1, 0, 0, 0)
            )
            info.external_attr = 0o644 << 16
            archive.writestr(info, f.read_bytes())
    sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
    print(f"{path.name}: {len(files)} files, {path.stat().st_size} bytes")
    return {
        "file": path.name,
        "sha256": sha256,
        "size": path.stat().st_size,
        "folders": folders,
    }


def main(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    registry = {
        name: build(name, folders, output_dir)
        for name, folders in ARCHIVES.items()
    }
    REGISTRY.write_text(json.dumps(registry, indent=2) + "\n")
    shutil.copy(
        Path(__file__).with_name("example_data_README.md"),
        output_dir / "README.md",
    )
    print(f"Wrote {REGISTRY.relative_to(ROOT)} and {output_dir}/README.md")


if __name__ == "__main__":
    main(sys.argv[1])

"""
Example data of the notebooks under examples/ (structures, potentials, Morse
parameters, Hessians, MD reference data and model results), archived on Zenodo
and downloaded on demand, so that the repository holds only code.

Each archive is a zip of folders under examples/, checked against the SHA-256
in example_data.json and unpacked in place.
"""

import hashlib
import json
import tempfile
import urllib.request
import zipfile
from pathlib import Path

__all__ = ["ARCHIVES", "ZENODO_RECORD", "fetch_example_data"]

# Zenodo record id of the example data, set once the record is published
ZENODO_RECORD = 24466123

ARCHIVES = json.loads(
    Path(__file__).with_name("example_data.json").read_text()
)


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _unpack(path, examples_dir):
    with zipfile.ZipFile(path) as archive:
        for name in archive.namelist():
            target = (examples_dir / name).resolve()
            if examples_dir.resolve() not in target.parents:
                raise RuntimeError(f"Unsafe path {name!r} in {path.name}.")
        archive.extractall(examples_dir)


def fetch_example_data(names, examples_dir, url=None, force=False):
    """
    Download, check and unpack example-data archives into examples_dir,
    skipping those whose folders are already there.

    Parameters
    ----------
    names : list of str
        Archives, keys of ARCHIVES: "inputs", "morse_hessians", "md_data",
        "mean_field_bond_model_results".
    examples_dir : str or Path
        The repository's examples/ directory.
    url : str, optional
        Base URL holding the archives, e.g. "file:///path/to/archives".
        Defaults to the files of the Zenodo record ZENODO_RECORD.
    force : bool, optional
        Download even if the folders exist. Defaults to False.

    Returns
    -------
    folders : list of Path
        The folders of the requested archives.
    """
    examples_dir = Path(examples_dir)
    folders = []
    for name in names:
        if name not in ARCHIVES:
            raise ValueError(
                f"Unknown archive {name!r}; choose from {list(ARCHIVES)}."
            )
        entry = ARCHIVES[name]
        paths = [examples_dir / folder for folder in entry["folders"]]
        folders += paths
        if not force and all(p.is_dir() for p in paths):
            continue
        if url is not None:
            source = f"{url.rstrip('/')}/{entry['file']}"
        elif ZENODO_RECORD is not None:
            source = (
                f"https://zenodo.org/records/{ZENODO_RECORD}/files/"
                f"{entry['file']}?download=1"
            )
        else:
            raise RuntimeError(
                "The example data is not on Zenodo yet; pass url, the "
                "location of the archives."
            )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / entry["file"]
            size = entry["size"] / 1e6
            print(f"Downloading {entry['file']} ({size:.1f} MB)")
            urllib.request.urlretrieve(source, path)
            if _sha256(path) != entry["sha256"]:
                raise RuntimeError(
                    f"{entry['file']} does not match its SHA-256; the "
                    "download is corrupt or the archive has changed."
                )
            _unpack(path, examples_dir)
    return folders

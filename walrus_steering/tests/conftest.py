import pathlib
import shutil
from typing import Dict, List

import pytest
from the_well.utils.dummy_data import write_dummy_data


@pytest.fixture()
def dummy_dataset(tmp_path):
    well_data_folder = tmp_path / "well_data"
    well_data_folder.mkdir()
    for split in ["train", "valid", "test"]:
        split_dir = well_data_folder / "dummy" / "data" / split
        split_dir.mkdir(parents=True)
        write_dummy_data(split_dir / "data.hdf5")
    return well_data_folder


@pytest.fixture()
def checkpoint_folder(tmp_path: pathlib.Path):
    """Create and clean a temporary folder for checkpoints"""
    yield tmp_path / "checkpoints"
    if tmp_path.exists():
        shutil.rmtree(tmp_path)


@pytest.fixture
def default_conf(dummy_dataset, checkpoint_folder) -> Dict[str, str | bool]:
    """Generate default training options as a dictionary."""
    return {
        "server": "local",
        "logger": "none",
        "trainer": "debug",
        "data": "test",
        "model": "debug",
        "data.well_base_path": dummy_dataset,
        "data_workers": "1",
        "name": "test",
        "checkpoint.save_dir": checkpoint_folder,
        "automatic_setup": False,
    }


def format_overrides(overrides: Dict[str, str]) -> List[str]:
    """Format training options from dictionary to list of overrides."""
    return [f"{key}={val}" for key, val in overrides.items()]


@pytest.fixture()
def conf(default_conf, request):
    """Generate overrides by combining default configuration and the various test options."""
    override_dict = default_conf
    if hasattr(request, "param"):
        override_dict.update(request.param)
    overrides = format_overrides(override_dict)
    return overrides

"""Test config.py"""

import os
from pathlib import Path

import pytest

from eeg_flow.config import load_triggers, create_config, load_config

fname_valid = Path(__file__).parent / "data" / "test_triggers.ini"
fname_invalid = Path(__file__).parent / "data" / "test_triggers_invalid.ini"


def test_load_triggers():
    """Test loading of triggers."""
    tdef = load_triggers(fname_valid)
    assert tdef["standard"] == 101
    assert tdef["target"] == 201
    assert tdef["novel"] == 301

    with pytest.raises(ValueError, match="Key 'novel' is missing"):
        load_triggers(fname_invalid)


def test_config(tmp_path):
    """Test creation and loading of a configuration file."""
    if (Path.home() / ".eeg_flow").exists():
        os.remove(Path.home() / ".eeg_flow")

    xdf_folder = tmp_path / "xdf"
    derivatives_folder = tmp_path / "derivatives"
    uname = "mscheltienne"
    os.makedirs(xdf_folder)
    os.makedirs(derivatives_folder)
    assert not (Path.home() / ".eeg_flow").exists()
    create_config(xdf_folder, derivatives_folder, uname)
    assert (Path.home() / ".eeg_flow").exists()
    xdf_folder_loaded, derivatives_folder_loaded, uname_loaded = load_config()
    assert xdf_folder_loaded == xdf_folder
    assert derivatives_folder_loaded == derivatives_folder
    assert uname_loaded == uname

    # overwrite configuration
    xdf_folder = tmp_path / "xdf2"
    os.makedirs(xdf_folder)
    assert (Path.home() / ".eeg_flow").exists()
    create_config(xdf_folder, derivatives_folder, uname)
    assert (Path.home() / ".eeg_flow").exists()
    xdf_folder_loaded, derivatives_folder_loaded, uname_loaded = load_config()
    assert xdf_folder_loaded == xdf_folder
    assert derivatives_folder_loaded == derivatives_folder
    assert uname_loaded == uname

    # remove configuration
    os.remove(Path.home() / ".eeg_flow")
    with pytest.raises(RuntimeError, match="folder does not exists. Call"):
        load_config()

    # try to create configuration with invalid paths
    with pytest.raises(FileNotFoundError, match="does not exist."):
        create_config(tmp_path / "101", derivatives_folder, uname)
    with pytest.raises(FileNotFoundError, match="does not exist."):
        create_config(xdf_folder, tmp_path / "101", uname)
    with pytest.raises(TypeError, match="must be an instance of"):
        create_config(xdf_folder, derivatives_folder, 101)

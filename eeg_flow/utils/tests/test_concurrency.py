import os

import pytest

from eeg_flow.utils.concurrency import lock_files


def test_concurrency(tmp_path):
    """Test lock/release of files for concurrent workflows."""
    files = (
        tmp_path / "test1.log",
        tmp_path / "test2.txt",
    )
    locks = lock_files(*files)
    assert all(file.with_suffix(file.suffix + ".lock").exists() for file in files)
    del locks
    assert all(not file.with_suffix(file.suffix + ".lock").exists() for file in files)
    locks = lock_files(*files)
    assert all(file.with_suffix(file.suffix + ".lock").exists() for file in files)
    with pytest.raises(RuntimeError, match="Could not lock all files."):
        lock_files(*files, timeout=0.2)
    os.remove(files[0].with_suffix(files[0].suffix + ".lock"))
    with pytest.raises(RuntimeError, match="Could not lock all files."):
        lock_files(*files, timeout=0.2)
    assert not files[0].with_suffix(files[0].suffix + ".lock").exists()
    os.remove(files[1].with_suffix(files[1].suffix + ".lock"))
    del locks

    locks = lock_files(*files)
    assert all(file.with_suffix(file.suffix + ".lock").exists() for file in files)
    for file in files:
        with open(file, "w+") as f:
            f.write("test")
    del locks
    assert all(not file.with_suffix(file.suffix + ".lock").exists() for file in files)

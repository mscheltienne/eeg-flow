"""Test config.py"""

from pathlib import Path

import pytest

from eeg_flow.config import load_triggers

fname_valid = Path(__file__).parent / "data" / "test_triggers.ini"
fname_invalid = Path(__file__).parent / "data" / "test_triggers_invalid.ini"


def test_load_triggers():
    """Test loading of triggers."""
    tdef = load_triggers(fname_valid)
    assert tdef.standard == 101
    assert tdef.target == 201
    assert tdef.novel == 301

    with pytest.raises(ValueError, match="Key 'novel' is missing"):
        load_triggers(fname_invalid)

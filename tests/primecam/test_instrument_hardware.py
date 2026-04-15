"""
Hardware tests for RFSOC instrument.

These tests require a live RFSOC board. Run with:
    pytest tests/primecam/test_instrument_hardware.py \
        --rfsoc_ip=192.168.3.40 \
        --rfsoc_out_dir=/path/to/output

All tests in this file are automatically skipped when --rfsoc_ip is not
provided on the command line.
"""

import pytest
from citkid.primecam.instrument import RFSOC


@pytest.fixture(scope='module', autouse=True)
def require_hardware(pytestconfig):
    ip = pytestconfig.getoption('--rfsoc_ip')
    if ip is None:
        pytest.skip('No RFSOC IP specified (--rfsoc_ip)')
    out_dir = pytestconfig.getoption('--rfsoc_out_dir')
    if out_dir is None:
        pytest.skip('No RFSOC output directory specified (--rfsoc_out_dir)')


def initialize_rfsoc(pytestconfig):
    ip = pytestconfig.getoption('--rfsoc_ip')
    out_dir = pytestconfig.getoption('--rfsoc_out_dir')
    return RFSOC(out_directory=out_dir, udp_ip=ip)


# ---------------------------------------------------------------------------
# Hardware tests — to be expanded when hardware is available
# ---------------------------------------------------------------------------

def test_rfsoc_hardware_init(pytestconfig):
    """RFSOC initializes against live hardware without error."""
    rfsoc = initialize_rfsoc(pytestconfig)
    assert rfsoc.bid is not None
    assert rfsoc.drid is not None


def test_set_nclo(pytestconfig):
    """set_nclo completes without error."""
    rfsoc = initialize_rfsoc(pytestconfig)
    rfsoc.set_nclo(500)  # 500 MHz


def test_write_vna_comb(pytestconfig):
    """write_vna_comb completes without error."""
    rfsoc = initialize_rfsoc(pytestconfig)
    rfsoc.set_nclo(500)
    rfsoc.write_vna_comb()


def test_vna_sweep(pytestconfig, tmp_path):
    """vna_sweep runs and produces an output file."""
    import os
    rfsoc = initialize_rfsoc(pytestconfig)
    rfsoc.set_nclo(500)
    rfsoc.write_vna_comb()
    filename = 's21_vna.npy'
    rfsoc.vna_sweep(filename, npoints=10, N_accums=1)
    assert os.path.isfile(os.path.join(rfsoc.out_directory, filename))

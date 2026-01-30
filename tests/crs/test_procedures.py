"""Tests for CRS procedures helpers."""

import numpy as np
import pytest
import warnings
import zarr

from citkid.crs import procedures


class DummyFirmwareRelease:
    def __init__(self, version="1.2.3"):
        self.version = version


class DummyCRS:
    pass


def make_dummy_crs(**overrides):
    crs = DummyCRS()
    crs.CRS = DummyCRS
    crs.ch_map = {1: [0, 1]}
    crs.nco_freqs = {1: 4.0e9}
    crs.firmware_release = DummyFirmwareRelease()
    crs.analog_bank_high = False
    crs.bw = 500e6
    crs.clock_source = "VCXO"
    crs.dec_module_idxs = np.array([1], dtype=np.uint8)
    crs.dec_short = False
    crs.dec_stage = 6
    crs.extended_bw = False
    crs.sample_freq = 976.0
    crs.serial_number = 27
    crs.rfmux_version = "1.3.2"
    crs.citkid_version = "1.0.0.dev0"

    for key, value in overrides.items():
        setattr(crs, key, value)
    return crs


def test_write_crs_config_to_zarr_writes_expected_arrays(tmp_path):
    grp = zarr.open_group(tmp_path / "config.zarr", mode="w")
    crs = make_dummy_crs()

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=Warning, message=".*Zarr V3 specification.*")
        procedures.write_crs_config_to_zarr(crs, grp)

    expected_names = {
        "chs_module1",
        "nco_module1",
        "firmware_version",
        "analog_bank_high",
        "bw",
        "clock_source",
        "dec_module_idxs",
        "dec_short",
        "dec_stage",
        "extended_bw",
        "sample_freq",
        "serial_number",
        "rfmux_version",
        "citkid_version",
    }
    assert expected_names.issubset(set(grp.keys()))

    np.testing.assert_array_equal(grp["chs_module1"][:], np.array([0, 1], dtype=np.int32))
    assert np.isclose(grp["nco_module1"][()], 4.0e9)
    assert str(grp["firmware_version"][()]) == "1.2.3"
    assert bool(grp["analog_bank_high"][()]) is False
    assert np.isclose(grp["bw"][()], 500e6)
    assert str(grp["clock_source"][()]) == "VCXO"
    np.testing.assert_array_equal(grp["dec_module_idxs"][:], np.array([1], dtype=np.uint8))
    assert bool(grp["dec_short"][()]) is False
    assert int(grp["dec_stage"][()]) == 6
    assert bool(grp["extended_bw"][()]) is False
    assert np.isclose(grp["sample_freq"][()], 976.0)
    assert int(grp["serial_number"][()]) == 27
    assert str(grp["rfmux_version"][()]) == "1.3.2"
    assert str(grp["citkid_version"][()]) == "1.0.0.dev0"


def test_write_crs_config_to_zarr_invalid_crs_type(tmp_path):
    grp = zarr.open_group(tmp_path / "config.zarr", mode="w")

    class NotCRS:
        pass

    not_crs = NotCRS()
    not_crs.CRS = DummyCRS

    with pytest.raises(TypeError, match="crs must be an instance of CRS class"):
        procedures.write_crs_config_to_zarr(not_crs, grp)


def test_write_crs_config_to_zarr_invalid_grp_type():
    crs = make_dummy_crs()
    with pytest.raises(TypeError, match="grp must be a zarr Group instance"):
        procedures.write_crs_config_to_zarr(crs, "not_a_group")


@pytest.mark.parametrize("missing_name", [
    "ch_map",
    "nco_freqs",
    "firmware_release",
    "analog_bank_high",
    "bw",
    "clock_source",
    "dec_module_idxs",
    "dec_short",
    "dec_stage",
    "extended_bw",
    "sample_freq",
    "serial_number",
    "rfmux_version",
    "citkid_version",
])
def test_write_crs_config_to_zarr_missing_crs_attribute(tmp_path, missing_name):
    grp = zarr.open_group(tmp_path / "config.zarr", mode="w")
    crs = make_dummy_crs()
    delattr(crs, missing_name)

    with pytest.raises(ValueError, match=f"crs is missing attribute '{missing_name}'"):
        procedures.write_crs_config_to_zarr(crs, grp)


@pytest.mark.parametrize("conflict_name", [
    "ch_map",
    "nco_freqs",
    "firmware_release",
    "analog_bank_high",
    "bw",
    "clock_source",
    "dec_module_idxs",
    "dec_short",
    "dec_stage",
    "extended_bw",
    "sample_freq",
    "serial_number",
    "rfmux_version",
    "citkid_version",
])
def test_write_crs_config_to_zarr_conflicting_dataset(tmp_path, conflict_name):
    grp = zarr.open_group(tmp_path / "config.zarr", mode="w")
    grp.create_array(conflict_name, data=np.array([1]))
    crs = make_dummy_crs()

    with pytest.raises(ValueError, match=f"Zarr group already contains dataset '{conflict_name}'"):
        procedures.write_crs_config_to_zarr(crs, grp)


def test_write_crs_config_to_zarr_invalid_firmware_version(tmp_path):
    grp = zarr.open_group(tmp_path / "config.zarr", mode="w")
    crs = make_dummy_crs(firmware_release=DummyFirmwareRelease(version=123))

    with pytest.raises(ValueError, match="crs.firmware_release.version must be a string"):
        procedures.write_crs_config_to_zarr(crs, grp)


def test_write_crs_config_to_zarr_invalid_ch_map(tmp_path):
    grp = zarr.open_group(tmp_path / "config.zarr", mode="w")
    crs = make_dummy_crs(ch_map={1: ["a"]})

    with pytest.raises((TypeError, ValueError)):
        procedures.write_crs_config_to_zarr(crs, grp)

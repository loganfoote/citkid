import pytest
import numpy as np
import warnings
import zarr
from citkid.crs import util 

################################################################################
################################# create_ch_map ################################
################################################################################
@pytest.mark.parametrize(
    "nco_freqs, freqs, bw, map_exp, missing_exp", 
    [
        (
            {0: 100e6, 1: 200e6},
            [100e6, 150e6, 200e6, 250e6],
            500e6,
            {0: [0, 1], 1: [2, 3]},
            []
        ),  # two modules, all tones assigned within broad bandwidth
        (
            {0: 50e6},
            [25e6, 75e6, 125e6],
            300e6,
            {0: [0, 1, 2]},
            []
        ),  # single module, all tones fall within bandwidth
        (
            {0: 100e6, 1: 150e6},
            [[95e6, 105e6], [145e6, 155e6], [120e6, 130e6]],
            100e6,
            {0: [0, 2], 1: [1]},
            []
        ),  # multi-tone channels; assign based on min/max within bw
        (
            {0: 100e6, 1: 300e6},
            [40e6, 100e6, 250e6, 360e6],
            100e6,
            {0: [1], 1: [2]},
            [0, 3]
        ),  # missing channels when outside all NCO bandwidths
        (
            {0: 200e6},
            [[150e6, 250e6], [149.9e6, 200e6]],
            100e6,
            {0: [0]},
            [1]
        ),  # boundary inclusion vs exclusion for multi-tone channels
        (
            {0: 100e6, 1: 200e6, 2: 300e6},
            [190e6, 260e6],
            300e6,
            {0: [], 1: [0], 2: [1]},
            []
        ),  # multiple candidates; choose closest to median frequency
    ],
)
def test_create_ch_map(nco_freqs, freqs, bw, map_exp, missing_exp):
    ch_map, missing_chs = util.create_ch_map(nco_freqs, freqs, bw)
    assert ch_map.keys() == map_exp.keys() 
    for k, v in ch_map.items():
        assert np.allclose(v, map_exp[k])
    assert np.allclose(missing_chs, missing_exp)


def test_create_ch_map_empty_freqs_returns_empty_arrays():
    ch_map, missing_chs = util.create_ch_map(
        {0: 100e6, 1: 200e6}, [], 200e6
    )
    assert set(ch_map.keys()) == {0, 1}
    assert ch_map[0].size == 0
    assert ch_map[1].size == 0
    assert missing_chs.size == 0


@pytest.mark.parametrize(
    "nco_freqs, freqs, bw",
    [
        ([(0, 100e6)], [100e6], 100e6),
        ({0: "100e6"}, [100e6], 100e6),
        ({"0": 100e6}, [100e6], 100e6),
        ({0: 100e6}, "100e6", 100e6),
        ({0: 100e6}, ["a"], 100e6),
        ({0: 100e6}, [["a"]], 100e6),
        ({0: 100e6}, [100e6], "a"), 
        ({0: 100e6}, [100e6], -100e6),
    ],
)
def test_create_ch_map_exceptions(nco_freqs, freqs, bw):
    with pytest.raises((TypeError, ValueError)):
        util.create_ch_map(nco_freqs, freqs, bw)


def test_create_ch_map_empty_nco_freqs_returns_missing_only():
    ch_map, missing_chs = util.create_ch_map({}, [100e6, 200e6], 100e6)
    assert ch_map == {}
    assert np.allclose(missing_chs, [0, 1])

################################################################################
################################## get_modules #################################
################################################################################
def test_get_modules_monkeypatched_happy_path(monkeypatch):
    class FakeModuleQuery:
        def __init__(self, indices):
            self.indices = indices

    class FakeReadoutModule:
        class module:
            @staticmethod
            def in_(module_indices):
                return FakeModuleQuery(module_indices)

    class FakeModules:
        def __init__(self):
            self.last_query = None
        def filter(self, query):
            self.last_query = query
            return f"filtered:{query.indices}"

    class FakeRfmux:
        ReadoutModule = FakeReadoutModule
        class core:
            class schema:
                class CRS:
                    pass

    class FakeCRS(FakeRfmux.core.schema.CRS):
        def __init__(self):
            self.modules = FakeModules()

    monkeypatch.setattr(util, "rfmux", FakeRfmux)

    d = FakeCRS()
    module_indices = [0, 2, 5]

    result = util.get_modules(d, module_indices)

    assert result == "filtered:[0, 2, 5]"
    assert d.modules.last_query is not None
    assert d.modules.last_query.indices == module_indices

def test_get_modules_passes_through_non_list_indices(monkeypatch):
    class FakeModuleQuery:
        def __init__(self, indices):
            self.indices = indices

    class FakeReadoutModule:
        class module:
            @staticmethod
            def in_(module_indices):
                return FakeModuleQuery(module_indices)

    class FakeModules:
        def __init__(self):
            self.last_query = None
        def filter(self, query):
            self.last_query = query
            return query.indices

    class FakeRfmux:
        ReadoutModule = FakeReadoutModule
        class core:
            class schema:
                class CRS:
                    pass

    class FakeCRS(FakeRfmux.core.schema.CRS):
        def __init__(self):
            self.modules = FakeModules()

    monkeypatch.setattr(util, "rfmux", FakeRfmux)

    d = FakeCRS()
    module_indices = (1, 3)

    result = util.get_modules(d, module_indices)

    assert result == (1, 3)
    assert d.modules.last_query.indices == (1, 3)

def test_get_modules_invalid_d_type_raises(monkeypatch):
    class FakeRfmux:
        class core:
            class schema:
                class CRS:
                    pass

    monkeypatch.setattr(util, "rfmux", FakeRfmux)

    with pytest.raises(TypeError):
        util.get_modules(d = "not_a_crs", module_indices = [1, 2])

def test_get_modules_invalid_module_indices_type_raises(monkeypatch):
    class FakeRfmux:
        class core:
            class schema:
                class CRS:
                    pass

    class FakeCRS(FakeRfmux.core.schema.CRS):
        def __init__(self):
            self.modules = object()

    monkeypatch.setattr(util, "rfmux", FakeRfmux)

    with pytest.raises(TypeError):
        util.get_modules(d = FakeCRS(), module_indices = 123)

def test_get_modules_invalid_module_indices_element_type_raises(monkeypatch):
    class FakeRfmux:
        class core:
            class schema:
                class CRS:
                    pass

    class FakeCRS(FakeRfmux.core.schema.CRS):
        def __init__(self):
            self.modules = object()

    monkeypatch.setattr(util, "rfmux", FakeRfmux)

    with pytest.raises((ValueError, TypeError)):
        util.get_modules(d = FakeCRS(), module_indices = [1, "2"])


def test_get_modules_empty_indices(monkeypatch):
    class FakeModuleQuery:
        def __init__(self, indices):
            self.indices = indices

    class FakeReadoutModule:
        class module:
            @staticmethod
            def in_(module_indices):
                return FakeModuleQuery(module_indices)

    class FakeModules:
        def __init__(self):
            self.last_query = None
        def filter(self, query):
            self.last_query = query
            return []

    class FakeRfmux:
        ReadoutModule = FakeReadoutModule
        class core:
            class schema:
                class CRS:
                    pass

    class FakeCRS(FakeRfmux.core.schema.CRS):
        def __init__(self):
            self.modules = FakeModules()

    monkeypatch.setattr(util, "rfmux", FakeRfmux)

    result = util.get_modules(d = FakeCRS(), module_idxs = [])

    assert result == []

################################################################################
################################ get_sample_freq ###############################
################################################################################
def test_get_sample_freq():
    for dec_stage in range(7):
        fs = 625e6 / (256 * 64 * 2 ** dec_stage)
        assert np.isclose(util.get_sample_freq(dec_stage), fs)

def test_get_sample_freq_invalid_decimation():
    msg = r"dec_stage must be an int in range \[0, 6\]"
    with pytest.raises(ValueError, match = msg):
        util.get_sample_freq(2.5)
    with pytest.raises(ValueError, match = msg):
        util.get_sample_freq(-1)
    with pytest.raises(ValueError, match = msg):
        util.get_sample_freq(7)


def test_get_sample_freq_accepts_numpy_int():
    assert np.isclose(
        util.get_sample_freq(np.int64(3)), 
        625e6 / (256 * 64 * 2 ** 3)
        )

################################################################################
################################ parser_to_zarr ################################
################################################################################
# NOTE: Tests for parser_to_zarr have been moved to test_parser_to_zarr.py
# to keep validation and functional tests together in a dedicated file.

################################################################################
############################ estimate_ts_data_size #############################
################################################################################
def _parse_size_line(line):
    label, value = line.split(':', 1)
    value = value.strip()
    num_str, unit = value.split()
    return label.strip(), float(num_str), unit


def _expected_sizes_mb(dec_stage, total_time, nmodules, max_ntones, ntones):
    sample_frequency = 625e6 / (256 * 64 * 2 ** dec_stage)
    size_per_ch = 4 * 2 * (total_time * sample_frequency)
    size_raw_mb = (size_per_ch * nmodules * max_ntones + 103) / 1e6
    size_proc_mb = (size_per_ch * ntones + 8 * ntones) / 1e6
    return size_raw_mb, size_proc_mb


def _to_mb(value, unit):
    if unit == 'GB':
        return value * 1e3
    return value


def test_estimate_ts_data_size_outputs_mb(capsys):
    args = dict(
        dec_stage = 6, 
        total_time = 10, 
        nmodules = 1, 
        max_ntones = 128, 
        ntones = 64
    )
    util.estimate_ts_data_size(**args)

    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 2

    raw_label, raw_val, raw_unit = _parse_size_line(out[0])
    proc_label, proc_val, proc_unit = _parse_size_line(out[1])

    assert raw_label.startswith('Raw parser data size')
    assert proc_label.startswith('Processed data size')
    assert raw_unit == 'MB'
    assert proc_unit == 'MB'

    exp_raw_mb, exp_proc_mb = _expected_sizes_mb(**args)
    assert np.isclose(
        _to_mb(raw_val, raw_unit), exp_raw_mb, rtol = 0.02, atol = 0.1
    )
    assert np.isclose(
        _to_mb(proc_val, proc_unit), exp_proc_mb, rtol = 0.02, atol = 0.1
    )


def test_estimate_ts_data_size_outputs_gb(capsys):
    args = dict(
        dec_stage = 0, 
        total_time = 100, 
        nmodules = 4, 
        max_ntones = 1024, 
        ntones = 2048
    )
    util.estimate_ts_data_size(**args)

    out = capsys.readouterr().out.strip().splitlines()
    assert len(out) == 2

    raw_label, raw_val, raw_unit = _parse_size_line(out[0])
    proc_label, proc_val, proc_unit = _parse_size_line(out[1])

    assert raw_label.startswith('Raw parser data size')
    assert proc_label.startswith('Processed data size')
    assert raw_unit == 'GB'
    assert proc_unit == 'GB'

    exp_raw_mb, exp_proc_mb = _expected_sizes_mb(**args)
    assert np.isclose(
        _to_mb(raw_val, raw_unit), exp_raw_mb, rtol = 0.02, atol = 0.1
    )
    assert np.isclose(
        _to_mb(proc_val, proc_unit), exp_proc_mb, rtol = 0.02, atol = 0.1
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(
            dec_stage = 7, total_time = 1, nmodules = 1, 
            max_ntones = 1, ntones = 1
        ),
        dict(
            dec_stage = -1, total_time = 1, nmodules = 1, 
            max_ntones = 1, ntones = 1
        ),
        dict(
            dec_stage = 1.5, total_time = 1, nmodules = 1, 
            max_ntones = 1, ntones = 1
        ),
        dict(
            dec_stage = 1, total_time = -0.1, nmodules = 1, 
            max_ntones = 1, ntones = 1
        ),
        dict(
            dec_stage = 1, total_time = 1, nmodules = 0, 
            max_ntones = 1, ntones = 1
        ),
        dict(
            dec_stage = 1, total_time = 1, nmodules = 5, 
            max_ntones = 1, ntones = 1
        ),
        dict(
            dec_stage = 1, total_time = 1, nmodules = 1, 
            max_ntones = -1, ntones = 1
        ),
        dict(
            dec_stage = 1, total_time = 1, nmodules = 1, 
            max_ntones = 1025, ntones = 1
        ),
        dict(
            dec_stage = 1, total_time = 1, nmodules = 1, 
            max_ntones = 1, ntones = -1
        ),
        dict(
            dec_stage = 1, total_time = 1, nmodules = 1, 
            max_ntones = 1, ntones = 4097
        ),
    ],
)
def test_estimate_ts_data_size_invalid_inputs(kwargs):
    with pytest.raises((TypeError, ValueError)):
        util.estimate_ts_data_size(**kwargs)

################################################################################
############################### interface_exists ###############################
################################################################################
def test_interface_exists(monkeypatch):
    def fake_if_nametoindex(name):
        if name == "lo":
            return 1
        raise OSError("no such interface")

    monkeypatch.setattr(util.socket, "if_nametoindex", fake_if_nametoindex)

    assert util.interface_exists("lo") is True
    assert util.interface_exists("non_existent_iface_12345") is False

def test_interface_exists_invalid_input():
    with pytest.raises(TypeError):
        util.interface_exists(123)


def test_interface_exists_passes_through_str(monkeypatch):
    captured = {}
    def fake_if_nametoindex(name):
        captured["name"] = name
        return 1

    monkeypatch.setattr(util.socket, "if_nametoindex", fake_if_nametoindex)

    assert util.interface_exists("eth0") is True
    assert captured["name"] == "eth0"

################################################################################
########## write_system_cfg_to_zarr and write_acq_cfg_to_zarr ###########
################################################################################

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

def test_write_system_cfg_to_zarr_writes_expected_arrays(tmp_path):
    grp = zarr.open_group(tmp_path / "config.zarr", mode="w")
    crs = make_dummy_crs()

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=Warning, message=".*Zarr V3 specification.*")
        util.write_system_cfg_to_zarr(crs, grp)

    # Check that all config is stored as attributes (ch_map and sample_freq are in write_acq_cfg_to_zarr now)
    expected_attrs = {
        "nco_module1",
        "firmware_version",
        "analog_bank_high",
        "bw",
        "clock_source",
        "extended_bw",
        "serial_number",
        "rfmux_version",
        "citkid_version",
    }
    assert expected_attrs.issubset(set(grp.attrs.keys()))

    assert np.isclose(grp.attrs["nco_module1"], 4.0e9)
    assert grp.attrs["firmware_version"] == "1.2.3"
    assert grp.attrs["analog_bank_high"] is False
    assert np.isclose(grp.attrs["bw"], 500e6)
    assert grp.attrs["clock_source"] == "VCXO"
    assert grp.attrs["extended_bw"] is False
    assert grp.attrs["serial_number"] == 27
    assert grp.attrs["rfmux_version"] == "1.3.2"
    assert grp.attrs["citkid_version"] == "1.0.0.dev0"


def test_write_system_cfg_to_zarr_invalid_crs_type(tmp_path):
    grp = zarr.open_group(tmp_path / "config.zarr", mode="w")

    class NotCRS:
        pass

    not_crs = NotCRS()
    not_crs.CRS = DummyCRS

    with pytest.raises(TypeError, match="crs must be an instance of CRS class"):
        util.write_system_cfg_to_zarr(not_crs, grp)


def test_write_system_cfg_to_zarr_invalid_grp_type():
    crs = make_dummy_crs()
    with pytest.raises(TypeError, match="grp must be a zarr Group instance"):
        util.write_system_cfg_to_zarr(crs, "not_a_group")


@pytest.mark.parametrize("missing_name", [
    "nco_freqs",
    "firmware_release",
    "analog_bank_high",
    "bw",
    "clock_source",
    "extended_bw",
    "serial_number",
    "rfmux_version",
    "citkid_version",
])
def test_write_system_cfg_to_zarr_missing_crs_attribute(tmp_path, missing_name):
    grp = zarr.open_group(tmp_path / "config.zarr", mode="w")
    crs = make_dummy_crs()
    delattr(crs, missing_name)

    with pytest.raises(ValueError, match=f"crs is missing attribute '{missing_name}'"):
        util.write_system_cfg_to_zarr(crs, grp)


@pytest.mark.parametrize("conflict_name", [
    "nco_freqs",
    "firmware_release",
    "analog_bank_high",
    "bw",
    "clock_source",
    "extended_bw",
    "serial_number",
    "rfmux_version",
    "citkid_version",
])
def test_write_system_cfg_to_zarr_conflicting_dataset(tmp_path, conflict_name):
    grp = zarr.open_group(tmp_path / "config.zarr", mode="w")
    # Map the old dataset names to the actual names saved
    # All are now stored as attributes
    if conflict_name == "nco_freqs":
        # nco_freqs is stored as attribute "nco_module1"
        grp.attrs["nco_module1"] = 4.0e9
        error_msg = "Zarr group already contains attribute 'nco_module1'"
    elif conflict_name == "firmware_release":
        # firmware_release is stored as attribute "firmware_version"
        grp.attrs["firmware_version"] = "1.2.3"
        error_msg = "Zarr group already contains attribute 'firmware_version'"
    else:
        # All others use the same name as attributes
        grp.attrs[conflict_name] = "existing_value"
        error_msg = f"Zarr group already contains attribute '{conflict_name}'"
    
    crs = make_dummy_crs()

    with pytest.raises(ValueError, match=error_msg.replace('[', r'\[').replace(']', r'\]')):
        util.write_system_cfg_to_zarr(crs, grp)


def test_write_system_cfg_to_zarr_invalid_firmware_version(tmp_path):
    grp = zarr.open_group(tmp_path / "config.zarr", mode="w")
    crs = make_dummy_crs(firmware_release=DummyFirmwareRelease(version=123))

    with pytest.raises(ValueError, match="crs.firmware_release.version must be a string"):
        util.write_system_cfg_to_zarr(crs, grp)


def test_write_acq_cfg_to_zarr_writes_expected_arrays(tmp_path):
    grp = zarr.open_group(tmp_path / "config.zarr", mode="w")
    crs = make_dummy_crs()

    util.write_acq_cfg_to_zarr(crs, grp)

    # Check decimation parameters as attributes
    expected_attrs = {"dec_module_idxs", "dec_short", "dec_stage", "sample_freq"}
    assert expected_attrs.issubset(set(grp.attrs.keys()))
    assert grp.attrs["dec_module_idxs"] == [1]
    assert grp.attrs["dec_short"] is False
    assert grp.attrs["dec_stage"] == 6
    assert np.isclose(grp.attrs["sample_freq"], 976.0)
    
    # Check ch_map is stored as array
    expected_arrays = {"chs_module1"}
    assert expected_arrays.issubset(set(grp.keys()))
    np.testing.assert_array_equal(grp["chs_module1"][:], np.array([0, 1], dtype=np.int32))


@pytest.mark.parametrize("missing_name", [
    "dec_module_idxs",
    "dec_short",
    "dec_stage",
    "ch_map",
    "sample_freq",
])
def test_write_acq_cfg_to_zarr_missing_crs_attribute(tmp_path, missing_name):
    grp = zarr.open_group(tmp_path / "config.zarr", mode="w")
    crs = make_dummy_crs()
    delattr(crs, missing_name)

    with pytest.raises(ValueError, match=f"crs is missing attribute '{missing_name}'"):
        util.write_acq_cfg_to_zarr(crs, grp)


@pytest.mark.parametrize("conflict_name", [
    "dec_module_idxs",
    "dec_short",
    "dec_stage",
    "sample_freq",
    "chs_module1",
])
def test_write_acq_cfg_to_zarr_conflicting_dataset(tmp_path, conflict_name):
    grp = zarr.open_group(tmp_path / "config.zarr", mode="w")
    if conflict_name == "chs_module1":
        # ch_map is stored as array
        grp.create_array(conflict_name, data=np.array([1]))
        error_msg = f"Zarr group already contains dataset '{conflict_name}'"
    else:
        # decimation parameters are stored as attributes
        grp.attrs[conflict_name] = "existing_value"
        error_msg = f"Zarr group already contains attribute '{conflict_name}'"
    crs = make_dummy_crs()

    with pytest.raises(ValueError, match=error_msg.replace('[', r'\[').replace(']', r'\]')):
        util.write_acq_cfg_to_zarr(crs, grp)


def test_write_acq_cfg_to_zarr_invalid_ch_map(tmp_path):
    grp = zarr.open_group(tmp_path / "config.zarr", mode="w")
    crs = make_dummy_crs(ch_map={1: ["a"]})

    with pytest.raises((TypeError, ValueError)):
        util.write_acq_cfg_to_zarr(crs, grp)

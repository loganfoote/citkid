import pytest
import numpy as np
import rfmux
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
        util.get_modules(d="not_a_crs", module_indices=[1, 2])

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
        util.get_modules(d=FakeCRS(), module_indices=123)

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
        util.get_modules(d=FakeCRS(), module_indices=[1, "2"])

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

################################################################################
################################ parser_to_zarr ################################
################################################################################

################################################################################
################################## import_ts ###################################
################################################################################

################################################################################
############################ estimate_ts_data_size #############################
################################################################################

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
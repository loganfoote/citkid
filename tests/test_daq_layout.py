from types import SimpleNamespace
import numpy as np
import pytest
import zarr
from citkid import daq_layout as dl


def _crs(**changes):
    """
    Make a fake configured CRS with the attributes the layout reads.

    Parameters:
    **changes: attributes to replace.

    Returns:
    crs (SimpleNamespace): fake CRS.
    """
    cfg = dict(full_scale_dbm = 7.0, adc_attenuation_db = 0.0,
               nyquist_zone = 1, cable_length_m = 0.0,
               adc_calibration_mode = 'AUTO', adc_autocal = True)
    crs = SimpleNamespace(
        serial_number = 46, firmware_release = SimpleNamespace(
            version = '1.6.0'), rfmux_version = '1.4.1',
        active_module_idxs = [2, 3], nco_freqs = {2: 1.432e9, 3: 1.956e9},
        analog_bank_high = False, extended_bw = True, bw = 600e6,
        clock_source = 'VCXO',
        module_cfg = {2: dict(cfg), 3: dict(cfg), 1: dict(cfg)})
    for key, value in changes.items():
        setattr(crs, key, value)
    return crs


@pytest.fixture
def store(tmp_path):
    """
    Create an empty device store.

    Returns:
    root (zarr.Group): store root.
    """
    return dl.create_store(tmp_path / 'dev.zarr', 'devA', 'cd1',
                           setup = dict(fridge_ch = 3))


################################################################################
################################ Store and groups ##############################
################################################################################
def test_create_store_attrs_and_reopen(tmp_path, store):
    """
    Check the root attributes, and that reopening for the same device keeps
    them and refuses another device or cooldown.
    """
    a = dict(store.attrs)
    assert a['layout_version'] == dl.LAYOUT_VERSION
    assert (a['device'], a['cooldown']) == ('devA', 'cd1')
    assert a['setup'] == {'fridge_ch': 3}
    again = dl.create_store(tmp_path / 'dev.zarr', 'devA', 'cd1',
                            setup = dict(fridge_ch = 9))
    assert again.attrs['setup'] == {'fridge_ch': 3}
    with pytest.raises(ValueError, match = 'device'):
        dl.create_store(tmp_path / 'dev.zarr', 'devB', 'cd1')
    with pytest.raises(ValueError, match = 'cooldown'):
        dl.create_store(tmp_path / 'dev.zarr', 'devA', 'cd2')


def test_open_store_rejects_non_layout(tmp_path):
    """
    Check that a plain zarr store is not accepted as a layout store, nor a
    store from a newer layout version.
    """
    zarr.open_group(str(tmp_path / 'plain.zarr'), mode = 'w')
    with pytest.raises(ValueError, match = 'not a DAQ layout store'):
        dl.open_store(tmp_path / 'plain.zarr')
    root = dl.create_store(tmp_path / 'new.zarr', 'd', 'c')
    root.attrs['layout_version'] = dl.LAYOUT_VERSION + 1
    with pytest.raises(ValueError, match = 'newer'):
        dl.open_store(tmp_path / 'new.zarr')


def test_indices(store):
    """
    Check index listing, next and latest, including unpadded sorting.
    """
    tunes = store.require_group('x')
    assert dl.next_index(tunes, 'tune') == 0
    with pytest.raises(KeyError):
        dl.latest_index(tunes, 'tune')
    for i in (0, 2, 10):
        tunes.create_group(f'tune{i}')
    tunes.create_group('tune_x')
    assert dl.list_indices(tunes, 'tune') == [0, 2, 10]
    assert dl.next_index(tunes, 'tune') == 11
    assert dl.latest_index(tunes, 'tune') == 10
    grp, idx = dl.new_indexed_group(tunes, 'tune', 'raw')
    assert idx == 11 and grp.path.endswith('tune11')


@pytest.mark.parametrize('kind', ['decision', 'analysis'])
def test_create_group_overwrite(store, kind):
    """
    Check that decision and analysis groups are replaced only with
    ``overwrite``.
    """
    g = dl.create_group(store, 'a', kind, attrs = dict(n = 1))
    assert g.attrs['kind'] == kind and g.attrs['n'] == 1
    assert 'created' in g.attrs and 'citkid_version' in g.attrs
    with pytest.raises(ValueError, match = 'overwrite'):
        dl.create_group(store, 'a', kind)
    g = dl.create_group(store, 'a', kind, attrs = dict(n = 2),
                        overwrite = True)
    assert g.attrs['n'] == 2


def test_raw_groups_are_write_once(store):
    """
    Check that raw groups are never replaced, and nothing replaces a group
    with a raw one.
    """
    dl.create_group(store, 'r', 'raw')
    with pytest.raises(ValueError, match = 'raw'):
        dl.create_group(store, 'r', 'raw', overwrite = True)
    with pytest.raises(ValueError, match = 'raw'):
        dl.create_group(store, 'r', 'analysis', overwrite = True)
    dl.create_group(store, 'd', 'decision')
    with pytest.raises(ValueError, match = 'raw'):
        dl.create_group(store, 'd', 'raw', overwrite = True)
    with pytest.raises(ValueError, match = 'kind'):
        dl.create_group(store, 'k', 'other')


def test_require_group(store):
    """
    Check that require_group creates once, reopens, and refuses another
    kind.
    """
    g = dl.require_group(store, 'vna', 'raw')
    g.attrs['x'] = 1
    again = dl.require_group(store, 'vna', 'raw')
    assert again.attrs['x'] == 1 and again.attrs['kind'] == 'raw'
    with pytest.raises(ValueError, match = "kind 'raw'"):
        dl.require_group(store, 'vna', 'analysis')


def test_write_array_sharding_and_strings(store, monkeypatch):
    """
    Check long arrays are sharded into one file, and strings round-trip.
    """
    monkeypatch.setattr(dl, '_CHUNK', 100)
    a = dl.write_array(store, 'long', np.arange(250.))
    assert a.chunks == (100,) and a.shards == (300,)
    np.testing.assert_array_equal(a[:], np.arange(250.))
    s = dl.write_array(store, 's', np.array(['vna/sweep0', 'x']))
    assert list(s[:]) == ['vna/sweep0', 'x']
    e = dl.write_array(store, 'e', np.array([], dtype = str))
    assert e.shape == (0,)


################################################################################
############################# CRS configuration ################################
################################################################################
def test_crs_identity_recorded_then_checked(store):
    """
    Check that the first acquisition records the CRS identity and a later
    one with different firmware is refused.
    """
    dl.check_crs_identity(store, _crs())
    assert store.attrs['crs_serial_number'] == 46
    assert store.attrs['firmware_version'] == '1.6.0'
    dl.check_crs_identity(store, _crs())
    with pytest.raises(RuntimeError, match = 'new store'):
        dl.check_crs_identity(store, _crs(firmware_release = SimpleNamespace(
            version = '2.0.0')))


def test_tune_config_contents():
    """
    Check the tuning-level settings: NCOs, modules, bank, bandwidth, and
    only the tuning-level module settings of this session's modules.
    """
    cfg = dl.tune_config(_crs())
    assert cfg['ncos'] == {'2': 1.432e9, '3': 1.956e9}
    assert cfg['modules'] == [2, 3]
    assert cfg['extended_bw'] is True and cfg['analog_bank_high'] is False
    assert set(cfg['module_settings']) == {'2', '3'}
    assert set(cfg['module_settings']['2']) == set(dl.TUNE_MODULE_SETTINGS)


def test_acquisition_settings_contents():
    """
    Check the per-acquisition settings: clock and ADC calibration only.
    """
    acq = dl.acquisition_settings(_crs(clock_source = 'SMA'))
    assert acq['clock_source'] == 'SMA'
    assert set(acq['module_settings']) == {'2', '3'}
    assert set(acq['module_settings']['3']) == set(dl.ACQ_MODULE_SETTINGS)


def test_check_tune_config(store):
    """
    Check that a matching board passes (also with extra saved keys and
    float noise) and a changed NCO, bandwidth or module setting fails.
    """
    tune = dl.create_group(store, 'tune0', 'decision')
    cfg = dl.tune_config(_crs())
    cfg['plan'] = {'tone_sets': 2}
    dl.write_tune_config(tune, cfg)
    assert tune['crs_config'].attrs['kind'] == 'decision'
    dl.check_tune_config(tune, _crs(nco_freqs = {2: 1.432e9 + 1e-7,
                                                 3: 1.956e9}))
    # Per-acquisition settings may change
    crs = _crs(clock_source = 'SMA')
    crs.module_cfg[2]['adc_autocal'] = False
    dl.check_tune_config(tune, crs)
    with pytest.raises(RuntimeError, match = 'ncos.2'):
        dl.check_tune_config(tune, _crs(nco_freqs = {2: 1.5e9, 3: 1.956e9}))
    with pytest.raises(RuntimeError, match = 'extended_bw'):
        dl.check_tune_config(tune, _crs(extended_bw = False))
    crs = _crs()
    crs.module_cfg[3]['full_scale_dbm'] = 5.0
    with pytest.raises(RuntimeError, match = 'full_scale_dbm'):
        dl.check_tune_config(tune, crs)
    with pytest.raises(RuntimeError, match = 'modules'):
        dl.check_tune_config(tune, _crs(active_module_idxs = [2]))


################################################################################
################################# VNA sweeps ###################################
################################################################################
def _bands(nb = 2):
    """
    Make band arrays for ``nb`` bands.

    Parameters:
    nb (int): number of bands. Default 2.

    Returns:
    bands (dict): ``VNA_BAND_ARRAYS`` by name.
    """
    return dict(chunk_edges = [[i * 5e8, (i + 1) * 5e8] for i in range(nb)],
                chunk_ncos = [(i + 0.5) * 5e8 for i in range(nb)],
                chunk_modules = [1] * nb, chunk_passes = list(range(nb)),
                chunk_zones = [1] * nb, chunk_nchannels = [1024] * nb,
                chunk_times = [1e9 + i for i in range(nb)])


def test_vna_sweep_round_trip(store):
    """
    Check that a VNA sweep is saved sorted, with its band arrays and
    settings, and that the latest sweep is loaded by default.
    """
    f = np.array([3e8, 1e8, 2e8, 7e8])
    z = f / 1e9 + 1j
    for k in range(2):
        grp, _ = dl.new_indexed_group(store.require_group('vna'), 'sweep',
                                      'raw')
        dl.save_vna_sweep(grp, f * (k + 1), z, **_bands(),
                          settings = dict(amplitude = -50., nsamps = 10))
    f0, z0, bands = dl.load_vna_sweep(store, 0, bands = True)
    np.testing.assert_array_equal(f0, np.sort(f))
    np.testing.assert_array_equal(z0, np.sort(f) / 1e9 + 1j)
    assert bands['chunk_edges'].shape == (2, 2)
    assert bands['chunk_modules'].dtype == np.int32
    assert store['vna/sweep0'].attrs['amplitude'] == -50.
    f1, _ = dl.load_vna_sweep(store)
    np.testing.assert_array_equal(f1, np.sort(f) * 2)


def test_vna_sweep_bad_shapes(store):
    """
    Check that mismatched f/z or band arrays are refused.
    """
    grp = dl.create_group(store, 'vna/sweep0', 'raw')
    with pytest.raises(ValueError, match = 'same length'):
        dl.save_vna_sweep(grp, [1., 2.], [1j], **_bands())
    bands = _bands()
    bands['chunk_ncos'] = [1.]
    with pytest.raises(ValueError, match = 'nb entries'):
        dl.save_vna_sweep(grp, [1., 2.], [1j, 2j], **bands)


################################################################################
############################ Resonators and tones ##############################
################################################################################
def test_add_resonators_appends(store):
    """
    Check that resonators get sequential indices that later additions never
    change.
    """
    idx = dl.add_resonators(store, [1e9, 2e9], 2e4, 'vna/sweep0',
                            Qr = [2.1e4, np.nan])
    np.testing.assert_array_equal(idx, [0, 1])
    idx = dl.add_resonators(store, [1.5e9], [3e4], 'vna/sweep1')
    np.testing.assert_array_equal(idx, [2])
    res = dl.load_resonators(store)
    np.testing.assert_array_equal(res['res_idxs_all'], [0, 1, 2])
    np.testing.assert_array_equal(res['fres_all'], [1e9, 2e9, 1.5e9])
    np.testing.assert_array_equal(res['qres_all'], [2e4, 2e4, 3e4])
    np.testing.assert_array_equal(res['Qr_all'], [2.1e4, np.nan, np.nan])
    assert list(res['source_all']) == ['vna/sweep0'] * 2 + ['vna/sweep1']
    assert store['resonators'].attrs['kind'] == 'decision'


def test_tones_round_trip(store):
    """
    Check saving and loading tones, the default tone set, and the length
    check.
    """
    grp = dl.create_group(store, 'tones', 'decision')
    dl.save_tones(grp, [1e9, 2e9], 2e4, -50., [0, -1],
                  source = dict(fres = 'tune0/fq_sweep/fres_opt'))
    t = dl.load_tones(grp)
    np.testing.assert_array_equal(t['qres'], [2e4, 2e4])
    np.testing.assert_array_equal(t['ares'], [-50., -50.])
    np.testing.assert_array_equal(t['tone_set'], [0, 0])
    assert t['res_idxs'].dtype == np.int32
    assert grp.attrs['source'] == {'fres': 'tune0/fq_sweep/fres_opt'}
    with pytest.raises(ValueError, match = 'same length'):
        dl.save_tones(dl.create_group(store, 't2', 'decision'), [1e9, 2e9],
                      2e4, -50., [0])


################################################################################
################################ Port, archive #################################
################################################################################
def _tuned_store(path):
    """
    Make a store with resonators, a VNA sweep, and a tuning with raw data,
    analysis, tones, and crs_config.

    Parameters:
    path (pathlib.Path): store path.

    Returns:
    root (zarr.Group): store root.
    """
    root = dl.create_store(path, 'devA', 'cd1', setup = dict(fridge_ch = 3))
    vna = dl.create_group(root, 'vna/sweep0', 'raw')
    dl.save_vna_sweep(vna, [1e8, 2e8], [1j, 1j], **_bands(1))
    dl.add_resonators(root, [1e9, 1.1e9, 2e9], 2e4, 'vna/sweep0')
    tune = dl.create_group(root, 'tune0', 'decision')
    dl.write_tune_config(tune, dl.tune_config(_crs()))
    dl.create_group(tune, 'power_sweep', 'raw')
    dl.create_group(tune, 'power_fits', 'analysis')
    dl.create_group(tune, 'sweep0', 'raw')
    tones = dl.create_group(tune, 'tones', 'decision')
    dl.save_tones(tones, [1e9, 2e9], 3e4, [-50., -48.], [0, 2],
                  tone_set = [0, 1], source = dict(ares = 'tune0/x'))
    return root


def test_port_tuning(tmp_path):
    """
    Check that porting copies resonators, tones and crs_config only, with
    provenance, and that a second port goes to the next tuning.
    """
    src = _tuned_store(tmp_path / 'a.zarr')
    k = dl.port_tuning(tmp_path / 'a.zarr', 0, tmp_path / 'b.zarr',
                       cooldown = 'cd2')
    dst = dl.open_store(tmp_path / 'b.zarr')
    assert k == 0
    assert (dst.attrs['device'], dst.attrs['cooldown']) == ('devA', 'cd2')
    assert dst.attrs['setup'] == {'fridge_ch': 3}
    assert sorted(dst.group_keys()) == ['resonators', 'tune0']
    assert sorted(dst['tune0'].group_keys()) == ['crs_config', 'tones']
    t = dl.load_tones(dst['tune0/tones'])
    np.testing.assert_array_equal(t['tone_set'], [0, 1])
    np.testing.assert_array_equal(t['ares'], [-50., -48.])
    pf = dst['tune0/tones'].attrs['ported_from']
    assert pf['group'] == 'tune0/tones' and pf['source'] == {
        'ares': 'tune0/x'}
    assert 'source' not in dst['tune0/tones'].attrs
    assert dst['tune0/crs_config'].attrs['config'] == \
        src['tune0/crs_config'].attrs['config']
    np.testing.assert_array_equal(dl.load_resonators(dst)['fres_all'],
                                  [1e9, 1.1e9, 2e9])
    assert 'crs_serial_number' not in dst.attrs
    # Same tuning again: the next index; dst keeps its cooldown and setup
    assert dl.port_tuning(src, 0, tmp_path / 'b.zarr', cooldown = 'x',
                          setup = dict(y = 1)) == 1
    dst = dl.open_store(tmp_path / 'b.zarr')
    assert dst.attrs['cooldown'] == 'cd2'
    assert dst.attrs['setup'] == {'fridge_ch': 3}
    assert not src.attrs.get('ported_from')


def test_port_tuning_refusals(tmp_path):
    """
    Check that porting refuses mismatched resonators, another device, and
    a tuning without tones.
    """
    _tuned_store(tmp_path / 'a.zarr')
    other = dl.create_store(tmp_path / 'c.zarr', 'devA', 'cd1')
    dl.add_resonators(other, [5e8], 2e4, 'vna/sweep0')
    with pytest.raises(ValueError, match = 'resonators differ'):
        dl.port_tuning(tmp_path / 'a.zarr', 0, tmp_path / 'c.zarr')
    dl.create_store(tmp_path / 'd.zarr', 'devB', 'cd1')
    with pytest.raises(ValueError, match = 'device'):
        dl.port_tuning(tmp_path / 'a.zarr', 0, tmp_path / 'd.zarr')
    src = dl.open_store(tmp_path / 'a.zarr', mode = 'a')
    dl.create_group(src, 'tune1', 'decision')
    with pytest.raises(KeyError, match = 'tones'):
        dl.port_tuning(src, 1, tmp_path / 'e.zarr')


def test_archive_store(tmp_path):
    """
    Check that an archived store opens read-only with the same contents,
    can be ported from, and an existing zip isn't overwritten.
    """
    _tuned_store(tmp_path / 'a.zarr')
    zp = dl.archive_store(tmp_path / 'a.zarr')
    assert zp.endswith('a.zarr.zip')
    z = dl.open_store(zp)
    np.testing.assert_array_equal(dl.load_resonators(z)['fres_all'],
                                  [1e9, 1.1e9, 2e9])
    assert dl.tree(z) == dl.tree(tmp_path / 'a.zarr')
    with pytest.raises(ValueError, match = 'read-only'):
        dl.open_store(zp, mode = 'a')
    assert dl.port_tuning(zp, 0, tmp_path / 'b.zarr') == 0
    with pytest.raises(FileExistsError):
        dl.archive_store(tmp_path / 'a.zarr')


def test_tree(tmp_path):
    """
    Check the text tree: natural order, kinds, and depth limit.
    """
    root = _tuned_store(tmp_path / 'a.zarr')
    for i in (2, 10):
        dl.create_group(root['tune0'], f'sweep{i}', 'raw')
    text = dl.tree(root)
    lines = text.splitlines()
    assert lines[0] == 'resonators/  [decision]'
    sweeps = [s.strip() for s in lines if s.strip().startswith('sweep')]
    assert sweeps[:3] == ['sweep0/  [raw]', 'sweep2/  [raw]',
                          'sweep10/  [raw]']
    assert all(not s.startswith(' ') for s in dl.tree(root, 1).splitlines())

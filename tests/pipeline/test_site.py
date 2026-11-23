"""
Tests of citkid.pipeline.site helpers (the whole site is built in
tests/procedures/test_scripts.py).
"""
import pytest
from citkid import daq_layout as dl
from citkid.pipeline import site


def test_store_suffix(tmp_path):
    """
    The suffix follows the device name; when the file name does not
    start with the device, it is what follows the last underscore.
    """
    root = dl.create_store(tmp_path / 'devA_screen.zarr', 'devA', 'cd')
    assert site.store_suffix(root) == ('devA_screen', 'screen')
    root = dl.create_store(tmp_path / 'L261022p1-die3_screen.zarr',
                           'L261002p1-die3', 'cd')
    assert site.store_suffix(root) == ('L261022p1-die3_screen', 'screen')
    root = dl.create_store(tmp_path / 'devB.zarr', 'devB', 'cd')
    assert site.store_suffix(root) == ('devB', '')


@pytest.mark.parametrize('schedule, text', [
    (None, ''),
    (dict(fmin_mhz = None), ''),
    (dict(fmin_mhz = 314.02, fmax_mhz = 2421.2, n = 938, outliers_mhz = [],
          modules = [4], bank = 'low', ncos_mhz = [666.77, 817.73]),
     '314 to 2421 MHz'),
    (dict(fmin_mhz = 500., fmax_mhz = 900.4, n = 12,
          outliers_mhz = [120., 2900.1, 3000.],
          modules = [1, 2], bank = None, ncos_mhz = []),
     '500 to 900 MHz, 1 below, 2 above'),
])
def test_schedule_text(schedule, text):
    """
    The schedule cell names the range and counts the outliers on
    each side.
    """
    assert site.schedule_text(schedule) == text


def test_frequency_schedule(tmp_path):
    """
    The schedule comes from the latest tuning's resonator list and its
    CRS configuration; an outlying resonance is listed apart.
    """
    import numpy as np
    root = dl.create_store(tmp_path / 'devA_screen.zarr', 'devA', 'cd')
    fres = list(np.linspace(1e9, 1.2e9, 20)) + [3.5e9]
    dl.save_resonators(root, 0, fres, 2e4, 'x')
    tune, _ = dl.new_tuning(root, 0)
    dl.write_tune_config(tune, dict(
        modules = [3], analog_bank_high = True, tone_sets = [{'3': 1.1e9}]))
    inv = site.inventory(root)
    sched = site.frequency_schedule(root, inv)
    assert sched['n'] == 21 and sched['outliers_mhz'] == [3500.]
    assert sched['fmin_mhz'] == pytest.approx(1000.)
    assert sched['fmax_mhz'] == pytest.approx(1200.)
    assert sched['modules'] == [3] and sched['bank'] == 'high'
    assert sched['ncos_mhz'] == [1100.]
    assert site.schedule_text(sched) == '1000 to 1200 MHz, 1 above'
    assert site.frequency_schedule(
        dl.create_store(tmp_path / 'b.zarr', 'b', 'cd'), {}) is None


def test_comments(tmp_path):
    """
    Comments are read from the device folder's comments.json, summed up
    for the top page, rendered on the device page, and kept when the
    folder is cleared for an overwrite.
    """
    import json
    dev = tmp_path / 'cd' / 'dev'
    dev.mkdir(parents = True)
    assert site.load_comments(str(dev)) == []
    assert site.comments_summary([]) == ''
    assert 'No comments yet' in site.comments_html([])
    (dev / 'comments.json').write_text(json.dumps([
        dict(who = 'LF', when = '2026-11-23 10:00:00', text = 'looks\nfine'),
        dict(who = '', when = '2026-11-23 11:00:00',
             text = 'x' * 100 + ' <b>')]), 'utf-8')
    comments = site.load_comments(str(dev))
    assert [c['who'] for c in comments] == ['LF', '']
    assert comments[0]['id'] == '' and comments[0]['edited'] == ''
    text = site.comments_summary(comments)
    assert text.startswith('2; latest 2026-11-23 11:00:00: xxx')
    assert text.endswith('...') and len(text) < 120
    html = site.comments_html(comments)
    assert 'looks<br>fine' in html and '&lt;b&gt;' in html
    edited = site.comments_html([dict(id = 'c1', who = 'LF',
                                      when = '2026-11-23 10:00:00',
                                      text = 'x', edited = '2026-11-23 '
                                      '12:00:00')])
    assert '(edited 2026-11-23 12:00:00)' in edited
    for word in ('add', 'edit', 'delete', 'uniqid', 'array_splice',
                 'LOCK_EX'):
        assert word in site.COMMENTS_PHP_SOURCE
    for word in ('data-edit', 'data-delete', 'Save edit', 'confirm('):
        assert word in site.COMMENTS_SCRIPT
    (dev / 'index.html').write_text('x')
    (dev / 'figures').mkdir()
    (dev / 'figures' / 'a.png').write_bytes(b'0')
    site.clear_device_dir(str(dev))
    assert sorted(p.name for p in dev.iterdir()) == ['comments.json']
    (dev / 'comments.json').write_text('not json', 'utf-8')
    assert site.load_comments(str(dev)) == []
    row = site.index_row(dict(device = 'dev'), 'cd', 'dev', comments)
    assert row['Comments'] == text and list(row) == site.INDEX_COLUMNS


def test_setup_note():
    """
    An item's readout chain is noted only where it differs from the
    store's, unset values included.
    """
    from citkid import daq_layout as dl
    base = dict(fridge_ch = '3', dac_attenuation = '20 dB', warm_amps = '')
    assert site.setup_note(None, base) == ''
    assert site.setup_note(dict(base), base) == ''
    assert site.setup_note(dict(base, dac_attenuation = '30 dB'), base) \
        == '; setup: dac_attenuation 30 dB'
    text = site.setup_note(dict(fridge_ch = '3', warm_amps = 'LNF x2'), base)
    assert 'warm_amps LNF x2' in text and 'dac_attenuation (unset)' in text
    assert dl.setup_text(base) == 'fridge_ch 3, dac_attenuation 20 dB'
    assert dl.setup_text(None) == ''

"""
Tests of citkid.pipeline.interactive.marks: the tone-marks dialogs and
the registry helpers the windows share.
"""
import numpy as np
import pytest
import zarr
from pyqtgraph.Qt import QtWidgets
from citkid import daq_layout as dl
from citkid.pipeline.interactive import marks
from citkid.qt_compat import get_qapp


@pytest.fixture
def qapp():
    """
    Return the QApplication.

    Returns:
    app (QApplication): the application.
    """
    return get_qapp()


def test_registry_and_record(tmp_path):
    """
    A zarr group of a store gives the writable root and its path; a mock
    or None gives nothing; ``record`` skips repeats and clears without a
    bad mark.
    """
    root = zarr.open_group(tmp_path / 's.zarr', mode = 'a')
    grp = root.create_group('tune0/fq_sweep')
    r, stage = marks.registry(grp)
    assert stage == 'tune0/fq_sweep' and r is not None
    assert marks.registry(None) == (None, '')
    assert marks.registry(object()) == (None, '')
    assert marks.record(None, 5, 'bad', 'x', stage) is False
    assert marks.record(r, None, 'bad', 'x', stage) is False
    # A clear without a bad mark is not recorded
    assert marks.record(r, 5, 'clear', '', stage) is False
    assert marks.record(r, 5, 'bad', 'bifurcated', stage) is True
    assert marks.record(r, 5, 'bad', 'bifurcated', stage) is False
    assert marks.record(r, 5, 'bad', 'other', stage) is True
    assert marks.record(r, 5, 'clear', 'fine', stage) is True
    assert marks.record(r, 5, 'clear', 'fine', stage) is False
    again = zarr.open_group(tmp_path / 's.zarr', mode = 'r')
    kinds = [e['kind'] for e in dl.tone_marks(again, 5)]
    assert kinds == ['bad', 'bad', 'clear'] and dl.bad_tones(again) == {}
    # In-memory groups work too (the tests' windows)
    mem = zarr.group()
    r2, stage2 = marks.registry(mem)
    assert r2 is not None and stage2 == ''
    assert marks.record(r2, 1, 'note', 'hello', '') is True
    assert dl.tone_marks(mem, 1)[0]['text'] == 'hello'


def test_mark_label(tmp_path):
    root = zarr.open_group(tmp_path / 's.zarr', mode = 'a')
    assert marks.mark_label(None, 3) == '' and marks.mark_label(root, 3) == ''
    dl.add_tone_mark(root, 3, 'note', 'first')
    assert marks.mark_label(root, 3) == '1 note'
    dl.add_tone_mark(root, 3, 'note', 'second')
    dl.add_tone_mark(root, 3, 'bad', 'high noise', 'ts_cal0/set0')
    assert marks.mark_label(root, 3) == \
        'BAD: high noise (ts_cal0/set0), 2 notes'


def test_dialogs(qapp, monkeypatch):
    """
    The dialogs return the chosen reason, note or keep/remove choice when
    accepted, and None when cancelled.
    """
    monkeypatch.setattr(QtWidgets.QDialog, 'exec_',
                        lambda self: QtWidgets.QDialog.Accepted)
    assert marks.ask_reason(None, 'Mark res 1 bad') == dl.BAD_REASONS[0]
    assert 'shallow resonance' in dl.BAD_REASONS
    assert marks.ask_reason(None, 'Mark res 1 bad',
                            current = 'high noise') == 'high noise'
    assert marks.ask_reason(None, 't', reasons = ('x', 'y')) == 'x'
    assert marks.ask_note(None, 'Note', text = '  a note ') == 'a note'
    assert marks.ask_note(None, 'Note') is None
    assert marks.ask_all_bad(None, 't') == ('keep', '')
    monkeypatch.setattr(QtWidgets.QDialog, 'exec_',
                        lambda self: QtWidgets.QDialog.Rejected)
    assert marks.ask_reason(None, 't') is None
    assert marks.ask_note(None, 't', text = 'x') is None
    assert marks.ask_all_bad(None, 't') == (None, '')

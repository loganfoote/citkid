"""
Tests for RFSOC utility methods and module-level helpers.

These tests require no hardware; they only exercise local logic.
"""

import os
import time
import shutil
import pytest
import numpy as np
from unittest.mock import patch


# ---------------------------------------------------------------------------
# separate_iq_data
# ---------------------------------------------------------------------------

class TestSeparateIqData:
    def test_splits_complex_to_real_imag(self, tmp_path):
        from citkid.primecam.instrument import separate_iq_data
        f = np.array([1.0 + 0j, 2.0 + 0j, 3.0 + 0j])
        z = np.array([1 + 2j, 3 + 4j, 5 + 6j])
        path = str(tmp_path / 'data.npy')
        np.save(path, [f, z])

        separate_iq_data(path)

        result = np.load(path)
        np.testing.assert_array_almost_equal(result[0], [1.0, 2.0, 3.0])  # f
        np.testing.assert_array_almost_equal(result[1], [1.0, 3.0, 5.0])  # i
        np.testing.assert_array_almost_equal(result[2], [2.0, 4.0, 6.0])  # q

    def test_output_is_real(self, tmp_path):
        from citkid.primecam.instrument import separate_iq_data
        f = np.array([1.0 + 0j])
        z = np.array([1 + 2j])
        path = str(tmp_path / 'data.npy')
        np.save(path, [f, z])
        separate_iq_data(path)
        result = np.load(path)
        assert result.dtype.kind == 'f'


# ---------------------------------------------------------------------------
# hidePrints
# ---------------------------------------------------------------------------

class TestHidePrints:
    def test_suppresses_stdout(self, capsys):
        from citkid.primecam.instrument import hidePrints
        with hidePrints():
            print("this should not appear")
        captured = capsys.readouterr()
        assert captured.out == ''

    def test_restores_stdout_after(self, capsys):
        import sys
        from citkid.primecam.instrument import hidePrints
        with hidePrints():
            pass
        print("after")
        captured = capsys.readouterr()
        assert "after" in captured.out


# ---------------------------------------------------------------------------
# get_recent_file
# ---------------------------------------------------------------------------

class TestGetRecentFile:
    def test_returns_none_when_no_match(self, base_rfsoc):
        path, name = base_rfsoc.get_recent_file('nonexistent')
        assert path is None
        assert name is None

    def test_returns_most_recent_matching_file(self, base_rfsoc):
        tmp = base_rfsoc.tmp_directory
        older = os.path.join(tmp, 's21_vna_old.npy')
        newer = os.path.join(tmp, 's21_vna_new.npy')
        np.save(older, [1])
        time.sleep(0.05)  # ensure different mtime
        np.save(newer, [2])

        path, name = base_rfsoc.get_recent_file('s21_vna')
        assert name == 's21_vna_new.npy'

    def test_returns_path_and_basename(self, base_rfsoc):
        tmp = base_rfsoc.tmp_directory
        fpath = os.path.join(tmp, 'f_res_vna_test.npy')
        np.save(fpath, [1])
        path, name = base_rfsoc.get_recent_file('f_res_vna')
        assert path == fpath
        assert name == 'f_res_vna_test.npy'


# ---------------------------------------------------------------------------
# transfer_file
# ---------------------------------------------------------------------------

class TestTransferFile:
    def test_moves_file_to_out_directory(self, base_rfsoc):
        tmp = base_rfsoc.tmp_directory
        src = os.path.join(tmp, 's21_vna_data.npy')
        np.save(src, [1, 2, 3])

        base_rfsoc.transfer_file('s21_vna', 'result.npy')

        dest = os.path.join(base_rfsoc.out_directory, 'result.npy')
        assert os.path.isfile(dest)
        assert not os.path.isfile(src)


# ---------------------------------------------------------------------------
# clear_tmp_directory / clear_tmp_directory_full
# ---------------------------------------------------------------------------

class TestClearTmpDirectory:
    def test_removes_npy_files(self, base_rfsoc):
        tmp = base_rfsoc.tmp_directory
        np.save(os.path.join(tmp, 'a.npy'), [1])
        np.save(os.path.join(tmp, 'b.npy'), [2])
        base_rfsoc.clear_tmp_directory()
        remaining = [f for f in os.listdir(tmp) if f.endswith('.npy')]
        assert remaining == []

    def test_leaves_non_npy_files(self, base_rfsoc):
        tmp = base_rfsoc.tmp_directory
        txt_path = os.path.join(tmp, 'log.txt')
        with open(txt_path, 'w') as f:
            f.write('x')
        np.save(os.path.join(tmp, 'data.npy'), [1])
        base_rfsoc.clear_tmp_directory()
        assert os.path.isfile(txt_path)

    def test_clear_full_removes_all_files(self, base_rfsoc):
        tmp = base_rfsoc.tmp_directory
        np.save(os.path.join(tmp, 'data.npy'), [1])
        txt_path = os.path.join(tmp, 'log.txt')
        with open(txt_path, 'w') as f:
            f.write('x')
        base_rfsoc.clear_tmp_directory_full()
        assert os.listdir(tmp) == []


# ---------------------------------------------------------------------------
# make_custom_tone_lists (local logic only)
# ---------------------------------------------------------------------------

class TestMakeCustomToneLists:
    def test_auto_generates_ares_shape(self, base_rfsoc):
        """make_custom_tone_lists auto-generates ares with same length as fres."""
        from unittest.mock import patch
        fres = np.array([100e6, 200e6, 300e6])

        with patch.object(base_rfsoc, 'transfer_custom_tone_lists'):
            base_rfsoc.make_custom_tone_lists(fres)

        ares = np.load(os.path.join(base_rfsoc.tmp_directory, 'custom_amps.npy'))
        assert ares.shape == fres.shape

    def test_auto_generates_pres_shape(self, base_rfsoc):
        from unittest.mock import patch
        fres = np.array([100e6, 200e6, 300e6])
        # genPhis is a mock; configure it to return a real array
        base_rfsoc.genPhis = lambda f, a: np.zeros(len(f))

        with patch.object(base_rfsoc, 'transfer_custom_tone_lists'):
            base_rfsoc.make_custom_tone_lists(fres)

        pres = np.load(os.path.join(base_rfsoc.tmp_directory, 'custom_phis.npy'))
        assert pres.shape == fres.shape

    def test_saves_fres_to_tmp(self, base_rfsoc):
        from unittest.mock import patch
        fres = np.array([100e6, 200e6])

        with patch.object(base_rfsoc, 'transfer_custom_tone_lists'):
            base_rfsoc.make_custom_tone_lists(fres)

        saved = np.load(
            os.path.join(base_rfsoc.tmp_directory, 'custom_freqs.npy')
        )
        np.testing.assert_array_equal(saved, fres)

    def test_custom_ares_respected(self, base_rfsoc):
        from unittest.mock import patch
        fres = np.array([100e6, 200e6])
        ares = np.array([0.5, 0.3])

        with patch.object(base_rfsoc, 'transfer_custom_tone_lists'):
            base_rfsoc.make_custom_tone_lists(fres, ares=ares)

        saved = np.load(
            os.path.join(base_rfsoc.tmp_directory, 'custom_amps.npy')
        )
        np.testing.assert_array_almost_equal(saved, ares)

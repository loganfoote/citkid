"""
Tests for RFSOC input validation.

All methods that accept filenames require them to end in '.npy'.
transfer_file raises when no matching file exists in the tmp directory.
"""

import pytest
import os
import numpy as np


class TestFilenameValidation:
    """Methods that accept filename arguments must enforce the .npy extension."""

    @pytest.mark.parametrize('bad_name', ['out.csv', 'out', 'out.NPY'])
    def test_vna_sweep_bad_filename(self, base_rfsoc, bad_name):
        with pytest.raises(ValueError, match=r'\.npy'):
            base_rfsoc.vna_sweep(bad_name)

    @pytest.mark.parametrize('bad_name', ['out.csv', 'out'])
    def test_target_sweep_bad_filename(self, base_rfsoc, bad_name):
        with pytest.raises(ValueError, match=r'\.npy'):
            base_rfsoc.target_sweep(bad_name)

    @pytest.mark.parametrize('bad_name', ['out.csv', 'out', ''])
    def test_capture_save_noise_bad_filename(self, base_rfsoc, bad_name):
        with pytest.raises(ValueError, match=r'\.npy'):
            base_rfsoc.capture_save_noise(1.0, bad_name)

    @pytest.mark.parametrize('bad_name', ['out.csv', 'out'])
    def test_find_vna_res_bad_filename(self, base_rfsoc, bad_name):
        with pytest.raises(ValueError, match=r'\.npy'):
            base_rfsoc.find_vna_res(bad_name)

    @pytest.mark.parametrize('kwarg', ['f_filename', 'a_filename', 'p_filename'])
    def test_write_targ_comb_from_vna_bad_filename(self, base_rfsoc, kwarg):
        with pytest.raises(ValueError, match=r'\.npy'):
            base_rfsoc.write_targ_comb_from_vna(**{kwarg: 'bad.csv'})

    @pytest.mark.parametrize('kwarg', ['f_filename', 'a_filename', 'p_filename'])
    def test_write_targ_comb_from_targ_bad_filename(self, base_rfsoc, kwarg):
        with pytest.raises(ValueError, match=r'\.npy'):
            base_rfsoc.write_targ_comb_from_targ(**{kwarg: 'bad.csv'})

    @pytest.mark.parametrize('kwarg', ['f_filename', 'a_filename', 'p_filename'])
    def test_find_targ_res_bad_filename(self, base_rfsoc, kwarg):
        with pytest.raises(ValueError, match=r'\.npy'):
            base_rfsoc.find_targ_res(**{kwarg: 'bad.csv'})


class TestFalseFilenamesSkipValidation:
    """Passing False for optional filename args must not raise."""

    def test_vna_sweep_false_filename_no_raise(self, base_rfsoc):
        # Will fail later when alcoveCommand is called, but not on validation
        try:
            base_rfsoc.vna_sweep(False)
        except ValueError:
            pytest.fail("vna_sweep raised ValueError for False filename")
        except Exception:
            pass  # expected: alcoveCommand mock may raise or return

    def test_write_targ_comb_from_vna_all_false(self, base_rfsoc):
        try:
            base_rfsoc.write_targ_comb_from_vna(
                f_filename=False, a_filename=False, p_filename=False
            )
        except ValueError:
            pytest.fail("raised ValueError for all-False filenames")
        except Exception:
            pass


class TestTransferFileNotFound:
    def test_raises_when_file_missing(self, base_rfsoc):
        with pytest.raises(Exception, match='not found'):
            base_rfsoc.transfer_file('nonexistent_type', 'out.npy')

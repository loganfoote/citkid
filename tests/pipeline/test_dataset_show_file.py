import os
import tempfile
from unittest.mock import patch
import pytest

from citkid.pipeline import dataset
from citkid.pipeline import util


@pytest.fixture
def temp_dir_file():
    with tempfile.TemporaryDirectory() as td:
        # create a zarr-like directory
        zarr_path = os.path.join(td, 'test.zarr')
        os.makedirs(zarr_path, exist_ok=True)
        # create cal yaml
        cal_yaml = os.path.join(td, 'cal.yaml')
        with open(cal_yaml, 'w') as f:
            f.write('# minimal yaml')
        # create analysis yaml
        analysis_yaml = os.path.join(td, 'analysis.yaml')
        with open(analysis_yaml, 'w') as f:
            f.write('# analysis yaml')
        # create custom py
        custom_py = os.path.join(td, 'custom.py')
        with open(custom_py, 'w') as f:
            f.write('# custom steps placeholder')

        yield {
            'td': td,
            'zarr': zarr_path,
            'cal': cal_yaml,
            'analysis': analysis_yaml,
            'custom': custom_py,
        }


def make_ds_stub(paths):
    # Create a DataSet instance without running __init__ to avoid side-effects.
    ds = object.__new__(dataset.DataSet)
    ds.cal_yaml_path = paths['cal']
    ds.analysis_yaml_path = None
    ds.custom_path = None
    ds.zarr_path = paths['zarr']
    return ds


def test_show_file_cal_opens_dir(temp_dir_file):
    ds = make_ds_stub(temp_dir_file)
    with patch('citkid.pipeline.util.open_in_file_explorer') as mock_open:
        ds.show_file('cal')
        mock_open.assert_called_once_with(os.path.dirname(ds.cal_yaml_path))


def test_show_file_zarr_opens_dir(temp_dir_file):
    ds = make_ds_stub(temp_dir_file)
    with patch('citkid.pipeline.util.open_in_file_explorer') as mock_open:
        ds.show_file('zarr')
        mock_open.assert_called_once_with(os.path.dirname(ds.zarr_path))


def test_show_file_analysis_missing_raises(temp_dir_file):
    ds = make_ds_stub(temp_dir_file)
    with pytest.raises(ValueError, match="No analysis YAML file was provided"):
        ds.show_file('analysis')


def test_show_file_analysis_present_opens(temp_dir_file):
    ds = make_ds_stub(temp_dir_file)
    ds.analysis_yaml_path = temp_dir_file['analysis']
    with patch('citkid.pipeline.util.open_in_file_explorer') as mock_open:
        ds.show_file('analysis')
        mock_open.assert_called_once_with(os.path.dirname(ds.analysis_yaml_path))


def test_show_file_custom_missing_raises(temp_dir_file):
    ds = make_ds_stub(temp_dir_file)
    with pytest.raises(ValueError, match="No custom steps file was provided"):
        ds.show_file('custom')


def test_show_file_custom_present_opens(temp_dir_file):
    ds = make_ds_stub(temp_dir_file)
    ds.custom_path = temp_dir_file['custom']
    with patch('citkid.pipeline.util.open_in_file_explorer') as mock_open:
        ds.show_file('custom')
        mock_open.assert_called_once_with(os.path.dirname(ds.custom_path))


def test_show_file_unknown_no_action(temp_dir_file):
    ds = make_ds_stub(temp_dir_file)
    # Current implementation silently ignores unknown ftype; ensure no call.
    with patch('citkid.pipeline.util.open_in_file_explorer') as mock_open:
        ds.show_file('notatype')
        mock_open.assert_not_called()

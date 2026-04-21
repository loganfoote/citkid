import pytest
import os
import sys
from unittest.mock import patch, MagicMock, call
import tempfile
from citkid.pipeline import util


################################################################################
######################### open_in_file_explorer ################################
################################################################################

class TestOpenInFileExplorer:
    """Comprehensive tests for open_in_file_explorer function."""
    
    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield tmpdir
    
    @pytest.fixture
    def temp_file(self, temp_dir):
        """Create a temporary file for testing."""
        filepath = os.path.join(temp_dir, "test_file.txt")
        with open(filepath, 'w') as f:
            f.write("test content")
        return filepath
    
@pytest.fixture(autouse=True)
def _always_exists(monkeypatch):
    """Make os.path.exists return True by default for these tests so
    we don't attempt to access the real filesystem for mocked absolute
    paths used in some test cases."""
    monkeypatch.setattr(os.path, 'exists', lambda p: True)
    yield
    
    # Windows platform tests
    @patch('sys.platform', 'win32')
    @patch('os.startfile')
    def test_windows_opens_with_startfile(self, mock_startfile, temp_dir):
        """Test that Windows uses os.startfile."""
        util.open_in_file_explorer(temp_dir)
        mock_startfile.assert_called_once_with(os.path.abspath(temp_dir))
    
    @patch('sys.platform', 'win32')
    @patch('os.startfile')
    def test_windows_with_relative_path(self, mock_startfile, temp_dir):
        """Test Windows with relative path converts to absolute."""
        with patch('os.path.abspath', return_value='/absolute/path') as mock_abspath:
            util.open_in_file_explorer('relative/path')
            mock_abspath.assert_called_once_with('relative/path')
            mock_startfile.assert_called_once_with('/absolute/path')
    
    @patch('sys.platform', 'win32')
    @patch('os.startfile')
    def test_windows_with_file(self, mock_startfile, temp_file):
        """Test Windows can open a file path."""
        util.open_in_file_explorer(temp_file)
        mock_startfile.assert_called_once_with(os.path.abspath(temp_file))
    
    # Linux platform tests
    @patch('sys.platform', 'linux')
    @patch('subprocess.run')
    def test_linux_opens_with_xdg_open(self, mock_run, temp_dir):
        """Test that Linux uses xdg-open."""
        util.open_in_file_explorer(temp_dir)
        mock_run.assert_called_once_with(["xdg-open", os.path.abspath(temp_dir)])
    
    @patch('sys.platform', 'linux2')
    @patch('subprocess.run')
    def test_linux2_platform_variant(self, mock_run, temp_dir):
        """Test Linux platform variant (linux2)."""
        util.open_in_file_explorer(temp_dir)
        mock_run.assert_called_once_with(["xdg-open", os.path.abspath(temp_dir)])
    
    @patch('sys.platform', 'linux')
    @patch('subprocess.run')
    def test_linux_with_relative_path(self, mock_run, temp_dir):
        """Test Linux with relative path converts to absolute."""
        with patch('os.path.abspath', return_value='/absolute/path') as mock_abspath:
            util.open_in_file_explorer('relative/path')
            mock_abspath.assert_called_once_with('relative/path')
            mock_run.assert_called_once_with(["xdg-open", '/absolute/path'])
    
    @patch('sys.platform', 'linux')
    @patch('subprocess.run')
    def test_linux_with_file(self, mock_run, temp_file):
        """Test Linux can open a file path."""
        util.open_in_file_explorer(temp_file)
        mock_run.assert_called_once_with(["xdg-open", os.path.abspath(temp_file)])
    
    # macOS platform tests
    @patch('sys.platform', 'darwin')
    @patch('subprocess.run')
    def test_macos_opens_with_open_command(self, mock_run, temp_dir):
        """Test that macOS uses 'open' command."""
        util.open_in_file_explorer(temp_dir)
        mock_run.assert_called_once_with(["open", os.path.abspath(temp_dir)])
    
    @patch('sys.platform', 'darwin')
    @patch('subprocess.run')
    def test_macos_with_relative_path(self, mock_run, temp_dir):
        """Test macOS with relative path converts to absolute."""
        with patch('os.path.abspath', return_value='/absolute/path') as mock_abspath:
            util.open_in_file_explorer('relative/path')
            mock_abspath.assert_called_once_with('relative/path')
            mock_run.assert_called_once_with(["open", '/absolute/path'])
    
    @patch('sys.platform', 'darwin')
    @patch('subprocess.run')
    def test_macos_with_file(self, mock_run, temp_file):
        """Test macOS can open a file path."""
        util.open_in_file_explorer(temp_file)
        mock_run.assert_called_once_with(["open", os.path.abspath(temp_file)])
    
    # Unsupported OS tests
    @patch('sys.platform', 'freebsd')
    def test_unsupported_os_raises_error(self, temp_dir):
        """Test that unsupported OS raises RuntimeError."""
        with pytest.raises(RuntimeError, match="Unsupported OS"):
            util.open_in_file_explorer(temp_dir)
    
    @patch('sys.platform', 'aix')
    def test_aix_platform_raises_error(self, temp_dir):
        """Test that AIX platform raises RuntimeError."""
        with pytest.raises(RuntimeError, match="Unsupported OS"):
            util.open_in_file_explorer(temp_dir)
    
    @patch('sys.platform', 'sunos')
    def test_sunos_platform_raises_error(self, temp_dir):
        """Test that SunOS platform raises RuntimeError."""
        with pytest.raises(RuntimeError, match="Unsupported OS"):
            util.open_in_file_explorer(temp_dir)
    
    # Path handling tests
    @patch('sys.platform', 'win32')
    @patch('os.startfile')
    def test_path_with_spaces(self, mock_startfile, temp_dir):
        """Test handling paths with spaces."""
        path_with_spaces = os.path.join(temp_dir, "folder with spaces")
        os.makedirs(path_with_spaces, exist_ok=True)
        util.open_in_file_explorer(path_with_spaces)
        mock_startfile.assert_called_once_with(os.path.abspath(path_with_spaces))
    
    @patch('sys.platform', 'linux')
    @patch('subprocess.run')
    def test_path_with_special_characters(self, mock_run, temp_dir):
        """Test handling paths with special characters."""
        special_path = os.path.join(temp_dir, "folder-with_special.chars")
        os.makedirs(special_path, exist_ok=True)
        util.open_in_file_explorer(special_path)
        mock_run.assert_called_once_with(["xdg-open", os.path.abspath(special_path)])
    
    @patch('sys.platform', 'darwin')
    @patch('subprocess.run')
    def test_unicode_path(self, mock_run, temp_dir):
        """Test handling Unicode characters in path."""
        # Create a path with Unicode characters
        unicode_path = os.path.join(temp_dir, "folder_测试_🔬")
        try:
            os.makedirs(unicode_path, exist_ok=True)
            util.open_in_file_explorer(unicode_path)
            mock_run.assert_called_once_with(["open", os.path.abspath(unicode_path)])
        except (OSError, UnicodeEncodeError):
            # Skip if filesystem doesn't support these characters
            pytest.skip("Filesystem doesn't support Unicode characters")
    
    # Path normalization tests
    @patch('sys.platform', 'win32')
    @patch('os.startfile')
    def test_dot_path_normalized(self, mock_startfile):
        """Test that '.' is converted to absolute path."""
        util.open_in_file_explorer('.')
        mock_startfile.assert_called_once_with(os.path.abspath('.'))
    
    @patch('sys.platform', 'linux')
    @patch('subprocess.run')
    def test_parent_directory_path(self, mock_run):
        """Test that '..' is converted to absolute path."""
        util.open_in_file_explorer('..')
        mock_run.assert_called_once_with(["xdg-open", os.path.abspath('..')])
    
    @patch('sys.platform', 'darwin')
    @patch('subprocess.run')
    def test_complex_relative_path(self, mock_run):
        """Test complex relative path like '../folder/subfolder'."""
        relative_path = '../folder/subfolder'
        util.open_in_file_explorer(relative_path)
        mock_run.assert_called_once_with(["open", os.path.abspath(relative_path)])
    
    # Integration-style tests (verify path conversion)
    @patch('sys.platform', 'win32')
    @patch('os.startfile')
    def test_ensures_absolute_path_called(self, mock_startfile, temp_dir):
        """Verify that os.path.abspath is actually called."""
        with patch('os.path.abspath', wraps=os.path.abspath) as mock_abspath:
            util.open_in_file_explorer(temp_dir)
            mock_abspath.assert_called_once_with(temp_dir)
            mock_startfile.assert_called_once()
    
    # Error propagation tests
    @patch('sys.platform', 'win32')
    @patch('os.startfile', side_effect=OSError("Access denied"))
    def test_windows_propagates_os_error(self, mock_startfile, temp_dir):
        """Test that OSError from os.startfile is propagated."""
        with pytest.raises(OSError, match="Access denied"):
            util.open_in_file_explorer(temp_dir)
    
    @patch('sys.platform', 'linux')
    @patch('subprocess.run', side_effect=FileNotFoundError("xdg-open not found"))
    def test_linux_propagates_file_not_found(self, mock_run, temp_dir):
        """Test that FileNotFoundError from subprocess is propagated."""
        with pytest.raises(FileNotFoundError, match="xdg-open not found"):
            util.open_in_file_explorer(temp_dir)
    
    @patch('sys.platform', 'darwin')
    @patch('subprocess.run', side_effect=PermissionError("Permission denied"))
    def test_macos_propagates_permission_error(self, mock_run, temp_dir):
        """Test that PermissionError from subprocess is propagated."""
        with pytest.raises(PermissionError, match="Permission denied"):
            util.open_in_file_explorer(temp_dir)


################################################################################
############################ group_unique_tuples ###############################
################################################################################

def test_group_unique_tuples_empty_list():
    """Test with empty input list."""
    unique, indices = util.group_unique_tuples([])
    assert unique == []
    assert indices == []


def test_group_unique_tuples_single_element():
    """Test with single tuple."""
    tuples = [(1, {'a': 1})]
    unique, indices = util.group_unique_tuples(tuples)
    assert unique == [(1, {'a': 1})]
    assert indices == [[0]]


def test_group_unique_tuples_all_identical():
    """Test with all identical tuples."""
    tuples = [(1, {'a': 1}), (1, {'a': 1}), (1, {'a': 1})]
    unique, indices = util.group_unique_tuples(tuples)
    assert len(unique) == 1
    assert unique[0] == (1, {'a': 1})
    assert indices == [[0, 1, 2]]


def test_group_unique_tuples_all_unique():
    """Test with all unique tuples."""
    tuples = [(1, {'a': 1}), (2, {'b': 2}), (3, {'c': 3})]
    unique, indices = util.group_unique_tuples(tuples)
    assert len(unique) == 3
    assert len(indices) == 3
    # Each should appear once
    for idx_list in indices:
        assert len(idx_list) == 1


def test_group_unique_tuples_basic_grouping():
    """Test basic grouping as shown in docstring."""
    tuples = [(1, {'a': 1}), (2, {'b': 2}), (1, {'a': 1})]
    unique, indices = util.group_unique_tuples(tuples)
    
    assert len(unique) == 2
    assert len(indices) == 2
    
    # Find which index corresponds to each unique tuple
    for i, tup in enumerate(unique):
        if tup == (1, {'a': 1}):
            assert set(indices[i]) == {0, 2}
        elif tup == (2, {'b': 2}):
            assert indices[i] == [1]


def test_group_unique_tuples_different_int_keys():
    """Test with different integer keys but same dict."""
    tuples = [(1, {'x': 10}), (2, {'x': 10}), (1, {'x': 10})]
    unique, indices = util.group_unique_tuples(tuples)
    
    assert len(unique) == 2
    # (1, {'x': 10}) should have indices 0 and 2
    # (2, {'x': 10}) should have index 1


def test_group_unique_tuples_same_key_different_dicts():
    """Test with same key but different dict values."""
    tuples = [(1, {'a': 1}), (1, {'a': 2}), (1, {'a': 1})]
    unique, indices = util.group_unique_tuples(tuples)
    
    assert len(unique) == 2
    # Should group by dict content, not just key


def test_group_unique_tuples_tuple_as_key():
    """Test with tuples as the first element."""
    tuples = [((1, 2), {'a': 1}), ((3, 4), {'b': 2}), ((1, 2), {'a': 1})]
    unique, indices = util.group_unique_tuples(tuples)
    
    assert len(unique) == 2
    for i, tup in enumerate(unique):
        if tup == ((1, 2), {'a': 1}):
            assert set(indices[i]) == {0, 2}
        elif tup == ((3, 4), {'b': 2}):
            assert indices[i] == [1]


def test_group_unique_tuples_string_as_key():
    """Test with strings as the first element."""
    tuples = [('key1', {'a': 1}), ('key2', {'b': 2}), ('key1', {'a': 1})]
    unique, indices = util.group_unique_tuples(tuples)
    
    assert len(unique) == 2


def test_group_unique_tuples_empty_dicts():
    """Test with empty dictionaries."""
    tuples = [(1, {}), (2, {}), (1, {})]
    unique, indices = util.group_unique_tuples(tuples)
    
    assert len(unique) == 2
    # (1, {}) and (2, {}) are different due to different keys


def test_group_unique_tuples_complex_dicts():
    """Test with complex dictionary values."""
    tuples = [
        (1, {'a': 1, 'b': 2, 'c': 3}),
        (2, {'x': 10}),
        (1, {'a': 1, 'b': 2, 'c': 3}),
        (3, {'a': 1, 'b': 2, 'c': 3})
    ]
    unique, indices = util.group_unique_tuples(tuples)
    
    assert len(unique) == 3
    # (1, {'a': 1, 'b': 2, 'c': 3}) appears at indices 0 and 2


def test_group_unique_tuples_dict_order_irrelevant():
    """Test that dict key order doesn't affect grouping."""
    # Python dicts maintain insertion order, but comparison should be 
    # content-based
    tuples = [
        (1, {'a': 1, 'b': 2}),
        (1, {'b': 2, 'a': 1}),  # Same content, different order
    ]
    unique, indices = util.group_unique_tuples(tuples)
    
    # Should be grouped together since dict content is the same
    assert len(unique) == 1
    assert indices == [[0, 1]]


def test_group_unique_tuples_nested_values():
    """Test with nested hashable structures in dict values."""
    tuples = [
        (1, {'a': (1, 2, 3)}),
        (2, {'a': (4, 5, 6)}),
        (1, {'a': (1, 2, 3)})
    ]
    unique, indices = util.group_unique_tuples(tuples)
    
    assert len(unique) == 2
    # Tuple (1, {'a': (1, 2, 3)}) should appear at indices 0 and 2


def test_group_unique_tuples_unhashable_values_raises():
    """Test that unhashable values in dict raise TypeError."""
    tuples = [
        (1, {'a': [1, 2, 3]}),  # List is unhashable
    ]
    with pytest.raises(TypeError, match="unhashable type"):
        util.group_unique_tuples(tuples)


def test_group_unique_tuples_multiple_groups():
    """Test with multiple groups of duplicates."""
    tuples = [
        (1, {'a': 1}),  # 0
        (2, {'b': 2}),  # 1
        (1, {'a': 1}),  # 2
        (3, {'c': 3}),  # 3
        (2, {'b': 2}),  # 4
        (1, {'a': 1}),  # 5
    ]
    unique, indices = util.group_unique_tuples(tuples)
    
    assert len(unique) == 3
    # Verify all original indices are accounted for
    all_indices = set()
    for idx_list in indices:
        all_indices.update(idx_list)
    assert all_indices == {0, 1, 2, 3, 4, 5}


def test_group_unique_tuples_preserves_first_occurrence():
    """Test that unique list uses first occurrence of each unique tuple."""
    tuples = [
        (1, {'a': 1}),  # First occurrence of this combo
        (2, {'b': 2}),
        (1, {'a': 1}),  # Duplicate
    ]
    unique, indices = util.group_unique_tuples(tuples)
    
    # The unique tuple should be from index 0
    for i, tup in enumerate(unique):
        if tup == (1, {'a': 1}):
            # First index in the group should be 0
            assert 0 in indices[i]


def test_group_unique_tuples_indices_maintain_order():
    """Test that indices within each group maintain original order."""
    tuples = [
        (1, {'a': 1}),  # 0
        (2, {'b': 2}),  # 1
        (1, {'a': 1}),  # 2
        (2, {'b': 2}),  # 3
        (1, {'a': 1}),  # 4
    ]
    unique, indices = util.group_unique_tuples(tuples)
    
    for idx_list in indices:
        # Check that indices are in ascending order
        assert idx_list == sorted(idx_list)


def test_group_unique_tuples_large_groups():
    """Test with many duplicates in a group."""
    base_tuple = (1, {'x': 100})
    tuples = [base_tuple] * 50 + [(2, {'y': 200})]
    
    unique, indices = util.group_unique_tuples(tuples)
    
    assert len(unique) == 2
    # One group should have 50 elements
    assert any(len(idx_list) == 50 for idx_list in indices)
    assert any(len(idx_list) == 1 for idx_list in indices)


def test_group_unique_tuples_numeric_dict_values():
    """Test with various numeric types in dict values."""
    tuples = [
        (1, {'val': 1}),
        (1, {'val': 1.0}),  # Different type but equal value
        (1, {'val': 1})
    ]
    unique, indices = util.group_unique_tuples(tuples)
    
    # 1 and 1.0 compare equal, so might be grouped together
    # This tests the actual behavior of frozenset(dict.items())
    assert len(unique) <= 2


def test_group_unique_tuples_none_values():
    """Test with None in dict values."""
    tuples = [
        (1, {'a': None}),
        (2, {'a': None}),
        (1, {'a': None})
    ]
    unique, indices = util.group_unique_tuples(tuples)
    
    assert len(unique) == 2

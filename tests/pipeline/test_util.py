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

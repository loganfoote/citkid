"""
Shared fixtures for primecam instrument tests.

External primecam_readout imports (queen, alcove_commands, alcove) are injected
into sys.modules so RFSOC.__init__ can complete without the hardware package.
"""

import sys
import pytest
from unittest.mock import MagicMock, patch


@pytest.fixture
def mock_primecam_imports():
    """
    Inject mock stand-ins for the external primecam_readout modules that
    RFSOC.__init__ dynamically imports:
        from queen import alcoveCommand
        from alcove_commands.tones import genPhis
        from alcove import comNumFromStr
    """
    mock_queen = MagicMock()
    mock_alcove_commands = MagicMock()
    mock_alcove_commands_tones = MagicMock()
    mock_alcove = MagicMock()

    modules = {
        'queen': mock_queen,
        'alcove_commands': mock_alcove_commands,
        'alcove_commands.tones': mock_alcove_commands_tones,
        'alcove': mock_alcove,
    }

    with patch.dict(sys.modules, modules):
        yield {
            'queen': mock_queen,
            'alcove_commands': mock_alcove_commands,
            'alcove_commands.tones': mock_alcove_commands_tones,
            'alcove': mock_alcove,
        }


@pytest.fixture
def base_rfsoc(mock_primecam_imports, tmp_path):
    """
    Create a basic RFSOC instance with all external dependencies mocked.

    Uses tmp_path so no real directories are created in the repo.
    noiseq=False to skip UDP socket binding.
    """
    from citkid.primecam.instrument import RFSOC

    with patch('os.getcwd', return_value=str(tmp_path)):
        rfsoc = RFSOC(
            out_directory=str(tmp_path / 'out'),
            bid=1,
            drid=2,
            noiseq=False,
        )
    return rfsoc


@pytest.fixture
def base_rfsoc_with_socket(mock_primecam_imports, tmp_path):
    """RFSOC instance with a mocked UDP socket (noiseq=True)."""
    from citkid.primecam.instrument import RFSOC

    mock_sock = MagicMock()
    with patch('os.getcwd', return_value=str(tmp_path)), \
         patch('socket.socket', return_value=mock_sock):
        rfsoc = RFSOC(
            out_directory=str(tmp_path / 'out'),
            bid=1,
            drid=1,
            udp_ip='192.168.3.40',
            noiseq=True,
        )
    rfsoc._mock_sock = mock_sock
    return rfsoc

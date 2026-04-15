"""
Tests for RFSOC.__init__ method.

Covers attribute storage, directory creation, and socket setup.
"""

import os
import pytest
from unittest.mock import MagicMock, patch


class TestRFSOCInitAttributes:
    def test_stores_bid(self, base_rfsoc):
        assert base_rfsoc.bid == 1

    def test_stores_drid(self, base_rfsoc):
        assert base_rfsoc.drid == 2

    def test_stores_out_directory(self, base_rfsoc, tmp_path):
        assert base_rfsoc.out_directory == os.path.normpath(
            str(tmp_path / 'out')
        )

    def test_stores_sample_time(self, base_rfsoc):
        assert base_rfsoc.sample_time == pytest.approx(5 / 2441)

    def test_stores_alcove_command(self, base_rfsoc):
        assert hasattr(base_rfsoc, 'alcoveCommand')
        assert callable(base_rfsoc.alcoveCommand)

    def test_stores_com_num_from_str(self, base_rfsoc):
        assert hasattr(base_rfsoc, 'comNumFromStr')
        assert callable(base_rfsoc.comNumFromStr)

    def test_stores_gen_phis(self, base_rfsoc):
        assert hasattr(base_rfsoc, 'genPhis')
        assert callable(base_rfsoc.genPhis)


class TestRFSOCInitDirectories:
    def test_creates_out_directory(self, base_rfsoc):
        assert os.path.isdir(base_rfsoc.out_directory)

    def test_creates_tmp_directory(self, base_rfsoc):
        assert os.path.isdir(base_rfsoc.tmp_directory)

    def test_tmp_directory_is_under_cwd(self, mock_primecam_imports, tmp_path):
        from citkid.primecam.instrument import RFSOC
        with patch('os.getcwd', return_value=str(tmp_path)):
            rfsoc = RFSOC(
                out_directory=str(tmp_path / 'out'),
                noiseq=False,
            )
        assert rfsoc.tmp_directory == os.path.join(
            os.path.normpath(str(tmp_path)), 'tmp'
        )


class TestRFSOCInitSocket:
    def test_no_sock_attribute_when_noiseq_false(self, base_rfsoc):
        assert not hasattr(base_rfsoc, 'sock')

    def test_sock_attribute_exists_when_noiseq_true(
        self, base_rfsoc_with_socket
    ):
        assert hasattr(base_rfsoc_with_socket, 'sock')

    def test_socket_bound_to_udp_ip(
        self, mock_primecam_imports, tmp_path
    ):
        from citkid.primecam.instrument import RFSOC
        mock_sock = MagicMock()
        with patch('os.getcwd', return_value=str(tmp_path)), \
             patch('socket.socket', return_value=mock_sock):
            rfsoc = RFSOC(
                out_directory=str(tmp_path / 'out'),
                udp_ip='10.0.0.1',
                noiseq=True,
            )
        mock_sock.bind.assert_called_once_with(('10.0.0.1', 4096))

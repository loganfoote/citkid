"""
Tests for streaming into memory (``CRS.stream(method = 'memory')``).

Covers ``util.capture_to_memory``, ``util.memory_to_zarr``, the memory check,
and the memory path of ``CRS.stream`` and ``CRS.capture_ts``.
"""

import os
import pytest
import numpy as np
import zarr
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, Mock, patch

from citkid.crs import util
from citkid.crs.instrument import _validate_stream_input


################################################################################
################################ fake receiver #################################
################################################################################

class FakePacket:
    """Packet as returned by ``PacketQueue.try_pop``."""

    def __init__(self, seq, raw_samples):
        """
        Store the packet contents.

        Parameters:
        seq (int): sequence number.
        raw_samples (np.array int32): interleaved I/Q samples.

        Returns:
        None
        """
        self.seq = seq
        self.raw_samples = raw_samples

    def to_python(self):
        """
        Return the packet object, like ``Packet.to_python``.

        Returns:
        FakePacket: self.
        """
        return self


class FakeQueue:
    """Packet queue with a non-blocking pop."""

    def __init__(self):
        """
        Create an empty queue.

        Returns:
        None
        """
        self.packets = []

    def try_pop(self):
        """
        Pop the oldest packet.

        Returns:
        FakePacket or None: the packet, or None if the queue is empty.
        """
        return self.packets.pop(0) if self.packets else None


def make_receiver_class(batches):
    """
    Create a fake ReadoutPacketReceiver class that delivers scripted batches.

    Parameters:
    batches (list): each element is a list of (serial, packet_module, packet)
        tuples delivered by one receive_batch call. Later calls deliver
        nothing.

    Returns:
    type: fake receiver class.
    """
    class FakeReceiver:
        def __init__(self, sock, reorder_window = 256):
            self.queues = {}
            self.batches = list(batches)

        def receive_batch(self, batch_size = 256, timeout_ms = None):
            if not self.batches:
                return 0
            batch = self.batches.pop(0)
            for serial, module, packet in batch:
                self.queues.setdefault((serial, module), FakeQueue())
                self.queues[(serial, module)].packets.append(packet)
            return len(batch)

        def get_all_queues(self):
            return [(s, m, q) for (s, m), q in self.queues.items()]
    return FakeReceiver


@contextmanager
def fake_socket(**kwargs):
    """
    Stand-in for ``streamer.get_multicast_socket``.

    Parameters:
    kwargs: ignored.

    Returns:
    generator: yields a Mock socket.
    """
    yield Mock()


@contextmanager
def patched_streamer(batches):
    """
    Patch the rfmux streamer and interface lookup used by capture_to_memory.

    Parameters:
    batches (list): see make_receiver_class.

    Returns:
    generator: yields the mock for get_multicast_socket.
    """
    with patch('rfmux.streamer.get_multicast_socket',
               side_effect = fake_socket) as mock_sock, \
         patch('rfmux.streamer.ReadoutPacketReceiver',
               make_receiver_class(batches)), \
         patch('rfmux.tools.parser.resolve_interface',
               return_value = '10.0.0.1'):
        yield mock_sock


def make_frames(nframes, nchannels, seed):
    """
    Create random int32 I/Q frames.

    Parameters:
    nframes (int): number of frames.
    nchannels (int): channels per frame.
    seed (int): random seed.

    Returns:
    np.array int32: frames of shape (nframes, nchannels, 2).
    """
    rng = np.random.default_rng(seed)
    return rng.integers(-2 ** 31, 2 ** 31 - 1, size = (nframes, nchannels, 2),
                        dtype = np.int32)


def packets_from_frames(frames, serial, packet_module, seq0 = 0):
    """
    Convert frames into scripted packets.

    Parameters:
    frames (np.array int32): frames of shape (nframes, nchannels, 2).
    serial (int): board serial number.
    packet_module (int): module index in the packet (0-3).
    seq0 (int): first sequence number.

    Returns:
    list: (serial, packet_module, FakePacket) tuples.
    """
    return [(serial, packet_module, FakePacket(seq0 + i, f.reshape(-1).copy()))
            for i, f in enumerate(frames)]


################################################################################
############################## capture_to_memory ###############################
################################################################################

def test_capture_to_memory_fills_buffers():
    """Test frames are stored per module and other packets are ignored."""
    f1 = make_frames(6, 8, 0)
    f2 = make_frames(6, 8, 1)
    other_board = packets_from_frames(make_frames(6, 8, 2), 99, 0)
    other_module = packets_from_frames(make_frames(6, 8, 3), 27, 3)
    p1 = packets_from_frames(f1, 27, 0)
    p2 = packets_from_frames(f2, 27, 1)
    # Interleave delivery across several batches
    batches = [other_board[:3] + p1[:2] + other_module + p2[:4],
               p1[2:] + other_board[3:], p2[4:]]

    with patched_streamer(batches) as mock_sock:
        buffers, dropped = util.capture_to_memory(
            'eth0', 27, [1, 2], max_ntones = 5, nframes = 4)

    assert sorted(buffers) == [1, 2]
    np.testing.assert_array_equal(buffers[1], f1[:4, :5])
    np.testing.assert_array_equal(buffers[2], f2[:4, :5])
    assert buffers[1].dtype == np.int32
    assert dropped == {1: 0, 2: 0}
    kwargs = mock_sock.call_args.kwargs
    assert kwargs['interface'] == '10.0.0.1'
    assert kwargs['crs_hostname'] is None


def test_capture_to_memory_high_bank_module_numbering():
    """Test packets from module 0 map to module 5 in the high bank."""
    frames = make_frames(3, 4, 4)
    batches = [packets_from_frames(frames, 27, 0)]

    with patched_streamer(batches):
        buffers, _ = util.capture_to_memory('eth0', 27, [5], 4, 3)

    np.testing.assert_array_equal(buffers[5], frames)


def test_capture_to_memory_counts_dropped_packets():
    """Test gaps in the sequence numbers are counted as dropped."""
    frames = make_frames(5, 2, 5)
    packets = packets_from_frames(frames, 27, 0)
    # Drop the packets with seq 1 and 2
    for i, (_, _, pkt) in enumerate(packets):
        pkt.seq = [0, 3, 4, 5, 6][i]
    with patched_streamer([packets]):
        buffers, dropped = util.capture_to_memory('eth0', 27, [1], 2, 5)

    assert dropped == {1: 2}
    np.testing.assert_array_equal(buffers[1], frames)


def test_capture_to_memory_sequence_wraparound():
    """Test sequence numbers that wrap around 2**32 are not drops."""
    frames = make_frames(3, 2, 6)
    packets = packets_from_frames(frames, 27, 0, seq0 = 2 ** 32 - 2)
    packets[2][2].seq = 0
    with patched_streamer([packets]):
        _, dropped = util.capture_to_memory('eth0', 27, [1], 2, 3)
    assert dropped == {1: 0}


def test_capture_to_memory_short_packets_raises():
    """Test an error if packets have fewer channels than max_ntones."""
    batches = [packets_from_frames(make_frames(2, 128, 7), 27, 0)]
    with patched_streamer(batches):
        with pytest.raises(RuntimeError, match = 'short packets'):
            util.capture_to_memory('eth0', 27, [1], 200, 2)


def test_capture_to_memory_stalled_module_raises():
    """Test an error if a module stops sending packets."""
    batches = [packets_from_frames(make_frames(2, 4, 8), 27, 0)]
    clock = Mock()
    clock.monotonic.side_effect = [float(t) for t in range(0, 1000, 4)]
    with patched_streamer(batches), \
         patch('citkid.crs.util.time', clock):
        with pytest.raises(RuntimeError, match = r'modules \[2\]'):
            util.capture_to_memory('eth0', 27, [1, 2], 4, 2,
                                   timeout_s = 10.0)


@pytest.mark.parametrize("kwargs", [
    dict(module_idxs = []),
    dict(module_idxs = [0]),
    dict(module_idxs = [1, 5]),
    dict(max_ntones = 0),
    dict(nframes = 0),
])
def test_capture_to_memory_invalid_inputs(kwargs):
    """Test invalid inputs raise ValueError."""
    args = dict(interface = 'eth0', serial_number = 27, module_idxs = [1],
                max_ntones = 4, nframes = 2)
    args.update(kwargs)
    with pytest.raises(ValueError):
        util.capture_to_memory(**args)


################################################################################
######################## memory_to_zarr vs parser_to_zarr ######################
################################################################################

def write_parser_files(path, serial_number, buffers):
    """
    Write buffers as rfmux parser raw32 files.

    Parameters:
    path (str): parser data directory.
    serial_number (int): board serial number.
    buffers (dict): module index -> int32 array (nframes, max_ntones, 2).

    Returns:
    None
    """
    board = os.path.join(path, f'serial_{serial_number:04d}')
    os.makedirs(board)
    for mi, buf in buffers.items():
        fname = 'm0%d_raw32' % (mi - (mi // 5) * 4)
        buf.tofile(os.path.join(board, fname))


@pytest.mark.parametrize("module_idxs, max_ntones, nframes, chunk_size_mb", [
    ([1], 4, 50, 128),
    ([1, 3], 6, 101, 0.001),
    ([5, 6], 3, 37, 0.0005),
])
def test_memory_to_zarr_matches_parser_to_zarr(
    tmp_path, module_idxs, max_ntones, nframes, chunk_size_mb
):
    """Test both methods save identical data and array layouts."""
    rng = np.random.default_rng(0)
    buffers = {mi: make_frames(nframes, max_ntones, mi)
               for mi in module_idxs}
    # Channels: module tones interleaved in the global ordering, with one
    # missing channel at the end
    ntones = sum(max_ntones - 1 for _ in module_idxs) + 1
    order = rng.permutation(ntones - 1)
    ch_map, ares_map, start = {}, {}, 0
    for mi in module_idxs:
        ch_map[mi] = order[start:start + max_ntones - 1].astype(np.int32)
        ares_map[mi] = rng.uniform(-60, -40, max_ntones - 1)
        start += max_ntones - 1
    dt = 1 / 596.

    parser_dir = str(tmp_path / 'parser')
    write_parser_files(parser_dir, 27, buffers)
    root = zarr.open_group(str(tmp_path / 'out.zarr'), mode = 'w')
    grp_parser = root.create_group('parser')
    grp_memory = root.create_group('memory')

    util.parser_to_zarr(parser_dir, grp_parser, 27, ntones, max_ntones,
                        ch_map, ares_map, dt,
                        batch_size_mb = max(0.01, chunk_size_mb),
                        chunk_size_mb = chunk_size_mb)
    util.memory_to_zarr(buffers, grp_memory, ntones, max_ntones, ch_map,
                        ares_map, dt, chunk_size_mb = chunk_size_mb)

    assert sorted(grp_parser.keys()) == sorted(grp_memory.keys())
    for name in grp_parser.keys():
        a, b = grp_parser[name], grp_memory[name]
        assert a.shape == b.shape
        assert a.dtype == b.dtype
        assert a.chunks == b.chunks
        assert a.shards == b.shards
        np.testing.assert_array_equal(a[...], b[...])

    # Spot check the layout against the input
    z = grp_memory['z'][...]
    mi = module_idxs[0]
    np.testing.assert_array_equal(z[0, ch_map[mi][0]], buffers[mi][:, 0, 0])
    np.testing.assert_array_equal(z[1, ch_map[mi][-1]],
                                  buffers[mi][:, max_ntones - 2, 1])
    assert np.all(z[:, ntones - 1] == 0)
    assert np.isnan(grp_memory['counts_to_s21'][ntones - 1])


def test_memory_to_zarr_uses_shortest_module(tmp_path):
    """Test the saved length is the shortest module's length."""
    buffers = {1: make_frames(10, 2, 0), 2: make_frames(7, 2, 1)}
    grp = zarr.open_group(str(tmp_path / 'out.zarr'), mode = 'w')
    util.memory_to_zarr(buffers, grp, 2, 2, {1: [0], 2: [1]},
                        {1: [-50.], 2: [-50.]}, 0.01)
    assert grp['z'].shape == (2, 2, 7)


@pytest.mark.parametrize("buffers, match", [
    ({}, 'does not contain module 1'),
    ({1: np.zeros((5, 3, 2), dtype = np.int32)}, 'must have shape'),
])
def test_memory_to_zarr_invalid_buffers(tmp_path, buffers, match):
    """Test buffers are checked against ch_map and max_ntones."""
    grp = zarr.open_group(str(tmp_path / 'out.zarr'), mode = 'w')
    with pytest.raises(ValueError, match = match):
        util.memory_to_zarr(buffers, grp, 1, 2, {1: [0]}, {1: [-50.]}, 0.01)


def test_memory_to_zarr_grp_conflict(tmp_path):
    """Test existing output names raise an error."""
    grp = zarr.open_group(str(tmp_path / 'out.zarr'), mode = 'w')
    grp.create_array(name = 'z', data = np.zeros(1))
    with pytest.raises(ValueError, match = 'already contains'):
        util.memory_to_zarr({1: np.zeros((2, 1, 2), dtype = np.int32)}, grp,
                            1, 1, {1: [0]}, {1: [-50.]}, 0.01)


################################################################################
################################# memory check #################################
################################################################################

def test_estimate_memory_stream_bytes():
    """Test the estimate counts buffers plus two write blocks."""
    # 2 modules x 1000 frames x 100 channels x 8 bytes = 1.6 MB buffers
    est = util.estimate_memory_stream_bytes(2, 100, 150, 1000, 128)
    data = 150 * 1000 * 8
    assert est == 2 * 1000 * 100 * 8 + 2 * data
    # Blocks are capped at the chunk size
    est = util.estimate_memory_stream_bytes(2, 100, 150, 1000, 0.1)
    assert est == 2 * 1000 * 100 * 8 + 2 * int(0.1 * 1024 ** 2)


def test_check_memory_for_stream_passes():
    """Test the check passes with enough memory."""
    mem = Mock(available = 10e9)
    with patch('citkid.crs.util.psutil.virtual_memory', return_value = mem):
        required = util.check_memory_for_stream(4, 1024, 4096, 10000, 128)
    assert required == util.estimate_memory_stream_bytes(4, 1024, 4096,
                                                         10000, 128)


def test_check_memory_for_stream_requires_headroom():
    """Test the check fails above the allowed fraction of available memory."""
    required = util.estimate_memory_stream_bytes(1, 1024, 1024, 100000, 128)
    # Enough memory in total, but not with headroom
    mem = Mock(available = required / util.MEMORY_STREAM_MAX_FRACTION * 0.99)
    with patch('citkid.crs.util.psutil.virtual_memory', return_value = mem):
        with pytest.raises(MemoryError, match = 'memory stream needs'):
            util.check_memory_for_stream(1, 1024, 1024, 100000, 128)


################################################################################
############################## CRS.stream (memory) #############################
################################################################################

@pytest.fixture
def memory_crs(base_crs, tmp_path):
    """CRS with tones on modules 1 and 2 and mocked decimation."""
    crs = base_crs
    crs.analog_bank_high = False
    crs.serial_number = 27
    crs.fres_map = {1: np.array([4.0e9, 4.1e9]), 2: np.array([4.5e9]),
                    3: np.array([])}
    crs.ares_map = {1: np.array([-50., -51.]), 2: np.array([-52.]),
                    3: np.array([])}
    crs.ch_map = {1: np.array([0, 2]), 2: np.array([1]), 3: np.array([])}
    crs.ntones = 3

    async def set_dec(stage, verbose = True):
        crs.sample_freq = util.get_sample_freq(stage)
        crs.dec_stage = stage
        crs.dec_short = True
        crs.dec_module_idxs = [1, 2]
    crs.set_decimation = AsyncMock(side_effect = set_dec)
    grp = zarr.open_group(str(tmp_path / 'data.zarr'), mode = 'w')
    return crs, grp


@pytest.mark.asyncio
async def test_stream_memory_saves_parser_format(memory_crs, tmp_path):
    """Test stream(method = 'memory') captures and saves without the parser."""
    crs, grp = memory_crs
    nframes = int(util.get_sample_freq(6) * 0.5)
    buffers = {1: make_frames(nframes, 2, 0), 2: make_frames(nframes, 2, 1)}
    tmp_dir = tmp_path / 'tmp'

    with patch('citkid.crs.instrument.util.capture_to_memory',
               return_value = (buffers, {1: 0, 2: 0})) as mock_capture, \
         patch('rfmux.tools.parser', create = True) as mock_parser:
        await crs.stream(0.5, 6, grp, verbose = False, method = 'memory',
                         tmp_directory = str(tmp_dir))

    mock_parser.main.assert_not_called()
    mock_capture.assert_called_once_with('eth0', 27, [1, 2], 2, nframes)
    assert not tmp_dir.exists()
    z = grp['z'][...]
    assert z.shape == (2, 3, nframes)
    np.testing.assert_array_equal(z[:, 0], buffers[1][:, 0].T)
    np.testing.assert_array_equal(z[:, 2], buffers[1][:, 1].T)
    np.testing.assert_array_equal(z[:, 1], buffers[2][:, 0].T)
    assert grp['dt'][...] == pytest.approx(1 / util.get_sample_freq(6))
    assert 'timestamp' in grp.attrs
    assert grp['crs_config'].attrs['dec_stage'] == 6


@pytest.mark.asyncio
async def test_stream_memory_warns_dropped_packets(memory_crs):
    """Test a warning when packets are dropped."""
    crs, grp = memory_crs
    buffers = {1: make_frames(4, 2, 0), 2: make_frames(4, 2, 1)}

    with patch('citkid.crs.instrument.util.capture_to_memory',
               return_value = (buffers, {1: 0, 2: 3})):
        with pytest.warns(UserWarning, match = r'\{2: 3\}'):
            await crs.stream(0.01, 6, grp, verbose = False,
                             method = 'memory')


@pytest.mark.asyncio
async def test_stream_memory_check_before_streaming(memory_crs):
    """Test the memory check fails before the decimation is changed."""
    crs, grp = memory_crs

    with patch('citkid.crs.instrument.util.check_memory_for_stream',
               side_effect = MemoryError('too big')) as mock_check, \
         patch('citkid.crs.instrument.util.capture_to_memory') as mock_cap:
        with pytest.raises(MemoryError, match = 'too big'):
            await crs.stream(10.0, 2, grp, verbose = False,
                             method = 'memory')

    nframes = int(util.get_sample_freq(2) * 10.0)
    mock_check.assert_called_once_with(2, 2, 3, nframes, 128.0)
    crs.set_decimation.assert_not_called()
    mock_cap.assert_not_called()


@pytest.mark.asyncio
async def test_stream_memory_restores_shared_decimation(memory_crs):
    """Test a shared session restores stage 6 if the capture fails."""
    crs, grp = memory_crs
    crs.module_idxs = [1, 2, 3]
    crs.shared = True

    with patch('citkid.crs.instrument.util.capture_to_memory',
               side_effect = RuntimeError('stalled')):
        with pytest.raises(RuntimeError, match = 'stalled'):
            await crs.stream(0.1, 3, grp, verbose = False, method = 'memory')

    assert [c.args[0] for c in crs.set_decimation.call_args_list] == [3, 6]


@pytest.mark.asyncio
async def test_stream_memory_verbose_uses_time_bar(memory_crs):
    """Test verbose memory streaming runs the capture with a time bar."""
    crs, grp = memory_crs
    buffers = {1: make_frames(2, 2, 0), 2: make_frames(2, 2, 1)}

    with patch('citkid.crs.instrument.run_with_time_bar',
               return_value = (buffers, {1: 0, 2: 0})) as mock_bar:
        await crs.stream(0.004, 6, grp, verbose = True, method = 'memory')

    args = mock_bar.call_args.args
    assert args[0] is util.capture_to_memory
    assert args[1] == pytest.approx(0.004)
    assert args[2] == 'Streaming'


################################################################################
############################ CRS.capture_ts (memory) ###########################
################################################################################

@pytest.mark.asyncio
async def test_capture_ts_memory_checks_before_writing(base_crs, tmp_path):
    """Test capture_ts checks memory before touching the board."""
    crs = base_crs
    crs.analog_bank_high = False
    crs.nco_freqs = {1: 4.0e9, 2: 4.5e9}
    crs._clear_channels = AsyncMock()
    crs.write_tones = AsyncMock()
    grp = zarr.open_group(str(tmp_path / 'data.zarr'), mode = 'w')

    with patch('citkid.crs.instrument.util.check_memory_for_stream',
               side_effect = MemoryError('too big')) as mock_check:
        with pytest.raises(MemoryError):
            await crs.capture_ts(np.linspace(4e9, 4.1e9, 2000),
                                 np.full(2000, -60.), 5.0, 4, grp,
                                 verbose = False, method = 'memory')

    nframes = int(util.get_sample_freq(4) * 5.0)
    mock_check.assert_called_once_with(2, 1024, 2000, nframes, 128.0)
    crs._clear_channels.assert_not_called()
    crs.write_tones.assert_not_called()


@pytest.mark.asyncio
async def test_capture_ts_passes_method_to_stream(base_crs, tmp_path):
    """Test capture_ts passes method to stream and skips the tmp dir."""
    crs = base_crs
    crs.analog_bank_high = False
    crs.nco_freqs = {1: 4.0e9}
    crs._clear_channels = AsyncMock()
    crs.write_tones = AsyncMock(return_value = np.array([4.0e9]))
    crs.stream = AsyncMock()
    grp = zarr.open_group(str(tmp_path / 'data.zarr'), mode = 'w')
    tmp_dir = tmp_path / 'tmp'

    with patch('citkid.crs.instrument.util.write_system_cfg_to_zarr'), \
         patch('citkid.crs.instrument.time.sleep'):
        await crs.capture_ts([4.0e9], [-50.], 1.0, 6, grp, verbose = False,
                             method = 'memory', tmp_directory = str(tmp_dir))

    assert crs.stream.call_args.kwargs['method'] == 'memory'
    assert not tmp_dir.exists()


################################################################################
############################## input validation ################################
################################################################################

def make_validate_args(tmp_path, **overrides):
    """
    Return valid arguments for _validate_stream_input.

    Parameters:
    tmp_path (pathlib.Path): pytest temporary directory.
    overrides: arguments to replace.

    Returns:
    dict: keyword arguments.
    """
    args = dict(
        ts_duration_s = 1.0, dec_stage = 6,
        grp = zarr.open_group(str(tmp_path / 'v.zarr'), mode = 'w'),
        ch_map = None, allow_missing = False,
        tmp_directory = str(tmp_path / 'tmp'), batch_size_mb = 1000,
        chunk_size_mb = 128, delete_parser_data = True, verbose = False,
    )
    args.update(overrides)
    return args


@pytest.mark.parametrize("method", ['disk', None, 'Memory'])
def test_validate_stream_input_invalid_method(tmp_path, method):
    """Test invalid stream methods raise ValueError."""
    with pytest.raises(ValueError, match = 'method must be one of'):
        _validate_stream_input(**make_validate_args(tmp_path), method = method)


def test_validate_stream_input_memory_skips_tmp_dir(tmp_path):
    """Test the memory method does not create or check the tmp dir."""
    tmp_dir = tmp_path / 'tmp'
    (tmp_dir / 'parser_data_00').mkdir(parents = True)
    result = _validate_stream_input(
        **make_validate_args(tmp_path), method = 'memory')
    assert result[5] is None


def test_validate_stream_input_parser_checks_tmp_dir(tmp_path):
    """Test the parser method still refuses an existing data directory."""
    (tmp_path / 'tmp' / 'parser_data_00').mkdir(parents = True)
    with pytest.raises(FileExistsError):
        _validate_stream_input(**make_validate_args(tmp_path))


@pytest.mark.parametrize("dec_stage", [-1, 7])
def test_validate_stream_input_dec_stage_range(tmp_path, dec_stage):
    """Test decimation stages outside [0, 6] raise ValueError."""
    with pytest.raises(ValueError, match = 'between 0 and 6'):
        _validate_stream_input(**make_validate_args(tmp_path,
                                                    dec_stage = dec_stage))

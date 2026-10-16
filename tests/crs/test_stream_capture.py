"""
Tests for streaming.

Covers the packet capture functions in ``citkid.crs.util``
(``capture_to_memory``, ``capture_to_zarr``, and their helpers) and the CRS
streaming methods (``write_tones_for_stream``, ``stream``,
``stream_to_memory``, and ``clear_tones``). Packets come from a scripted fake
of the rfmux receiver.
"""

import multiprocessing
import os
import warnings
import pytest
import numpy as np
import rfmux
import zarr
from contextlib import contextmanager
from unittest.mock import AsyncMock, MagicMock, Mock, call, patch

from citkid.crs import util


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
############################### _receive_blocks ################################
################################################################################

def test_receive_blocks_counts_packets_without_free_slot():
    """Test packets received while no slot is free are counted as dropped."""
    frames = make_frames(5, 2, 9)
    blocks = np.zeros((1, 1, 2, 4), dtype = np.int32)
    free = iter([0, None, 0])
    filled = []

    def on_block(m, slot, k, rows):
        filled.append((m, slot, k, rows, blocks[m, slot, :rows].copy()))

    with patched_streamer([packets_from_frames(frames, 27, 0)]):
        dropped = util._receive_blocks(
            'eth0', 27, [1], 2, 4, blocks, lambda m: next(free), on_block)

    # Frame 2 arrived while no slot was free
    assert dropped == {1: 1}
    assert [f[:4] for f in filled] == [(0, 0, 0, 2), (0, 0, 1, 2)]
    np.testing.assert_array_equal(filled[0][4], frames[:2].reshape(2, 4))
    np.testing.assert_array_equal(filled[1][4], frames[3:].reshape(2, 4))


def test_receive_blocks_calls_poll():
    """Test poll is called after each batch, and its errors propagate."""
    batches = [packets_from_frames(make_frames(4, 2, 10), 27, 0)]
    poll = Mock(side_effect = RuntimeError('writer failed'))
    blocks = np.zeros((1, 1, 4, 4), dtype = np.int32)
    with patched_streamer(batches):
        with pytest.raises(RuntimeError, match = 'writer failed'):
            util._receive_blocks('eth0', 27, [1], 2, 4, blocks,
                                 lambda m: 0, Mock(), poll = poll)
    poll.assert_called_once_with()


################################################################################
############################### capture_to_zarr ################################
################################################################################

def expected_z(frames, ch_map, ntones, nframes):
    """
    Build the expected z array from per-module frames.

    Parameters:
    frames (dict): keys (int) are module indices and values (np.array int32)
        are frames of shape (n, nchannels, 2).
    ch_map (dict): keys (int) are module indices and values (list of int)
        are tone indices.
    ntones (int): number of tones.
    nframes (int): number of time samples.

    Returns:
    np.array int32: array of shape (2, ntones, nframes).
    """
    z = np.zeros((2, ntones, nframes), dtype = np.int32)
    for mi, ch_idxs in ch_map.items():
        data = frames[mi][:nframes, :len(ch_idxs)]
        z[0, ch_idxs] = data[:, :, 0].T
        z[1, ch_idxs] = data[:, :, 1].T
    return z


def two_module_batches(nframes, nchannels):
    """
    Create interleaved packets for modules 1 and 2 of board 27, plus packets
    from another board.

    Parameters:
    nframes (int): frames per module.
    nchannels (int): channels per packet.

    Returns:
    frames (dict): frames by module index.
    batches (list): scripted receive batches.
    """
    frames = {1: make_frames(nframes, nchannels, 0),
              2: make_frames(nframes, nchannels, 1)}
    p1 = packets_from_frames(frames[1], 27, 0)
    p2 = packets_from_frames(frames[2], 27, 1)
    other = packets_from_frames(make_frames(nframes, nchannels, 2), 99, 0)
    h = nframes // 2
    batches = [p1[:h] + other + p2[:3], p2[3:], p1[h:]]
    return frames, batches


@pytest.mark.parametrize("writer", ['thread', 'process'])
def test_capture_to_zarr_writes_tone_order(tmp_path, writer):
    """Test shards are written in tone order, with unused tones left 0."""
    nframes, ntones = 10, 4
    ch_map = {1: [0, 2], 2: [1]}
    frames, batches = two_module_batches(12, 3)
    grp = zarr.open_group(str(tmp_path / 'data.zarr'), mode = 'w')
    z, chunk_N = util._create_z_array(grp, ntones, nframes, 4)
    assert chunk_N == 4

    with patched_streamer(batches):
        dropped = util.capture_to_zarr('eth0', 27, [1, 2], 2, nframes, z,
                                       ch_map, writer = writer)

    assert dropped == {1: 0, 2: 0}
    np.testing.assert_array_equal(
        zarr.open_group(str(tmp_path / 'data.zarr'))['z'][...],
        expected_z(frames, ch_map, ntones, nframes))


def test_capture_to_zarr_memory_store_uses_thread():
    """Test the default writer works for a store that is not local."""
    nframes = 6
    ch_map = {1: [1], 2: [0]}
    frames, batches = two_module_batches(nframes, 2)
    grp = zarr.open_group(zarr.storage.MemoryStore(), mode = 'w')
    z, _ = util._create_z_array(grp, 2, nframes, 4)

    with patched_streamer(batches), \
         patch('citkid.crs.util.multiprocessing.get_context') as mock_ctx:
        util.capture_to_zarr('eth0', 27, [1, 2], 1, nframes, z, ch_map)

    mock_ctx.assert_not_called()
    np.testing.assert_array_equal(z[...],
                                  expected_z(frames, ch_map, 2, nframes))


def test_capture_to_zarr_local_store_uses_process(tmp_path):
    """Test the default writer is a spawned process for a local store."""
    ch_map = {1: [0]}
    batches = [packets_from_frames(make_frames(3, 1, 3), 27, 0)]
    grp = zarr.open_group(str(tmp_path / 'data.zarr'), mode = 'w')
    z, _ = util._create_z_array(grp, 1, 3, 128)

    with patched_streamer(batches), \
         patch('citkid.crs.util.multiprocessing.get_context',
               wraps = multiprocessing.get_context) as mock_ctx:
        util.capture_to_zarr('eth0', 27, [1], 1, 3, z, ch_map)

    mock_ctx.assert_called_once_with('spawn')
    assert not multiprocessing.active_children()


def test_capture_to_zarr_counts_dropped_packets():
    """Test sequence gaps are returned as dropped packets."""
    frames = make_frames(4, 1, 4)
    packets = packets_from_frames(frames, 27, 0)
    for i, (_, _, pkt) in enumerate(packets):
        pkt.seq = [0, 1, 5, 6][i]
    grp = zarr.open_group(zarr.storage.MemoryStore(), mode = 'w')
    z, _ = util._create_z_array(grp, 1, 4, 128)

    with patched_streamer([packets]):
        dropped = util.capture_to_zarr('eth0', 27, [1], 1, 4, z, {1: [0]})

    assert dropped == {1: 3}
    np.testing.assert_array_equal(z[:, 0], frames[:, 0].T)


@pytest.mark.parametrize("writer", ['thread', 'process'])
def test_capture_to_zarr_writer_failure_raises(tmp_path, writer):
    """Test an error in the writer is raised in the caller."""
    batches = [packets_from_frames(make_frames(4, 1, 5), 27, 0)]
    grp = zarr.open_group(str(tmp_path / 'data.zarr'), mode = 'w')
    z, _ = util._create_z_array(grp, 1, 4, 128)

    # Tone index 3 is outside z, so the writer fails on the first shard
    with patched_streamer(batches):
        with pytest.raises(RuntimeError, match = 'Zarr writer failed'):
            util.capture_to_zarr('eth0', 27, [1], 1, 4, z, {1: [3]},
                                 writer = writer)
    assert not multiprocessing.active_children()


def test_capture_to_zarr_stops_writer_on_receive_error(tmp_path):
    """Test the writer process is stopped if receiving fails."""
    grp = zarr.open_group(str(tmp_path / 'data.zarr'), mode = 'w')
    z, _ = util._create_z_array(grp, 1, 4, 128)

    with patch('citkid.crs.util._receive_blocks',
               side_effect = RuntimeError('stalled')):
        with pytest.raises(RuntimeError, match = 'stalled'):
            util.capture_to_zarr('eth0', 27, [1], 1, 4, z, {1: [0]},
                                 writer = 'process')
    assert not multiprocessing.active_children()


@pytest.mark.parametrize("kwargs, match", [
    (dict(nframes = 5), 'z_out must have shape'),
    (dict(ch_map = {1: [0, 1]}), 'more channels than max_ntones'),
    (dict(writer = 'fork'), 'writer must be'),
    (dict(nslots = 0), 'nslots must be'),
    (dict(module_idxs = [1, 5]), 'both analog banks'),
])
def test_capture_to_zarr_invalid_inputs(kwargs, match):
    """Test invalid inputs raise ValueError."""
    grp = zarr.open_group(zarr.storage.MemoryStore(), mode = 'w')
    z, _ = util._create_z_array(grp, 2, 4, 128)
    args = dict(interface = 'eth0', serial_number = 27, module_idxs = [1],
                max_ntones = 1, nframes = 4, z_out = z, ch_map = {1: [0]})
    args.update(kwargs)
    with pytest.raises(ValueError, match = match):
        util.capture_to_zarr(**args)


def test_spawn_without_main_removes_main_module():
    """Test spawned processes are not told to import __main__."""
    original = util.mp_spawn.get_preparation_data
    with util._spawn_without_main():
        data = util.mp_spawn.get_preparation_data('test')
    assert 'init_main_from_name' not in data
    assert 'init_main_from_path' not in data
    assert util.mp_spawn.get_preparation_data is original


################################################################################
############################### output helpers #################################
################################################################################

@pytest.mark.parametrize("dec_stage, nmodules, short, ok", [
    (0, 1, True, True), (1, 3, True, True), (4, 4, False, True),
    (3, 2, False, True), (0, 2, True, False), (1, 4, True, False),
    (3, 3, False, False), (4, 4, True, True)])
def test_check_frame_rate(dec_stage, nmodules, short, ok):
    """Test the frame rate limit, which allows the verified rates."""
    frames = 1 if short else util.LONG_PACKET_FRAMES
    if ok:
        rate = util.check_frame_rate(dec_stage, nmodules, short)
        assert rate == pytest.approx(
            util.get_sample_freq(dec_stage) * nmodules * frames)
    else:
        with pytest.raises(ValueError, match = 'MAX_FRAME_RATE'):
            util.check_frame_rate(dec_stage, nmodules, short)


@pytest.mark.parametrize("dec_stage, ntones, ok", [
    (4, 4096, True), (3, 2048, True), (0, 128, True), (3, 2049, False),
    (0, 257, False)])
def test_check_stream_write_rate(dec_stage, ntones, ok):
    """Test the write rate limit, which allows the verified rates."""
    if ok:
        rate = util.check_stream_write_rate(dec_stage, ntones)
        assert rate == pytest.approx(util.get_sample_freq(dec_stage) * ntones * 8)
    else:
        with pytest.raises(ValueError, match = 'MAX_STREAM_WRITE_RATE'):
            util.check_stream_write_rate(dec_stage, ntones)


def available_memory(nbytes):
    """Patch the available memory seen by citkid.crs.util."""
    return patch('citkid.crs.util.psutil.virtual_memory',
                 return_value = Mock(available = nbytes))


def test_stream_chunk_length_few_tones_fills_min_shard():
    """Test few tones get chunks long enough to fill MIN_SHARD_BYTES."""
    with available_memory(10 ** 12):
        N = util.get_stream_chunk_length(128, 1, 128, 10 ** 9)
    assert N == util.MIN_SHARD_BYTES // (2 * 128 * 4)


def test_stream_chunk_length_many_tones_uses_min_chunk():
    """Test many tones get at least MIN_CHUNK_SAMPLES per chunk."""
    with available_memory(10 ** 12):
        N = util.get_stream_chunk_length(4096, 4, 1024, 10 ** 9)
    assert N == util.MIN_CHUNK_SAMPLES


def test_stream_chunk_length_capped_to_nframes():
    """Test the chunk length is capped to the timestream length."""
    with available_memory(10 ** 12):
        assert util.get_stream_chunk_length(4096, 4, 1024, 100) == 100


def test_stream_chunk_length_memory_cap_warns():
    """Test low memory shortens the chunks, with a warning."""
    # Buffer bytes per sample: 4 modules x 3 slots x 1024 channels x 8 bytes
    # plus 2 shards of 4096 tones x 8 bytes
    per_sample = 4 * util.STREAM_NSLOTS * 1024 * 8 + 2 * 4096 * 8
    available = 1000 * per_sample / util.STREAM_BUFFER_MAX_FRACTION
    with available_memory(available):
        with pytest.warns(UserWarning, match = 'Not enough memory'):
            N = util.get_stream_chunk_length(4096, 4, 1024, 10 ** 9)
    assert N == 1000


def test_stream_chunk_length_no_warning_for_short_streams():
    """Test no warning when the timestream itself is shorter than the
    minimum chunk."""
    per_sample = 4 * util.STREAM_NSLOTS * 1024 * 8 + 2 * 4096 * 8
    available = 1000 * per_sample / util.STREAM_BUFFER_MAX_FRACTION
    with available_memory(available):
        with warnings.catch_warnings():
            warnings.simplefilter('error')
            assert util.get_stream_chunk_length(4096, 4, 1024, 500) == 500


def test_copy_z_array(tmp_path):
    """Test copy_z_array copies the data and the array layout."""
    src_grp = zarr.open_group(str(tmp_path / 'src.zarr'), mode = 'w')
    z_src, _ = util._create_z_array(src_grp, 3, 10, 4)
    data = make_frames(3, 10, 6).transpose(2, 0, 1).copy()
    z_src[...] = data
    dst_grp = zarr.open_group(str(tmp_path / 'dst.zarr'), mode = 'w')

    z_out = util.copy_z_array(z_src, dst_grp, verbose = False)

    np.testing.assert_array_equal(dst_grp['z'][...], data)
    assert z_out.chunks == z_src.chunks
    assert z_out.shards == z_src.shards
    assert z_out.compressors == z_src.compressors


def test_get_counts_to_s21():
    """Test the scale factors, with NaN for tones missing from ch_map."""
    ch_map = {1: np.array([0, 2]), 2: np.array([3])}
    ares_map = {1: np.array([-50., -40.]), 2: np.array([-30.])}
    scale = util.get_counts_to_s21(5, ch_map, ares_map)
    roc = rfmux.core.transferfunctions.VOLTS_PER_ROC / 256 / np.sqrt(2)
    np.testing.assert_allclose(
        scale[[0, 2, 3]], roc / util.dbm_to_vrms(np.array([-50., -40., -30.])))
    assert np.isnan(scale[[1, 4]]).all()


def test_local_store_path(tmp_path):
    """Test the directory of a local group, and None for other stores."""
    root = zarr.open_group(str(tmp_path / 'data.zarr'), mode = 'w')
    grp = root.require_group('a').require_group('b')
    assert util.local_store_path(grp) == \
        os.path.join(str(tmp_path / 'data.zarr'), 'a/b')
    mem = zarr.open_group(zarr.storage.MemoryStore(), mode = 'w')
    assert util.local_store_path(mem) is None


def test_check_free_space(tmp_path):
    """Test the free space check, including for paths that don't exist."""
    usage = Mock(free = 1000)
    with patch('citkid.crs.util.shutil.disk_usage',
               return_value = usage) as mock_usage:
        util.check_free_space(str(tmp_path / 'new' / 'dir'), 1000)
        with pytest.raises(OSError, match = 'only 0.0 GB is free'):
            util.check_free_space(str(tmp_path), 1001)
    assert mock_usage.call_args_list[0] == call(str(tmp_path))


def test_estimate_memory_stream_bytes():
    """Test the receive buffers plus the complex output and conversion."""
    estimate = util.estimate_memory_stream_bytes(2, 100, 150, 1000)
    assert estimate == 2 * 1000 * 100 * 8 + (150 + 100) * 1000 * 16


def test_check_memory_for_stream():
    """Test the check passes with headroom and fails without it."""
    with patch('citkid.crs.util.psutil.virtual_memory',
               return_value = Mock(available = 10 ** 9)):
        assert util.check_memory_for_stream(1, 10, 10, 1000) == \
            util.estimate_memory_stream_bytes(1, 10, 10, 1000)
        with pytest.raises(MemoryError, match = 'CRS.stream'):
            util.check_memory_for_stream(1, 1024, 1024, 100000)


################################################################################
################################## CRS.stream ##################################
################################################################################

@pytest.fixture
def stream_crs(base_crs, tmp_path):
    """CRS with tones on modules 1 and 2, mocked decimation, and a group."""
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


def check_saved_stream(crs, grp, frames, nframes, dec_stage):
    """
    Check the data and metadata saved by CRS.stream.

    Parameters:
    crs (CRS): the streaming CRS.
    grp (zarr.Group): the group streamed to.
    frames (dict): streamed frames by module index.
    nframes (int): expected number of samples.
    dec_stage (int): streaming decimation stage.

    Returns:
    None
    """
    ch_map = {mi: list(v) for mi, v in crs.ch_map.items() if len(v)}
    np.testing.assert_array_equal(grp['z'][...],
                                  expected_z(frames, ch_map, 3, nframes))
    np.testing.assert_array_equal(
        grp['counts_to_s21'][...],
        util.get_counts_to_s21(3, crs.ch_map, crs.ares_map))
    assert grp['dt'][...] == pytest.approx(1 / util.get_sample_freq(dec_stage))
    assert 'timestamp' in grp.attrs
    assert grp['crs_config'].attrs['dec_stage'] == dec_stage


@pytest.mark.asyncio
async def test_stream_saves_data(stream_crs):
    """Test stream writes the streamed data and metadata to grp."""
    crs, grp = stream_crs
    nframes = int(util.get_sample_freq(6) * 0.01)
    frames, batches = two_module_batches(nframes + 2, 2)

    with patched_streamer(batches):
        await crs.stream(0.01, 6, grp, verbose = False)

    check_saved_stream(crs, grp, frames, nframes, 6)


@pytest.mark.asyncio
async def test_stream_tmp_directory_copies_and_deletes(stream_crs, tmp_path):
    """Test stream copies from the temporary store and then deletes it."""
    crs, grp = stream_crs
    nframes = int(util.get_sample_freq(6) * 0.01)
    frames, batches = two_module_batches(nframes, 2)
    tmp_dir = tmp_path / 'fast'

    with patched_streamer(batches):
        await crs.stream(0.01, 6, grp, tmp_directory = str(tmp_dir),
                         verbose = False)

    check_saved_stream(crs, grp, frames, nframes, 6)
    assert tmp_dir.is_dir() and not os.listdir(tmp_dir)


@pytest.mark.asyncio
async def test_stream_tmp_directory_kept_on_failure(stream_crs, tmp_path):
    """Test the temporary store is kept if streaming fails."""
    crs, grp = stream_crs
    tmp_dir = tmp_path / 'fast'

    with patch('citkid.crs.instrument.util.capture_to_zarr',
               side_effect = RuntimeError('stalled')):
        with pytest.raises(RuntimeError, match = 'stalled'):
            await crs.stream(1.0, 6, grp, tmp_directory = str(tmp_dir),
                             verbose = False)

    kept = os.listdir(tmp_dir)
    assert len(kept) == 1 and kept[0].startswith('stream_')
    assert 'z' not in grp and 'counts_to_s21' not in grp


@pytest.mark.asyncio
async def test_stream_capture_arguments(stream_crs):
    """Test stream passes the streamed modules and channels to the capture."""
    crs, grp = stream_crs

    with patch('citkid.crs.instrument.util.capture_to_zarr',
               return_value = {1: 0, 2: 0}) as mock_capture, \
         patch('citkid.crs.instrument.util.get_stream_chunk_length',
               return_value = 1000) as mock_length:
        await crs.stream(100.0, 6, grp, verbose = False)

    nframes = int(util.get_sample_freq(6) * 100.0)
    mock_length.assert_called_once_with(3, 2, 2, nframes)
    args = mock_capture.call_args.args
    assert args[:5] == ('eth0', 27, [1, 2], 2, nframes)
    assert args[5].shape == (2, 3, nframes)
    assert args[5].shards == (2, 3, 1000)
    assert args[5].chunks == (1, 1, 1000)
    assert args[6] is crs.ch_map


@pytest.mark.asyncio
async def test_stream_warns_dropped_packets(stream_crs):
    """Test a warning when packets are dropped."""
    crs, grp = stream_crs
    with patch('citkid.crs.instrument.util.capture_to_zarr',
               return_value = {1: 0, 2: 3}):
        with pytest.warns(UserWarning, match = r'\{2: 3\}'):
            await crs.stream(0.01, 6, grp, verbose = False)


@pytest.mark.asyncio
async def test_stream_restores_shared_decimation_on_failure(stream_crs):
    """Test a shared session restores stage 6 if the capture fails."""
    crs, grp = stream_crs
    crs.module_idxs = [1, 2, 3]
    crs.shared = True

    with patch('citkid.crs.instrument.util.capture_to_zarr',
               side_effect = RuntimeError('stalled')):
        with pytest.raises(RuntimeError, match = 'stalled'):
            await crs.stream(0.1, 3, grp, verbose = False)

    assert [c.args[0] for c in crs.set_decimation.call_args_list] == [3, 6]


@pytest.mark.asyncio
async def test_stream_verbose_uses_time_bar(stream_crs):
    """Test verbose streaming runs the capture with a time bar."""
    crs, grp = stream_crs
    with patch('citkid.crs.instrument.run_with_time_bar',
               return_value = {1: 0, 2: 0}) as mock_bar:
        await crs.stream(0.01, 6, grp, verbose = True)

    args = mock_bar.call_args.args
    assert args[0] is util.capture_to_zarr
    assert args[1] == pytest.approx(0.01)
    assert args[2] == 'Streaming'


@pytest.mark.asyncio
async def test_stream_rejects_write_rate(stream_crs):
    """Test a write rate above the verified limit raises before the
    decimation is changed or anything is created."""
    crs, grp = stream_crs
    with patch('citkid.crs.instrument.util.MAX_STREAM_WRITE_RATE', 1.0):
        with pytest.raises(ValueError, match = 'MAX_STREAM_WRITE_RATE'):
            await crs.stream(1.0, 6, grp, verbose = False)
    crs.set_decimation.assert_not_called()
    assert 'z' not in grp


@pytest.mark.asyncio
async def test_stream_no_tones_raises(stream_crs):
    """Test stream raises before changing the decimation if no tones are
    written."""
    crs, grp = stream_crs
    crs.fres_map, crs.ch_map, crs.ntones = {}, {}, 0
    with pytest.raises(RuntimeError, match = 'No tones are written'):
        await crs.stream(1.0, 6, grp, verbose = False)
    crs.set_decimation.assert_not_called()


@pytest.mark.asyncio
async def test_stream_checks_free_space_first(stream_crs, tmp_path):
    """Test the free space checks run before anything is created."""
    crs, grp = stream_crs
    with patch('citkid.crs.instrument.util.check_free_space',
               side_effect = OSError('full')) as mock_check:
        with pytest.raises(OSError, match = 'full'):
            await crs.stream(1.0, 6, grp, verbose = False)
    nbytes = 2 * 3 * int(util.get_sample_freq(6)) * 4
    mock_check.assert_called_once_with(util.local_store_path(grp), nbytes)
    crs.set_decimation.assert_not_called()
    assert 'z' not in grp


@pytest.mark.asyncio
async def test_stream_checks_tmp_directory_free_space(stream_crs, tmp_path):
    """Test the temporary directory's drive is checked too."""
    crs, grp = stream_crs
    tmp_dir = str(tmp_path / 'fast')
    with patch('citkid.crs.instrument.util.check_free_space') as mock_check, \
         patch('citkid.crs.instrument.util.capture_to_zarr',
               return_value = {1: 0, 2: 0}):
        await crs.stream(0.01, 6, grp, tmp_directory = tmp_dir,
                         verbose = False)
    assert [c.args[0] for c in mock_check.call_args_list] == \
        [util.local_store_path(grp), tmp_dir]


################################################################################
############################# CRS.stream_to_memory #############################
################################################################################

@pytest.mark.asyncio
async def test_stream_to_memory_returns_s21(stream_crs):
    """Test the returned timestream is complex, in tone order, and scaled."""
    crs, _ = stream_crs
    crs.ntones = 4  # tone 3 is not in ch_map
    nframes = int(util.get_sample_freq(6) * 0.01)
    buffers = {1: make_frames(nframes, 2, 0), 2: make_frames(nframes, 2, 1)}

    with patch('citkid.crs.instrument.util.capture_to_memory',
               return_value = (dict(buffers), {1: 0, 2: 0})) as mock_cap:
        zt = await crs.stream_to_memory(0.01, 6, verbose = False)

    mock_cap.assert_called_once_with('eth0', 27, [1, 2], 2, nframes)
    scale = util.get_counts_to_s21(4, crs.ch_map, crs.ares_map)
    assert zt.dtype == np.complex128 and zt.shape == (4, nframes)
    for tone, (mi, ch) in {0: (1, 0), 2: (1, 1), 1: (2, 0)}.items():
        data = buffers[mi][:, ch].astype(np.float64)
        np.testing.assert_allclose(
            zt[tone], (data[:, 0] + 1j * data[:, 1]) * scale[tone])
    assert np.isnan(zt[3].real).all() and np.isnan(zt[3].imag).all()


@pytest.mark.asyncio
async def test_stream_to_memory_checks_memory_first(stream_crs):
    """Test the memory check fails before the decimation is changed."""
    crs, _ = stream_crs
    with patch('citkid.crs.instrument.util.check_memory_for_stream',
               side_effect = MemoryError('too big')) as mock_check, \
         patch('citkid.crs.instrument.util.capture_to_memory') as mock_cap:
        with pytest.raises(MemoryError, match = 'too big'):
            await crs.stream_to_memory(10.0, 2, verbose = False)

    nframes = int(util.get_sample_freq(2) * 10.0)
    mock_check.assert_called_once_with(2, 2, 3, nframes)
    crs.set_decimation.assert_not_called()
    mock_cap.assert_not_called()


@pytest.mark.asyncio
async def test_stream_to_memory_warns_dropped_packets(stream_crs):
    """Test a warning when packets are dropped."""
    crs, _ = stream_crs
    buffers = {1: make_frames(5, 2, 0), 2: make_frames(5, 2, 1)}
    with patch('citkid.crs.instrument.util.capture_to_memory',
               return_value = (buffers, {1: 4, 2: 0})):
        with pytest.warns(UserWarning, match = r'\{1: 4\}'):
            await crs.stream_to_memory(0.01, 6, verbose = False)


@pytest.mark.asyncio
async def test_stream_to_memory_restores_shared_decimation(stream_crs):
    """Test a shared session restores stage 6 after streaming."""
    crs, _ = stream_crs
    crs.module_idxs = [1, 2, 3]
    crs.shared = True
    buffers = {1: make_frames(1192, 2, 0), 2: make_frames(1192, 2, 1)}
    with patch('citkid.crs.instrument.util.capture_to_memory',
               return_value = (buffers, {1: 0, 2: 0})):
        await crs.stream_to_memory(1.0, 5, verbose = False)
    assert [c.args[0] for c in crs.set_decimation.call_args_list] == [5, 6]


@pytest.mark.asyncio
@pytest.mark.parametrize("ts_duration_s, dec_stage", [(0, 6), (-1, 6),
                                                      (1, 7), (1, -1)])
async def test_stream_to_memory_invalid_inputs(stream_crs, ts_duration_s,
                                               dec_stage):
    """Test invalid durations and stages raise before streaming."""
    crs, _ = stream_crs
    with pytest.raises(ValueError):
        await crs.stream_to_memory(ts_duration_s, dec_stage, verbose = False)
    crs.set_decimation.assert_not_called()


################################################################################
########################## CRS.write_tones_for_stream ##########################
################################################################################

@pytest.fixture
def tones_crs(base_crs):
    """CRS with mocked channel clearing and tone writing."""
    crs = base_crs
    crs.analog_bank_high = False
    crs._clear_channels = AsyncMock()
    crs.write_tones = AsyncMock(return_value = np.array([4.0e9 + 3, np.nan]))
    with patch('citkid.crs.instrument.util.write_system_cfg_to_zarr') \
            as mock_cfg, \
         patch('citkid.crs.instrument.time.sleep') as mock_sleep:
        yield crs, mock_cfg, mock_sleep


@pytest.mark.asyncio
async def test_write_tones_for_stream_saves_tones(tones_crs, tmp_path):
    """Test the tones and configuration are saved, and tones are not
    cleared afterwards."""
    crs, mock_cfg, mock_sleep = tones_crs
    grp = zarr.open_group(str(tmp_path / 'data.zarr'), mode = 'w')
    fres, ares = [4.0e9, 9.0e9], [-50., -51.]

    written = await crs.write_tones_for_stream(
        fres, ares, grp = grp, ch_map = {1: [0]}, allow_missing = True)

    np.testing.assert_array_equal(written, [4.0e9 + 3, np.nan])
    np.testing.assert_array_equal(grp['fres'][:], written)
    np.testing.assert_array_equal(grp['fres_requested'][:], fres)
    np.testing.assert_array_equal(grp['ares'][:], ares)
    mock_cfg.assert_called_once_with(crs, grp['crs_config'])
    crs.write_tones.assert_called_once_with(fres, ares, ch_map = {1: [0]},
                                            allow_missing = True)
    assert crs._clear_channels.call_args_list == [call([1, 2, 3, 4])]
    mock_sleep.assert_called_once_with(0.5)


@pytest.mark.asyncio
async def test_write_tones_for_stream_without_grp(tones_crs):
    """Test nothing is saved if grp is None."""
    crs, mock_cfg, _ = tones_crs
    await crs.write_tones_for_stream([4.0e9], [-50.])
    crs.write_tones.assert_called_once()
    mock_cfg.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ['fres', 'fres_requested', 'ares',
                                  'crs_config'])
async def test_write_tones_for_stream_conflicts(tones_crs, tmp_path, name):
    """Test grp conflicts are checked before touching the board."""
    crs, _, _ = tones_crs
    grp = zarr.open_group(str(tmp_path / 'data.zarr'), mode = 'w')
    if name == 'crs_config':
        grp.require_group(name)
    else:
        grp.create_array(name = name, data = np.array([1.0]))
    with pytest.raises(ValueError, match = 'already contains required names'):
        await crs.write_tones_for_stream([4.0e9], [-50.], grp = grp)
    crs._clear_channels.assert_not_called()


@pytest.mark.asyncio
async def test_write_tones_for_stream_invalid_grp(tones_crs):
    """Test grp must be a zarr group or None."""
    crs, _, _ = tones_crs
    with pytest.raises(TypeError, match = 'grp must be'):
        await crs.write_tones_for_stream([4.0e9], [-50.], grp = {})
    crs._clear_channels.assert_not_called()


@pytest.mark.asyncio
async def test_write_tones_for_stream_clears_on_failure(tones_crs, tmp_path):
    """Test tones are cleared again if writing or saving fails."""
    crs, mock_cfg, _ = tones_crs
    mock_cfg.side_effect = ValueError('cfg failed')
    grp = zarr.open_group(str(tmp_path / 'data.zarr'), mode = 'w')
    with pytest.raises(ValueError, match = 'cfg failed'):
        await crs.write_tones_for_stream([4.0e9], [-50.], grp = grp)
    assert crs._clear_channels.call_args_list == [call([1, 2, 3, 4])] * 2


@pytest.mark.asyncio
async def test_capture_ts_rejects_write_rate(tones_crs, tmp_path):
    """Test capture_ts checks the write rate before touching the board."""
    crs, _, _ = tones_crs
    grp = zarr.open_group(str(tmp_path / 'data.zarr'), mode = 'w')
    with pytest.raises(ValueError, match = 'MAX_STREAM_WRITE_RATE'):
        await crs.capture_ts(np.full(257, 4.0e9), np.full(257, -50.), 1.0, 0,
                             grp, verbose = False)
    crs._clear_channels.assert_not_called()
    crs.write_tones.assert_not_called()


@pytest.mark.asyncio
async def test_write_tones_for_stream_then_several_streams(
    tones_crs, stream_crs
):
    """Test the intended workflow: write tones once, stream to several
    subgroups, then clear the tones."""
    crs, _, _ = tones_crs
    _, grp = stream_crs
    await crs.write_tones_for_stream([4.0e9], [-50.], grp = grp)
    with patch('citkid.crs.instrument.util.capture_to_zarr',
               return_value = {1: 0, 2: 0}):
        for name in ['ts_0', 'ts_1']:
            await crs.stream(0.01, 6, grp.require_group(name),
                             verbose = False)
    await crs.clear_tones()

    for name in ['ts_0', 'ts_1']:
        assert {'z', 'counts_to_s21', 'dt', 'crs_config'} <= \
            set(grp[name].keys())
    assert crs.write_tones.call_count == 1
    assert crs._clear_channels.call_count == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("analog_bank_high, expected", [
    (False, [1, 2, 3, 4]), (True, [5, 6, 7, 8])])
async def test_clear_tones(base_crs, analog_bank_high, expected):
    """Test clear_tones clears this session's modules in the bank."""
    crs = base_crs
    crs.analog_bank_high = analog_bank_high
    crs._clear_channels = AsyncMock()
    await crs.clear_tones()
    crs._clear_channels.assert_called_once_with(expected)

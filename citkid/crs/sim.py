"""
A simulated CRS that measures a simulated device, for trying procedures
and analysis without hardware (tests, example notebooks).

``SimulatedCRS`` has the methods the procedures use (``configure_system``,
``set_extended_bw``, ``attach``, ``configure_modules``, ``set_nco``,
``sweep``) and returns the S21 of a ``vna.simulate`` device through a
drawn readout chain, with:

- rfmux's frequency dithering (up to 50 Hz, inward at each channel's
  ends);
- white noise per quadrature, relative to the gain, at ``snr_db`` for
  ``p_ref`` dBm and 100 samples per point, scaling with power and
  sqrt(nsamps);
- a small random complex gain step each time a module's NCO is set (the
  seams between bands);
- optionally, photon pulses and cosmic rays in streams (``pulses``).

``simulate_vna_store`` makes a device store with one VNA sweep of a
simulated device (``crs.procedures.vna_sweep_to_store``) and saves the
device's true parameters in its ``simulation`` group (``load_truth``).
"""
import uuid
from types import SimpleNamespace
import numpy as np
import pandas as pd
from scipy.signal import lfilter
from .. import __version__ as citkid_version
from .. import daq_layout as dl
from . import util
from .instrument import uses_board
from ..vna import simulate as sim
from .procedures import NYQUIST_FREQ, VNA_STEP, vna_range, vna_sweep_to_store

DEFAULT_MODULE_CFG = dict(full_scale_dbm = 7.0, adc_attenuation_db = 0.0,
                          nyquist_zone = 1, cable_length_m = 0.0,
                          adc_calibration_mode = 'AUTO', adc_autocal = True)
# Serial number of simulated boards
SIM_SERIAL = 0
# Default photon and cosmic-ray pulses (``SimulatedCRS.pulses``): rates in
# Hz (photons per tone, cosmic rays on every streamed tone at once),
# amplitudes as fractional frequency shifts, decay time in s
DEFAULT_PULSES = dict(rate = 20., x = 2e-6, tau = 150e-6, cosmic_rate = 1.,
                      cosmic_x = 1e-5, cosmic_spread = 0.5)

class SimulatedCRS:
    """
    A configured CRS measuring a simulated device.

    Attributes:
    truth (pd.DataFrame): the device (``vna.simulate.draw_device``).
    gain (dict): the readout chain (``vna.simulate.draw_gain``).
    snr_db (float): S/N per quadrature at ``p_ref`` and 100 samples.
    p_ref (float): reference power in dBm.
    seam_db, seam_deg (float): std of the gain step at each NCO setting.
    pulses (dict or None): pulses added to streams (``DEFAULT_PULSES``
        keys), or None for none.
    pulse_log (list of dict): one entry per stream with pulses: ``n``
        (samples), ``dt``, ``photons`` (per streamed tone, the sample of
        each photon's maximum), ``cosmic`` (samples of the cosmic rays),
        ``cosmic_x`` (per tone and cosmic ray, the amplitude), ``rows``
        (the streamed tones' rows of the stream).
    """

    nco_state_dir = None    # no NCO state file unless set (tests)

    def __init__(
        self, truth, gain, module_idxs = (1,), snr_db = 40., p_ref = -55.,
        seam_db = 0.05, seam_deg = 0.5, seed = 0, serial_number = SIM_SERIAL,
        global_control = False, pulses = None, board = None
    ):
        """
        Set up the simulated board.

        Parameters:
        truth (pd.DataFrame): device table.
        gain (dict): readout chain.
        module_idxs (tuple of int): this session's modules (1-4).
        snr_db (float): S/N per quadrature at ``p_ref`` dBm and 100
            samples per point. Default 40.
        p_ref (float): reference power in dBm. Default -55.
        seam_db, seam_deg (float): std of the gain step at each NCO
            setting, in dB and degrees. Default 0.05, 0.5.
        seed (int): noise seed. Default 0.
        serial_number (int): reported serial number. Default
            ``SIM_SERIAL`` (0).
        global_control (bool): If True, the session may change the
            decimation. Default False.
        pulses (dict, bool, or None): photon pulses and cosmic rays in
            streams. True: ``DEFAULT_PULSES``; a dict overrides some of
            its keys. None or False (default): no pulses.
        board (dict or None): the simulated hardware's NCO state, module:
            (frequency, gain step), shared between sessions on the same
            board to simulate a restart: ``set_nco`` writes it and
            ``attach`` reads it (``recover_nco_state``, with the NCO
            state file ``nco_state_dir`` as the CRS). Default None: this
            session's own.
        """
        self.truth, self.gain = truth, gain
        if pulses is True:
            pulses = {}
        self.pulses = None if pulses is None or pulses is False else \
            dict(DEFAULT_PULSES, **pulses)
        self.pulse_log = []
        self.snr_db, self.p_ref = float(snr_db), float(p_ref)
        self.seam_db, self.seam_deg = float(seam_db), float(seam_deg)
        self.rng = np.random.default_rng(seed)
        self.serial_number = int(serial_number)
        self.firmware_release = SimpleNamespace(version = 'simulated')
        self.rfmux_version = 'simulated'
        self.module_idxs = list(module_idxs)
        self.active_module_idxs = list(module_idxs)
        self.global_control = bool(global_control)
        # As the CRS: shared if other sessions may use the other modules
        self.shared = sorted(self.module_idxs) != [1, 2, 3, 4] or \
            not self.global_control
        self.analog_bank_high = False
        self.extended_bw = False
        self.bw = 500e6
        self.clock_source = 'VCXO'
        self.dec_stage = 6
        self.dec_module_idxs, self.dec_short = None, None
        self.citkid_version = citkid_version
        self.ch_map = {}
        self.ntones, self._stream_tones = 0, None
        self.sample_freq = 625e6 / (256 * 64 * 2 ** 6)
        self.module_cfg = {m: dict(DEFAULT_MODULE_CFG) for m in module_idxs}
        self.nco_freqs, self.nco_ids, self._seam = {}, {}, {}
        self.board = {} if board is None else board

    @uses_board
    async def configure_system(
        self, clock_source = 'VCXO', full_scale_dbm = 7,
        analog_bank_high = False, verbose = True
    ):
        """
        Configure the simulated board.

        Parameters:
        clock_source (str): recorded.
        full_scale_dbm (float): DAC full scale of this session's modules.
        analog_bank_high (bool): recorded.
        verbose (bool): ignored.
        """
        self.clock_source = clock_source
        self.analog_bank_high = bool(analog_bank_high)
        self.global_control = True
        await self.configure_modules(full_scale_dbm = full_scale_dbm)

    async def set_extended_bw(self, extended):
        """
        Choose the 500 or 600 MHz module bandwidth.

        Parameters:
        extended (bool): If True, 600 MHz.
        """
        self.extended_bw = bool(extended)
        self.bw = 600e6 if extended else 500e6

    @uses_board
    async def attach(self, full_scale_dbm = 7, verbose = True):
        """
        Attach to the board: default module settings with this full scale,
        and the NCO settings of an earlier session that the board still
        has (``recover_nco_state``).

        Parameters:
        full_scale_dbm (float): DAC full scale.
        verbose (bool): ignored.
        """
        await self.configure_modules(full_scale_dbm = full_scale_dbm)
        await self.recover_nco_state()

    async def recover_nco_state(self):
        """
        Take over the NCO settings the simulated board still has from an
        earlier session, as the CRS: where the board's NCO frequency is
        the one in the NCO state file (``nco_state_dir``), the frequency,
        the setting's id, and the gain step are adopted.

        Returns:
        found (list of int): the modules recovered.
        """
        path = util.nco_state_path(self.serial_number, self.nco_state_dir)
        state = util.load_nco_state(path)
        board = {m: self.board[m][0] for m in self.active_module_idxs
                 if m in self.board}
        found = util.recover_nco_ids(state, board)
        for m, (f, i) in found.items():
            self.nco_freqs[m], self.nco_ids[m] = f, i
            self._seam[m] = self.board[m][1]
        return sorted(found)

    async def configure_modules(
        self, full_scale_dbm = 7, adc_attenuation_db = 0, nyquist_zone = 1,
        cable_length_m = 0.0, adc_autocal = True,
        adc_calibration_mode = 'AUTO', module_idxs = None
    ):
        """
        Set the module settings (unset ones go back to their defaults).

        Parameters:
        full_scale_dbm, adc_attenuation_db, nyquist_zone, cable_length_m,
            adc_autocal, adc_calibration_mode: module settings.
        module_idxs (list of int or None): modules. None (default): this
            session's.
        """
        for m in self.active_module_idxs if module_idxs is None \
                else module_idxs:
            self.module_cfg[m] = dict(
                full_scale_dbm = float(full_scale_dbm),
                adc_attenuation_db = float(adc_attenuation_db),
                nyquist_zone = int(nyquist_zone),
                cable_length_m = float(cable_length_m),
                adc_calibration_mode = str(adc_calibration_mode),
                adc_autocal = bool(adc_autocal))

    async def set_nco(self, nco_freqs, verbose = True, force = False):
        """
        Set NCO frequencies not there already, as the CRS; each setting
        gets a new small gain step (and a new id in ``nco_ids``).

        Parameters:
        nco_freqs (dict): module: NCO frequency in Hz.
        verbose (bool): ignored.
        force (bool): If True, set every module's NCO. Default False.

        Returns:
        changed (list of int): the modules whose NCO was set.
        """
        changed = []
        for m, nco in nco_freqs.items():
            if not force and m in self.nco_freqs and \
                    np.isclose(self.nco_freqs[m], nco, atol = 1., rtol = 0.):
                continue
            self.nco_freqs[m] = float(nco)
            self.nco_ids[m] = uuid.uuid4().hex[:12]
            self._seam[m] = 10 ** (self.rng.normal(0, self.seam_db) / 20) * \
                np.exp(1j * np.radians(self.rng.normal(0, self.seam_deg)))
            self.board[m] = (float(nco), self._seam[m])
            changed.append(m)
        util.save_nco_state(
            util.nco_state_path(self.serial_number, self.nco_state_dir),
            self.nco_freqs, self.nco_ids, changed)
        return changed

    @uses_board
    async def sweep(
        self, frequencies, ares, nsamps, ch_map = None, allow_missing = False,
        acq_cfg_grp = None, verbose = True, pbar_description = '',
        progress = None
    ):
        """
        Sweep the simulated device.

        Parameters:
        frequencies (np.array): (M, N) frequencies in Hz.
        ares (np.array): (M,) tone powers in dBm.
        nsamps (int): samples per point.
        ch_map (dict): module: channel rows.
        allow_missing (bool): ignored.
        acq_cfg_grp (zarr.Group or None): if given, the acquisition
            configuration is written there, as by the CRS.
        verbose (bool): ignored.
        pbar_description (str): ignored.
        progress (callable or None): called as ``progress(done, total)``
            in points, at the start and the end. Default None.

        Returns:
        f (np.array): (M, N) dithered frequencies in Hz.
        z (np.array): (M, N) complex S21.

        Raises:
        ValueError: if ``ch_map`` is missing, or a module's tones are
            outside its NCO's bandwidth or its Nyquist zone.
        """
        f = np.array(frequencies, dtype = np.float64)
        ares = np.asarray(ares, dtype = np.float64)
        await self.set_decimation(6)
        if ch_map is None:
            raise ValueError('the simulated CRS needs a ch_map')
        self.ch_map = {int(m): list(rows) for m, rows in ch_map.items()}
        if acq_cfg_grp is not None:
            util.write_acq_cfg_to_zarr(self, acq_cfg_grp)
        if progress is not None:
            progress(0, f.size)
        step = np.ones(f.shape, complex)
        for m, rows in ch_map.items():
            nco = self.nco_freqs[m]
            fm = f[rows]
            fm[:, 1:-1] += 100 * (self.rng.random(fm[:, 1:-1].shape) - 0.5)
            for j in (0, -1):
                fm[:, j] -= 50 * np.sign(fm[:, j] - nco) * \
                    self.rng.random(len(fm))
            if np.any(np.abs(fm - nco) > self.bw / 2):
                raise ValueError(f'module {m}: tones outside the bandwidth')
            zone = self.module_cfg[m]['nyquist_zone']
            if np.any((fm >= NYQUIST_FREQ) != (zone == 2)):
                raise ValueError(f'module {m}: tones outside Nyquist zone '
                                 f'{zone}')
            f[rows] = fm
            step[rows] = self._seam[m]
        power = np.broadcast_to(ares[:, None], f.shape)
        g = sim.gain_s21(f, self.gain) * step
        z = g * sim.device_s21(f, self.truth, power_dbm = power,
                               p_ref = self.p_ref)
        sigma = np.abs(g) * 10 ** (-self.snr_db / 20) * \
            10 ** (-(power - self.p_ref) / 20) * np.sqrt(100 / nsamps)
        z += sigma * (self.rng.standard_normal(f.shape) +
                      1j * self.rng.standard_normal(f.shape))
        if progress is not None:
            progress(f.size, f.size)
        return f, z

    async def sweep_qres(
        self, fres, ares, qres, npoints, nsamps, ch_map = None,
        allow_missing = False, acq_cfg_grp = None, verbose = True,
        pbar_description = 'Sweeping', progress = None
    ):
        """
        Sweep downward over fres / qres around each tone, as the CRS.

        Parameters:
        fres (array-like): center frequencies in Hz.
        ares (array-like): powers in dBm.
        qres (array-like): span factors (spans are fres / qres).
        npoints (int): points per tone.
        nsamps (int): samples per point.
        ch_map, allow_missing, acq_cfg_grp, verbose, pbar_description,
            progress: see ``sweep``.

        Returns:
        f, z (np.array): (M, npoints) frequencies in Hz and S21.
        """
        fres = np.asarray(fres, dtype = np.float64)
        spans = fres / np.asarray(qres, dtype = np.float64)
        freqs = np.linspace(fres + spans / 2, fres - spans / 2, int(npoints)).T
        return await self.sweep(
            freqs, np.asarray(ares, dtype = np.float64), nsamps,
            ch_map = ch_map, allow_missing = allow_missing,
            acq_cfg_grp = acq_cfg_grp, progress = progress)

    async def set_decimation(
        self, dec_stage, short = None, module_idxs = None, verbose = True
    ):
        """
        Set the decimation stage (with global control), or check it
        (without), as the CRS does.

        Parameters:
        dec_stage (int): decimation stage (0-6).
        short, module_idxs: ignored.
        verbose (bool): ignored.

        Raises:
        RuntimeError: without global control, if the board is at another
            stage.
        """
        dec_stage = int(dec_stage)
        if not self.global_control and dec_stage != self.dec_stage:
            raise RuntimeError(
                f'The CRS decimation stage is {self.dec_stage}, but this '
                f'session needs stage {dec_stage}. This session does not '
                'have global control.')
        self.dec_stage = dec_stage
        self.sample_freq = util.get_sample_freq(dec_stage)

    @uses_board
    async def write_tones_for_stream(
        self, fres, ares, grp = None, ch_map = None, allow_missing = False
    ):
        """
        Write tones for streaming, saving them and the configuration to
        ``grp`` as the CRS does ('fres', 'fres_requested', 'ares',
        'crs_config').

        Parameters:
        fres, ares (array-like): tone frequencies (Hz) and powers (dBm).
        grp (zarr.Group or None): group to save to. Default None.
        ch_map (dict): module: tone indices (required).
        allow_missing (bool): If False (default), every tone must be in
            ``ch_map``.

        Returns:
        fres_written (np.array): dithered tone frequencies (NaN for tones
            not written).

        Raises:
        ValueError: if ``ch_map`` is missing, misses tones without
            ``allow_missing``, or puts a tone outside its module's band.
        """
        fres = np.asarray(fres, dtype = np.float64)
        ares = np.broadcast_to(np.asarray(ares, dtype = np.float64),
                               fres.shape)
        if ch_map is None:
            raise ValueError('the simulated CRS needs a ch_map')
        ch_map = {int(m): list(rows) for m, rows in ch_map.items()}
        rows = [r for rs in ch_map.values() for r in rs]
        if not allow_missing and sorted(rows) != list(range(len(fres))):
            raise ValueError('ch_map must hold every tone')
        written = np.full(len(fres), np.nan)
        for m, rs in ch_map.items():
            nco = self.nco_freqs[m]
            fw = fres[rs] + 100 * (self.rng.random(len(rs)) - 0.5)
            if np.any(np.abs(fw - nco) > self.bw / 2):
                raise ValueError(f'module {m}: tones outside the bandwidth')
            written[rs] = fw
        self.ch_map, self.ntones = ch_map, len(fres)
        self._stream_tones = (written, np.array(ares))
        if grp is not None:
            util.write_system_cfg_to_zarr(self, grp.require_group(
                'crs_config'))
            grp.create_array(name = 'fres', data = written)
            grp.create_array(name = 'fres_requested', data = fres)
            grp.create_array(name = 'ares', data = np.array(ares))
        return written

    async def clear_tones(self):
        """
        Clear the written tones.
        """
        self.ch_map, self.ntones, self._stream_tones = {}, 0, None

    @uses_board
    async def stream_to_memory(
        self, ts_duration_s, dec_stage, verbose = True, progress = None
    ):
        """
        Stream the written tones into memory, as S21.

        The S21 at each tone is the device's (``vna.simulate.device_s21``,
        at the tone's power) through the readout chain, plus white noise
        per sample: the sweep noise at 100 samples (``snr_db``) times
        sqrt(100), scaled by sqrt(sample rate / stage-6 sample rate).

        Parameters:
        ts_duration_s (float): length in seconds.
        dec_stage (int): decimation stage (``set_decimation``).
        verbose (bool): ignored.
        progress (callable or None): called as ``progress(0, nsamples)``
            and ``progress(nsamples, nsamples)``, as the CRS reports a
            stream's progress in samples. Default None.

        Returns:
        zt (np.array): (ntones, nsamples) complex S21; NaN rows for tones
            not written.

        Raises:
        RuntimeError: if no tones are written.
        ValueError: if the stream exceeds the frame or write rate limits.
        """
        if not getattr(self, '_stream_tones', None):
            raise RuntimeError('write tones first (write_tones_for_stream)')
        await self.set_decimation(dec_stage)
        fres, ares = self._stream_tones
        per_module = max(len(r) for r in self.ch_map.values())
        util.check_frame_rate(dec_stage, len(self.ch_map),
                              per_module <= util.SHORT_PACKET_CHANNELS)
        util.check_stream_write_rate(dec_stage, self.ntones)
        fs = util.get_sample_freq(dec_stage)
        n = int(round(ts_duration_s * fs))
        if progress is not None:
            progress(0, n)
        ok = np.isfinite(fres)
        zt = np.full((len(fres), n), np.nan + 1j * np.nan)
        f = fres[ok]
        g = sim.gain_s21(f, self.gain)
        z0 = g * sim.device_s21(f, self.truth, power_dbm = ares[ok],
                                p_ref = self.p_ref)
        sigma = np.abs(g) * 10 ** (-self.snr_db / 20) * 10 * \
            10 ** (-(ares[ok] - self.p_ref) / 20) * \
            np.sqrt(fs / util.get_sample_freq(6))
        noise = self.rng.standard_normal((2, ok.sum(), n))
        zt[ok] = z0[:, None] + sigma[:, None] * (noise[0] + 1j * noise[1])
        if self.pulses is not None:
            # dz/dx: a shift x of the resonance acts as f -> f (1 - x)
            eps = 1e-8
            zp = g * sim.device_s21(f * (1 + eps), self.truth,
                                    power_dbm = ares[ok], p_ref = self.p_ref)
            dzdx = -(zp - z0) / eps
            x = self._pulse_x(ok.sum(), n, 1 / fs)
            x['rows'] = np.flatnonzero(ok)
            zt[ok] += dzdx[:, None] * x.pop('x')
            self.pulse_log.append(x)
        if progress is not None:
            progress(n, n)
        # As the CRS: a shared session puts the board back at the sweep
        # stage, so other sessions can sweep
        if self.shared and self.global_control and dec_stage != 6:
            await self.set_decimation(6)
        return zt

    def _pulse_x(self, ntones, n, dt):
        """
        Draw the resonance shift x(t) of photon pulses and cosmic rays.

        Photons arrive on each tone independently (Poisson, ``rate``);
        cosmic rays hit every tone at once (``cosmic_rate``), with an
        amplitude ``cosmic_x`` times a log-normal factor per tone
        (``cosmic_spread``). Pulses rise in one sample and decay with
        ``tau``; x is negative (the resonance moves down).

        Parameters:
        ntones (int): streamed tones.
        n (int): samples.
        dt (float): sample period, s.

        Returns:
        out (dict): ``x`` (ntones, n), and the ``pulse_log`` entries
            ``n``, ``dt``, ``photons``, ``cosmic``, ``cosmic_x``.
        """
        p = self.pulses
        impulses = np.zeros((ntones, n))
        photons = []
        for k in range(ntones):
            t = np.flatnonzero(self.rng.random(n) < p['rate'] * dt)
            impulses[k, t] += p['x']
            photons.append(t)
        cosmic = np.flatnonzero(self.rng.random(n) < p['cosmic_rate'] * dt)
        cx = p['cosmic_x'] * np.exp(
            p['cosmic_spread'] * self.rng.standard_normal((ntones,
                                                           len(cosmic))))
        impulses[:, cosmic] += cx
        a = np.exp(-dt / p['tau'])
        x = -lfilter([1.], [1., -a], impulses, axis = 1)
        return dict(x = x, n = n, dt = dt, photons = photons, cosmic = cosmic,
                    cosmic_x = cx)

    @uses_board
    async def stream(
        self, ts_duration_s, dec_stage, grp, tmp_directory = None,
        verbose = True, progress = None
    ):
        """
        Stream the written tones to a zarr group, in the CRS's format
        'z' int32 counts (2, ntones, nsamples), chunked and sharded along
        time as by the CRS, 'counts_to_s21', 'dt', 'crs_config', attrs
        'timestamp').

        Parameters:
        ts_duration_s (float): length in seconds.
        dec_stage (int): decimation stage.
        grp (zarr.Group): group to save to.
        tmp_directory: ignored.
        verbose (bool): ignored.
        progress (callable or None): see ``stream_to_memory``.
        """
        from datetime import datetime
        grp.attrs['timestamp'] = datetime.now().strftime('%Y%m%d,%H:%M:%S')
        zt = await self.stream_to_memory(ts_duration_s, dec_stage,
                                         progress = progress)
        scale = np.where(np.isfinite(zt[:, 0]),
                         np.nanmax(np.abs(zt), axis = 1) / 2 ** 30, np.nan)
        counts = np.zeros((2,) + zt.shape, dtype = np.int32)
        ok = np.isfinite(scale)
        counts[0, ok] = np.round(zt[ok].real / scale[ok, None])
        counts[1, ok] = np.round(zt[ok].imag / scale[ok, None])
        per_module = max(len(r) for r in self.ch_map.values())
        short = per_module <= util.SHORT_PACKET_CHANNELS
        max_ntones = util.SHORT_PACKET_CHANNELS if short else \
            util.LONG_PACKET_CHANNELS
        z, _ = util._create_z_array(
            grp, zt.shape[0], zt.shape[1], util.get_stream_chunk_length(
                zt.shape[0], len(self.ch_map), max_ntones, zt.shape[1]))
        z[...] = counts
        grp.create_array(name = 'counts_to_s21', data = scale)
        grp.create_array(name = 'dt', data = np.array(
            1 / util.get_sample_freq(dec_stage)))
        util.write_acq_cfg_to_zarr(self, grp.require_group('crs_config'))

################################################################################
################################# Simulated store ##############################
################################################################################
def save_truth(root, truth, attrs = None):
    """
    Save a simulated device's true parameters in the store's
    ``simulation`` group (one array per column).

    Parameters:
    root (zarr.Group): store root, writable.
    truth (pd.DataFrame): device table (``vna.simulate.dressed_truth``).
    attrs (dict or None): settings to record. Default None.
    """
    grp = dl.create_group(root, 'simulation', 'raw', attrs = attrs)
    for c in truth.columns:
        dl.write_array(grp, c, truth[c].values)
    grp.attrs['columns'] = list(truth.columns)

def load_truth(root):
    """
    Load a simulated device's true parameters.

    Parameters:
    root (zarr.Group, str, or os.PathLike): store root or path.

    Returns:
    truth (pd.DataFrame): device table.
    """
    grp = dl.open_store(root)['simulation']
    return pd.DataFrame({c: grp[c][:] for c in grp.attrs['columns']})

async def simulate_vna_store(
    path, device = 'simdevice', cooldown = 'sim', seed = 0, nyquist = '1',
    module_idxs = (1,), amplitude = -55., nsamps = 100, step = VNA_STEP,
    snr_db = 40., device_kwargs = None, gain_kwargs = None, verbose = True
):
    """
    Make a device store holding one VNA sweep (``vna0``) of a simulated
    device, taken with the same procedure as a real sweep
    (``crs.procedures.vna_sweep_to_store``), and the device's true
    parameters (``simulation``, ``load_truth``).

    Parameters:
    path (str or os.PathLike): new store path.
    device, cooldown (str): store metadata. Default 'simdevice', 'sim'.
    seed (int): random seed of the device, gain, and noise. Default 0.
    nyquist (str): sweep range: '1' (default), '2', or 'both'.
    module_idxs (tuple of int): modules to sweep with. Default (1,).
    amplitude (float): tone power in dBm. Default -55.
    nsamps (int): samples per point. Default 100.
    step (float): df/f. Default ``VNA_STEP``.
    snr_db (float): S/N per quadrature at -55 dBm and 100 samples. Default
        40.
    device_kwargs (dict or None): ``vna.simulate.draw_device`` arguments.
    gain_kwargs (dict or None): ``vna.simulate.draw_gain`` arguments.
    verbose (bool): If True (default), print progress.

    Returns:
    root (zarr.Group): the store root.
    truth (pd.DataFrame): the device's true (bare and dressed) parameters.
    """
    rng = np.random.default_rng(seed)
    truth = sim.draw_device(rng, **(device_kwargs or {}))
    gain = sim.draw_gain(rng, **(gain_kwargs or {}))
    crs = SimulatedCRS(truth, gain, module_idxs, snr_db = snr_db,
                       seed = seed + 1)
    await crs.attach()
    root = dl.create_store(path, device, cooldown,
                           setup = dict(simulated = True, seed = seed))
    fmin, fmax = vna_range(nyquist)
    await vna_sweep_to_store(crs, root, fmin, fmax, amplitude, nsamps,
                             step = step, verbose = verbose)
    truth = sim.dressed_truth(truth)
    save_truth(root, truth, attrs = dict(
        seed = seed, amplitude = amplitude, nsamps = nsamps, snr_db = snr_db,
        device_kwargs = device_kwargs or {}, gain_kwargs = gain_kwargs or {},
        gain = gain))
    return root, truth

def simulated_crs(
    root, module_idxs = (1,), seed = None, global_control = False,
    pulses = None
):
    """
    Make a simulated CRS measuring the device of a simulated store, for
    the later steps (tuning, streams) of a store made by
    ``simulate_vna_store``.

    Parameters:
    root (zarr.Group, str, or os.PathLike): the simulated store.
    module_idxs (tuple of int): this session's modules. Default (1,).
    seed (int or None): noise seed. None (default): the store's seed + 2.
    global_control (bool): If True, the session may change the decimation.
        Default False.
    pulses (dict, bool, or None): photon pulses and cosmic rays in streams
        (``SimulatedCRS``). Default None (none).

    Returns:
    crs (SimulatedCRS): the board, not yet attached.

    Raises:
    KeyError: if the store has no simulated device.
    """
    root = dl.open_store(root)
    attrs = dict(root['simulation'].attrs)
    gain = {k: np.asarray(v, dtype = np.float64) if isinstance(v, list)
            else v for k, v in attrs['gain'].items()}
    return SimulatedCRS(
        load_truth(root), gain, module_idxs, snr_db = attrs['snr_db'],
        seed = attrs['seed'] + 2 if seed is None else seed,
        global_control = global_control, pulses = pulses)

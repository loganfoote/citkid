"""
A fake configured CRS for testing procedures and procedure scripts without
hardware.
"""
from types import SimpleNamespace
import numpy as np

DEFAULT_MODULE_CFG = dict(full_scale_dbm = 7.0, adc_attenuation_db = 0.0,
                          nyquist_zone = 1, cable_length_m = 0.0,
                          adc_calibration_mode = 'AUTO', adc_autocal = True)


class FakeVnaCRS:
    """
    Fake configured CRS: records NCO settings, sweeps and configuration
    calls, and returns S21 with one dip per resonance in ``fres``.

    Attributes:
    nco_calls (list of dict): every ``set_nco`` argument.
    sweep_calls (list of dict): every ``sweep``'s inputs.
    calls (list of tuple): (method name, kwargs) of the configuration
        methods called.
    sweep_zones (list of dict): each sweep's module Nyquist zones.
    """

    def __init__(
        self, modules = (1,), bw = 500e6, fres = (), fail_pass = None
    ):
        """
        Set up the fake board.

        Parameters:
        modules (tuple of int): the session's modules in the active bank.
        bw (float): module bandwidth in Hz.
        fres (tuple of float): resonance frequencies in Hz.
        fail_pass (int or None): sweep call index that raises. Default None.
        """
        self.serial_number = 46
        self.firmware_release = SimpleNamespace(version = '1.6.0')
        self.rfmux_version = '1.4.1'
        self.module_idxs = list(modules)
        self.active_module_idxs = list(modules)
        self.global_control = True
        self.nco_freqs = {}
        self.analog_bank_high = False
        self.extended_bw = bw > 500e6
        self.bw = bw
        self.clock_source = 'VCXO'
        self.module_cfg = {m: dict(DEFAULT_MODULE_CFG) for m in modules}
        self.fres = np.asarray(fres, dtype = float)
        self.fail_pass = fail_pass
        self.nco_calls, self.sweep_calls, self.calls = [], [], []
        self.sweep_zones = []
        self.rng = np.random.default_rng(0)

    async def configure_system(self, **kwargs):
        """
        Record a ``configure_system`` call.

        Parameters:
        **kwargs: recorded.
        """
        self.calls.append(('configure_system', kwargs))
        self.clock_source = kwargs.get('clock_source', self.clock_source)
        self.analog_bank_high = kwargs.get('analog_bank_high',
                                           self.analog_bank_high)

    async def set_extended_bw(self, extended):
        """
        Record and apply ``set_extended_bw``.

        Parameters:
        extended (bool): extended bandwidth on/off.
        """
        self.calls.append(('set_extended_bw', dict(extended = extended)))
        self.extended_bw = bool(extended)
        self.bw = 600e6 if extended else 500e6

    async def attach(self, full_scale_dbm = 7, verbose = True):
        """
        Record an ``attach`` call.

        Parameters:
        full_scale_dbm (float): DAC full scale.
        verbose (bool): ignored.
        """
        self.calls.append(('attach', dict(full_scale_dbm = full_scale_dbm)))
        for cfg in self.module_cfg.values():
            cfg['full_scale_dbm'] = float(full_scale_dbm)

    async def configure_modules(self, module_idxs = None, **kwargs):
        """
        Record and apply ``configure_modules``.

        Parameters:
        module_idxs (list of int or None): modules to configure. None
            (default): all of them.
        **kwargs: module settings.
        """
        self.calls.append(('configure_modules',
                           dict(kwargs, module_idxs = module_idxs)))
        for m in self.module_cfg if module_idxs is None else module_idxs:
            self.module_cfg[m].update(
                {k: v for k, v in kwargs.items() if k in self.module_cfg[m]})

    async def set_nco(self, nco_freqs, verbose = True):
        """
        Record and set NCO frequencies.

        Parameters:
        nco_freqs (dict): module: NCO frequency in Hz.
        verbose (bool): ignored.
        """
        self.nco_calls.append(dict(nco_freqs))
        self.nco_freqs.update(nco_freqs)

    async def sweep(
        self, frequencies, ares, nsamps, ch_map = None, allow_missing = False,
        verbose = True, pbar_description = ''
    ):
        """
        Record a sweep and return S21 at the requested frequencies,
        dithered like rfmux (up to 50 Hz, inward at each channel's ends).
        Checks that each module's tones are within ``bw / 2`` of its NCO and
        in its Nyquist zone.

        Parameters:
        frequencies (np.array): (M, N) frequencies in Hz.
        ares (np.array): (M,) powers in dBm.
        nsamps (int): samples per point.
        ch_map (dict): module: channel rows.
        allow_missing (bool): ignored.
        verbose (bool): ignored.
        pbar_description (str): recorded.

        Returns:
        f, z (np.array): (M, N) frequencies and S21.
        """
        if self.fail_pass is not None and \
                len(self.sweep_calls) == self.fail_pass:
            raise RuntimeError('sweep failed')
        self.sweep_calls.append(dict(
            frequencies = np.array(frequencies), ares = np.array(ares),
            nsamps = nsamps, ch_map = ch_map, desc = pbar_description))
        self.sweep_zones.append({m: self.module_cfg[m]['nyquist_zone']
                                 for m in ch_map})
        f = np.array(frequencies, dtype = float)
        for m, rows in ch_map.items():
            nco = self.nco_freqs[m]
            fm = f[rows]
            fm[:, 1:-1] += 100 * (self.rng.random(fm[:, 1:-1].shape) - 0.5)
            for j in (0, -1):
                fm[:, j] -= 50 * np.sign(fm[:, j] - nco) * \
                    self.rng.random(len(fm))
            f[rows] = fm
            assert np.all(np.abs(fm - nco) <= self.bw / 2)
            zone = self.module_cfg[m]['nyquist_zone']
            assert np.all(fm < 2.5e9) if zone == 1 else np.all(fm >= 2.5e9)
        z = np.ones(f.shape, complex) * np.exp(-2j * np.pi * f * 20e-9)
        for fr in self.fres:
            z *= 1 - 0.5 / (1 + 2j * 2e4 * (f - fr) / fr)
        return f, z

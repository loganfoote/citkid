import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.widgets import TextBox
from ..util import outpath

class peakFinder:
    def __init__(self, x_data, y_data, fr_initial, outpath, overwrite = False):
        """Interactive peak finder for a full vna sweep.

        Parameters:
        x_data (np.array): Frequency data in Hz
        y_data (np.array): |S21| Magnitude data
        fr_initial (np.array): Resonance frequency guesses in Hz
        outpath (str): path to an npy file to save the list of resonance
            frequencies after adjustments
        overwrite (bool): if True, overwrites the npy file. If False and outpath
            exists, raises an error.
        """
        outpath = fix_path(outpath)
        if os.path.exists(outpath) and not overwrite:
            raise FileExistsError(f'{outpath} already exists')
        self.control_is_held = False
        self.shift_is_held = False
        self.x_data = np.array(x_data)
        self.y_data = np.array(y_data)
        self.outpath = outpath
        self.fres = list(fr_initial)

        self.vlines = []
        self.line = None

        self.setup_plot()
        self.ax.plot([],[],'sk',
                     label = 'shift + right click: place resonance')
        self.ax.plot([], [], 'sk',
                     label = 'ctrl + right click: remove resonance')
        self.ax.plot([],[],'sk', label = 'a/enter: save')
        self.ax.plot([],[],'sk', label = 'z: pan left')
        self.ax.plot([],[],'sk', label = 'x: pan right')
        self.ax.legend(fontsize = 5, ncols = 2, loc = 'lower right')
        self.initialize_plot()
        plt.show()

    def setup_plot(self):
        """
        Sets up the plot
        """
        self.fig, self.ax = plt.subplots(figsize = [6, 4], dpi = 300)
        self.ax.set_xlabel('Frequency (MHz)')
        self.ax.set_ylabel(r'$|S_{21}|$ (dB)')
        self.ax.grid()

        self.done_button_ax = self.fig.add_axes([0.85, 0.9, 0.1, 0.05])
        self.done_button = plt.Button(self.done_button_ax, 'Done')
        self.done_button.on_clicked(self._on_done)
        self.cid = self.fig.canvas.mpl_connect('button_press_event',
                                               self._on_click)
        self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)
        self.fig.canvas.mpl_connect('key_release_event', self._on_key_release)
        # Enable interactive navigation toolbar
        self.toolbar = plt.rcParams['toolbar']
        if self.toolbar not in {'None', 'toolbar2', 'toolmanager'}:
            self.toolbar = plt.rcParams['toolbar'] = 'toolbar2'

    def initialize_plot(self):
        """
        Initializes the plot with the resonance data and resonances
        """
        if len(self.vlines):
            for vline in self.vlines:
                vline.remove()
        self.vlines = []
        for xi in self.fres:
            self.vlines.append(self.ax.axvline(xi / 1e6, color='red',
                                               linestyle = '--',
                                               alpha = 0.8))
        if self.line is not None:
            self.line.remove()
        self.line, = self.ax.plot(self.x_data / 1e6, self.y_data,
                                  color='blue', linewidth=1)
        xd = (max(self.x_data) - min(self.x_data)) / 10 * 1e-6
        self.ax.set_xlim(min(self.x_data) / 1e6 - xd,
                         max(self.x_data) / 1e6 + xd)
        yd = (max(self.y_data) - min(self.y_data)) / 10
        self.ax.set_ylim(min(self.y_data) - yd, max(self.y_data) + yd)
        self.fig.canvas.draw()

    def _on_click(self, event):
        """
        shift + right click: place new resonance
        ctrl + right click: remove nearest resonance in plot xlim
        """
        if event.button == 3:
            clicked_x, clicked_y = event.xdata, event.ydata
            if self.shift_is_held:
                self.fres.append(clicked_x * 1e6)
                self.fres = list(np.sort(self.fres))
            elif self.control_is_held:
                if len(self.fres):
                    ix = np.argmin(abs(clicked_x * 1e6 - self.fres))
                    fr = self.fres[ix]
                    x0, x1 = self.ax.get_xlim()
                    if (fr <= x1 * 1e6) and (fr >= x0 * 1e6):
                        self.fres.pop(ix)
            else:
                print('shift + right click to add peak')
        self._update_plot()

    def _on_key_press(self, event):
        """
        a or enter: saves data
        x: pan right
        z: pan left
        """
        if event.key == 'shift':
            self.shift_is_held = True
        elif event.key == 'control':
            self.control_is_held = True
        elif event.key in ['a', 'enter']:
            self._save()
        elif event.key == ' ':
            self._go_back()
        elif event.key == 'x':
            x0, x1 = self.ax.get_xlim()
            xd = x1 - x0
            self.ax.set_xlim(x0 + xd * 0.2, x1 + xd * 0.2)
            self._update_plot()
        elif event.key == 'z':
            x0, x1 = self.ax.get_xlim()
            xd = x1 - x0
            self.ax.set_xlim(x0 - xd * 0.2, x1 - xd * 0.2)
            self._update_plot()

    def _on_key_release(self, event):
        """
        Marks if shift or control are released
        """
        if event.key == 'shift':
            self.shift_is_held = False
        elif event.key == 'control':
            self.control_is_held = False

    def _update_plot(self):
        """
        Updates the plot after modifying fres
        """
        old_vlines = self.vlines
        self.vlines = []
        for vline in old_vlines:
            vline.remove()
        for xi in self.fres:
            self.vlines.append(self.ax.axvline(xi / 1e6, color='red',
                                               linestyle = '--',
                                               alpha = 0.8))
        self.fig.canvas.draw()

    def _go_back(self):
        """
        Does nothing, used for single resonance peak finder
        """
        pass

    def _save(self):
        """
        Saves fres
        """
        np.save(self.outpath, np.array(self.fres))

    def _on_done(self, event):
        """
        Disconnect and close the plot, save data
        """
        self.fig.canvas.mpl_disconnect(self.cid)  # Disconnect event handler
        plt.close(self.fig)  # Close the plot
        self._save()

################################################################################
########################### Single resonance classes ###########################
################################################################################

class poptFinderSingle:
    def __init__(self, power, anl, sfactor, res, anl_threshold = 0.65, res_threshold = 2e-3):
        """
        interactive optimal power finder

        Parameters:
        power (array-like): array of powers to optimize
        anl (array-like): array of nonlinearity parameters
        sfactor (array-like): array of ratios of parallel to perpendicular noise
        res (array-like): array of IQ fit residuals
        anl_threshold (float): highest value of anl to allow for optimization, unless all the
            IQ fits were bad
        res_threshold (float): IQ fits are considered 'bad' and disregarded for res > res_threshold
        """
        power, anl, sfactor, res = np.asarray(power), np.asarray(anl), np.asarray(sfactor), np.asarray(res)
        ix = np.argsort(power)
        self.p0, self.a0, self.s0, self.r0 = power[ix], anl[ix], sfactor[ix], res[ix]
        self.anl_threshold = anl_threshold
        self.res_threshold = res_threshold

        self.setup_plot()
        self.ax_s.plot([],[],'sk',
                        label = 'click: choose optimal power')
        self.ax_s.plot([],[],'sk', label = 'a/enter: save')
        self.ax_s.legend(fontsize = 5, ncols = 2, loc = 'lower right')
        self.initialize_popt()
        self.initialize_plot()
        plt.show()

    def setup_plot(self):
        """
        Sets up the plot
        """
        self.fig, [self.ax_s, self.ax_a] = plt.subplots(2, 1, figsize = [6, 5], dpi = 300,
                                                        layout = 'tight', sharex = True)
        self.initialize_plot_axes()
        self.cid = self.fig.canvas.mpl_connect('button_press_event',
                                               self._on_click)
        self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)
        self.fig.canvas.mpl_connect('key_release_event', self._on_key_release)
        # Enable interactive navigation toolbar
        self.toolbar = plt.rcParams['toolbar']
        if self.toolbar not in {'None', 'toolbar2', 'toolmanager'}:
            self.toolbar = plt.rcParams['toolbar'] = 'toolbar2'

    def initialize_plot_axes(self):
        """
        Initializes the plot axis labels
        """
        self.ax_s.set(ylabel = r'$S_{\mathrm{par}} / S_{\mathrm{per}}$')
        self.ax_a.set(ylabel = r'$a_{\mathrm{nl}}$', xlabel = 'ouput power (dBm)')
        self.ax_s.grid()
        self.ax_a.grid()

    def initialize_popt(self):
        ix = self.r0 < self.res_threshold
        self.p1, self.a1, self.s1 = self.p0[ix], self.a0[ix], self.s0[ix]
        ix = np.argmax(self.s1)
        self.p2, self.a2, self.s2 = self.p0[ix], self.a1[ix], self.s1[ix]
        if len(self.p1):
            ix = np.where(self.a1 > self.anl_threshold)[0]
            if len(ix):
                ix = ix[0]
                self.p2, self.a2, self.s2 = self.p1[ix], self.a1[ix], self.s1[ix]

    def initialize_plot(self):
        """
        Initializes the plot with the resonance data and resonances
        """
        self.ax_s.plot(self.p0, self.s0, 's', color = plt.cm.viridis(0.), label = 'raw data')
        self.ax_a.plot(self.p0, self.a0, 's', color = plt.cm.viridis(0.), label = 'raw data')
        self.ax_a.plot(self.p1, self.a1, 's', color = plt.cm.viridis(0.5), label = 'good fits')
        self.popt_point_s, = self.ax_s.plot(self.p2, self.s2, 's', color = plt.cm.viridis(1.), label = 'opt power')
        self.popt_point_a, = self.ax_a.plot(self.p2, self.a2, 's', color = plt.cm.viridis(1.), label = 'opt power')
        # self.ax_s.legend()
        self.fig.canvas.draw()

    def _on_click(self, event):
        """
        If shift is held and the right mouse button is clicked, places a new
        resonance frequency where clicked
        """
        if event.button == 1:
            clicked_x, clicked_y = event.xdata, event.ydata
            ix = np.argmin(abs(clicked_x - self.p0))
            self.p2, self.a2 = np.asarray([self.p0[ix]]), np.asarray([self.a0[ix]])
            self.s2 = np.asarray([self.s0[ix]])
        self._update_plot()

    def _on_key_press(self, event):
        """
        Marks if shift or control are pressed. Deletes all resonance frequencies
        if 'x' is pressed
        """
        if event.key in ['a', 'enter']:
            self._on_done(None)
        elif event.key == ' ':
            self._go_back()

    def _on_key_release(self, event):
        """
        Marks if shift or control are released
        """
        pass

    def _update_plot(self):
        """Updates the plot after changing fres"""

        self.popt_point_s.set_data(self.p2, self.s2)
        self.popt_point_a.set_data(self.p2, self.a2)
        self.fig.canvas.draw()

    def _go_back(self):
        """
        Does nothing for single resonance
        """
        pass

    def _on_done(self, event):
        """
        Disconnect and close the plot, save data
        """
        self.popt = float(self.p2[0])
        self.fig.canvas.mpl_disconnect(self.cid)  # Disconnect event handler
        plt.close(self.fig)  # Close the plot

class qresFinderSingle:
    def __init__(self, x_data, y_data, fres, span0, other_fres, fres_outpath,
                 span_outpath, x_data_previous = None, y_data_previous = None,
                 fres_previous = None):
        """
        Interactive quality factor finder for a single resonance target sweep.
        Use this to confirm and adjust ranges for fitting fine scan data.

        Parameters:
        x_data (array-like): Frequency data in Hz
        y_data (array-like): complex iq data
        fres (float): Resonance frequency in Hz
        span0 (float): Starting span in Hz around fr for fitting
        other_fres (array-like): list of other resonance frequencies,
            omitting fr. Can include values outside the range of x_data
        fres_outpath (str): path to the NPY file to save the list of fres after
            adjustments
        span_outpath (str): path to the NPY file to save the list of [fmin, fmax]
            spans for fitting
        """
        self.x_data = np.asarray(x_data)
        self.y_data = np.asarray(y_data)
        self.x_data_previous = np.asarray(x_data_previous) if x_data_previous is not None else None
        self.y_data_previous = np.asarray(y_data_previous) if y_data_previous is not None else None
        self.fres_previous = float(fres_previous) if fres_previous is not None else None
        self.dB_data = 20 * np.log10(np.abs(self.y_data))
        self.dB_data_previous = 20 * np.log10(np.abs(self.y_data_previous))
        self.fres_outpath = fres_outpath
        self.span_outpath = span_outpath
        self.fres = float(fres)
        self.span0 = span0
        self.fmin = self.fres - self.span0 / 2
        self.fmax = self.fres + self.span0 / 2
        other_fres = np.asarray(other_fres)
        ix = (other_fres >= min(self.x_data)) & (other_fres <= max(self.x_data))
        self.other_fres = other_fres[ix]

        self.other_vlines = []
        self.vlines = []
        self.line = None
        self.iq_cut = []
        self.span_fill = None
        self.iq_scatter = None
        self.control_is_held = False
        self.shift_is_held = False
        self.span_vline = None
        self.fres_point = []

        self.setup_plot()
        ax = self.ax_iq_p if self.x_data_previous is not None else self.ax_iq
        # Add annotations of commands
        annotation_text = (
            "shift + right click:\nplace resonance\n\n"
            "control + right click:\nchange span\n\n"
            "a/enter:\nsave resonance")
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(1.05, 0.8, annotation_text, transform=ax.transAxes, fontsize=5,
                verticalalignment='top', bbox=props)
        self.initialize_plot()
        plt.show()

    def setup_plot(self):
        """
        Sets up the plot
        """
        if self.x_data_previous is None:
            self.fig, [self.ax, self.ax_iq] = plt.subplots(1, 2, figsize = [6, 3],
                                                        dpi = 300, layout = 'tight')
        else:
            self.fig, [[self.ax_p, self.ax_iq_p], [self.ax, self.ax_iq]] =\
                 plt.subplots(2, 2, figsize = [6, 8], dpi = 300, layout = 'tight')
            self.ax_p.sharex(self.ax)
        self.initialize_plot_axes()
        self.done_button_ax = self.fig.add_axes([0.85, 0.9, 0.1, 0.05])
        self.done_button = plt.Button(self.done_button_ax, 'Done')
        self.done_button.on_clicked(self._on_done)
        self.cid = self.fig.canvas.mpl_connect('button_press_event',
                                               self._on_click)
        self.fig.canvas.mpl_connect('key_press_event', self._on_key_press)
        self.fig.canvas.mpl_connect('key_release_event', self._on_key_release)
        # Enable interactive navigation toolbar
        self.toolbar = plt.rcParams['toolbar']
        if self.toolbar not in {'None', 'toolbar2', 'toolmanager'}:
            self.toolbar = plt.rcParams['toolbar'] = 'toolbar2'

    def initialize_plot_axes(self):
        """
        Initializes the plot axis labels
        """
        lbl = int(round(self.fres / 1e3, 0))
        if self.x_data_previous is not None:
            axs_iq = [self.ax_iq, self.ax_iq_p]
            axs = [self.ax, self.ax_p]
        else:
            axs_iq = [self.ax_iq]
            axs = [self.ax]
        for ax in axs:
            ax.set_ylabel(r'$|S_{21}|$ (dB)')
            ax.grid()
        for ax in axs_iq:
            ax.set_xlabel('I')
            ax.set_ylabel('Q')
            ax.grid()
        self.ax.set_xlabel(f'f - {lbl:,} (kHz)')


    def initialize_plot(self):
        """
        Initializes the plot with the resonance data and resonances
        """
        self.x0 = np.mean(self.x_data) / 1e3
        for p in self.iq_cut:
            p.remove()
            self.iq_cut = []
        if self.span_fill is not None:
            self.span_fill.remove()
            self.span_fill = None
        if self.span_vline is not None:
            self.span_vline.remove()
            self.span_vline = None
        if len(self.other_vlines):
            for vline in self.other_vlines:
                vline.remove()
        for p in self.fres_point:
            p.remove()
        self.other_vlines = []
        if self.line is not None:
            self.line.remove()
        if self.iq_scatter is not None:
            for p in self.iq_scatter:
                p.remove()
        if self.iq_cut is not None:
            for p in self.iq_cut:
                p.remove()


        for xi in self.other_fres:
            self.other_vlines.append(self.ax.axvline(xi / 1e3 - self.x0,
                                    color = plt.cm.viridis(1.), linestyle = '--', alpha = 0.3))
        self.fres_vline = self.ax.axvline(self.fres / 1e3 - self.x0, color='k',
                                           linestyle = '--',
                                           alpha = 0.8)
        if self.line is not None:
            self.line.remove()
        self.line, = self.ax.plot(self.x_data / 1e3 - self.x0,
                                  20 * np.log10(np.abs(self.y_data)),
                                  color=plt.cm.viridis(0.), linewidth=1)
        if self.x_data_previous is not None:
            self.line_p, = self.ax_p.plot(self.x_data_previous / 1e3 - self.x0,
                                  20 * np.log10(np.abs(self.y_data_previous)),
                                  color=plt.cm.viridis(0.), linewidth=1)
        # Set axis limits
        xd = (max(self.x_data) - min(self.x_data)) / 10 * 1e-3

        xx = self.x_data
        self.ax.set_xlim(min(xx) / 1e3 - xd - self.x0,
                         max(xx) / 1e3 + xd - self.x0)
        yd = (max(self.dB_data) - min(self.dB_data)) / 10
        self.ymin = min(self.dB_data) - yd
        self.ymax = max(self.dB_data) + yd
        self.ax.set_ylim(self.ymin, self.ymax)
        imin, imax = min(np.real(self.y_data)), max(np.real(self.y_data))
        qmin, qmax = min(np.imag(self.y_data)), max(np.imag(self.y_data))
        idd = (imax - imin) / 10
        qdd = (qmax - qmin) / 10
        self.ax_iq.set_xlim(imin - idd, imax + idd)
        self.ax_iq.set_ylim(qmin - qdd, qmax + qdd)

        self.span_fill = self.ax.fill_between([self.fmin / 1e3 - self.x0,
                                               self.fmax / 1e3 - self.x0],
                            self.ymin, self.ymax, color = plt.cm.viridis(0.67), alpha = 0.3)
        self.span_vline = None

        self.ydata_ix = (self.x_data <= self.fmax) & (self.x_data >= self.fmin)

        self.iq_scatter = self.ax_iq.plot(np.real(self.y_data),
                                          np.imag(self.y_data), '.', color = plt.cm.viridis(0.))
        if self.x_data_previous is not None:
            self.iq_scatter_p = self.ax_iq_p.plot(np.real(self.y_data_previous),
                                            np.imag(self.y_data_previous), '.', color = plt.cm.viridis(0.))
        self.iq_cut = self.ax_iq.plot(np.real(self.y_data[self.ydata_ix]),
                                      np.imag(self.y_data[self.ydata_ix]), '.', color = plt.cm.viridis(0.67),
                                       alpha = 1)
        ix = np.argmin(abs(self.fres - self.x_data))
        self.fres_point = self.ax_iq.plot(np.real(self.y_data[ix]),
                                        np.imag(self.y_data[ix]), color = 'black',
                                        marker = 'x', markersize = 5)
        if self.x_data_previous is not None:
            ix = np.argmin(abs(self.fres_previous - self.x_data_previous))
            self.fres_point_p = self.ax_iq_p.plot(np.real(self.y_data_previous[ix]),
                                            np.imag(self.y_data_previous[ix]), color = 'black',
                                            marker = 'x', markersize = 5)
        self.fig.canvas.draw()

    def _on_click(self, event):
        """
        If shift is held and the right mouse button is clicked, places a new
        resonance frequency where clicked
        """
        if event.button == 3:
            clicked_x, clicked_y = event.xdata, event.ydata
            if self.shift_is_held:
                self.fres = clicked_x * 1e3 + self.x0 * 1e3
            elif self.control_is_held:
                if self.fmax is None:
                    self.fmax = clicked_x * 1e3 + self.x0 * 1e3
                    if self.fmax < self.fmin:
                        self.fmax, self.fmin = self.fmin, self.fmax
                else:
                    self.fmin = clicked_x * 1e3 + self.x0 * 1e3
                    self.fmax = None
            else:
                print('shift + right click to add peak')
        self._update_plot()

    def _on_key_press(self, event):
        """
        Marks if shift or control are pressed. Deletes all resonance frequencies
        if 'x' is pressed
        """
        if event.key == 'shift':
            self.shift_is_held = True
        elif event.key == 'control':
            self.control_is_held = True
        elif event.key in ['a', 'enter']:
            self._on_done(None)
        elif event.key == 'z':
            self.fres = np.nan
            self.fmin = np.nan
            self.fmax = np.nan
            self._update_plot()
        elif event.key == ' ':
            self._go_back()

    def _on_key_release(self, event):
        """
        Marks if shift or control are released
        """
        if event.key == 'shift':
            self.shift_is_held = False
        elif event.key == 'control':
            self.control_is_held = False

    def _update_plot(self):
        """Updates the plot after changing fres"""
        self.fres_vline.remove()
        self.fres_vline = self.ax.axvline(self.fres / 1e3 - self.x0, color='black',
                                           linestyle = '--',
                                           alpha = 1)

        for p in self.iq_cut:
            p.remove()
            self.iq_cut = []
        if self.span_fill is not None:
            self.span_fill.remove()
            self.span_fill = None
        if self.span_vline is not None:
            self.span_vline.remove()
            self.span_vline = None
        if self.fmax is None:
            self.span_vline = self.ax.axvline(self.fmin / 1e3 - self.x0,
                                color = plt.cm.viridis(0.67), linestyle = '--', alpha = 0.3)
        else:
            self.ydata_ix = (self.x_data <= self.fmax) & (self.x_data >= self.fmin)
            self.span_fill = self.ax.fill_between([self.fmin / 1e3 - self.x0,
                                                   self.fmax / 1e3 - self.x0],
                                                   self.ymin, self.ymax,
                                                   color = plt.cm.viridis(0.67), alpha = 0.3)
            self.iq_cut = self.ax_iq.plot(np.real(self.y_data[self.ydata_ix]),
                                          np.imag(self.y_data[self.ydata_ix]),
                                          '.', color = plt.cm.viridis(0.67), alpha = 0.8)
        for p in self.fres_point:
            p.remove()
            print(p)
        if np.isfinite(self.fres):
            ix = np.argmin(abs(self.fres - self.x_data))
            x, y = np.real(self.y_data[ix]), np.imag(self.y_data[ix])
        else:
            x, y = [], []
        self.fres_point = self.ax_iq.plot(x, y, color = 'black', marker = 'x',
                                          markersize = 5)
        self.fig.canvas.draw()

    def _go_back(self):
        """
        Does nothing for single resonance
        """
        pass

    def _on_done(self, event):
        """
        Disconnect and close the plot, save data
        """
        self.fig.canvas.mpl_disconnect(self.cid)  # Disconnect event handler
        plt.close(self.fig)  # Close the plot
        np.save(self.fres_outpath, np.array(self.fres))
        np.save(self.span_outpath, np.array([self.fmin, self.fmax]))

################################################################################
####################### Single resonance looped classes ########################
################################################################################
class poptFinder(poptFinderSingle):
    def __init__(self, outpath, powers, anls, sfactors, ress, res_indices,
                 anl_threshold = 0.65, res_threshold = 2e-3):
        """
        interactive optimal power finder

        Parameters:
        outpath (str): path to save the output file. Must end in .npy

        Parameters: list of the following parameters
        powers (array-like): array of powers to optimize
        anls (array-like): array of nonlinearity parameters
        sfactors (array-like): array of ratios of parallel to perpendicular noise
        ress (array-like): array of IQ fit residuals
        anl_threshold (float): highest value of anl to allow for optimization, unless all the
            IQ fits were bad
        res_threshold (float): IQ fits are considered 'bad' and disregarded for res > res_threshold
        """
        self.data_index = 0

        self.anl_threshold = anl_threshold
        self.res_threshold = res_threshold
        self.res_indices = np.asarray(res_indices)
        self.powers = powers
        self.anls = anls
        self.sfactors = sfactors
        self.ress = ress
        self.powers_new = np.asarray([p[0] for p in self.powers])
        self.powers_new[self.res_indices >=0] = np.nan
        self.outpath = outpath

        self.setup_plot()
        self.ax_s.plot([],[],'sk',
                        label = 'click: choose optimal power')
        self.ax_s.plot([],[],'sk', label = 'a/enter: save')
        self.ax_s.legend(fontsize = 5, ncols = 2, loc = 'lower right')
        self.set_data_index()
        self.initialize_popt()
        self.initialize_plot()
        plt.show()

    def set_data_index(self):
        """Sets all of the variables for the current data index
           and updates the plot
        """
        di = self.data_index
        while self.res_indices[di] < 0:
            di += 1
            self.data_index = di
        power, anl, sfactor, res = self.powers[di], self.anls[di], self.sfactors[di], self.ress[di]
        power, anl, sfactor, res = np.asarray(power), np.asarray(anl), np.asarray(sfactor), np.asarray(res)
        ix = np.argsort(power)
        self.p0, self.a0, self.s0, self.r0 = power[ix], anl[ix], sfactor[ix], res[ix]

        self.ax_s.set_title(f'Resonator: {int(self.res_indices[di])}')

        self.ax_a.cla()
        self.ax_s.cla()
        self.initialize_plot_axes()
        self.initialize_popt()
        self.initialize_plot()

    def _go_back(self):
        """
        Goes back to the previous data index
        """
        self.powers_new[self.data_index] = np.nan
        np.save(self.outpath, self.powers_new)
        if self.data_index != 0:
            self.data_index -= 1
            while self.res_indices[self.data_index] < 0:
                self.data_index -= 1
            self.set_data_index()

    def _on_done(self, event):
        """
        Saves the data and moves on to the next plot. Disconnects when
           finished with all the resonators
        """
        self.powers_new[self.data_index] = self.p2
        np.save(self.outpath, self.powers_new)
        if self.data_index == len(self.powers) - 1:
            self.fig.canvas.mpl_disconnect(self.cid)
            plt.close(self.fig)
        else:
            self.data_index += 1
            self.set_data_index()


class qresFinder(qresFinderSingle):
    def __init__(self, x_datas, y_datas, fress, span0s,
                 fres_outpaths, span_outpaths, res_indices, ares = None, ares_title = '',
                 titles = None, titles_previous = None,
                 x_datas_previous = None, y_datas_previous = None, fress_previous = None):
        """
        Interactive qres finder to loop over single resonance target sweeps.
        Use this to confirm and adjust ranges for fitting fine scan data.
        All parameters are lists of parameters passed into qresFinderSingle

        Parameters:
        x_datas (array-like, array-like): list of frequency datas in Hz
        y_datas (array-like, array-like): list of complex iq datas
        fress (array-like, float): list of resonance frequencies in Hz
        span0s (array-like, float): Starting spans in Hz around fr for fitting
        other_fres (array-like, array-like): list of lists of other resonance
            frequencies, omitting fr. Can include values outside the range of
            x_data
        fres_outpaths (array-like, str): paths to save the list of fres after
            adjustments
        span_outpaths (array-like, str): paths to save the list of [fmin, fmax]
            spans for fitting
        """
        self.control_is_held = False
        self.shift_is_held = False
        self.resonator_index = 0
        self.x_datas = x_datas
        self.y_datas = y_datas
        self.fress = fress
        self.fres = fress[0]
        self.span0s = span0s
        self.other_fress = np.array([np.delete(fress, i) for i in range(len(fress))])
        self.fres_outpaths = fres_outpaths
        self.span_outpaths = span_outpaths
        self.x_datas_previous = x_datas_previous
        self.y_datas_previous = y_datas_previous
        self.fress_previous = fress_previous
        self.res_indices = res_indices
        self.titles = titles
        self.titles_previous = titles_previous
        self.ares = ares
        self.ares_title = ares_title

        self.other_vlines = []
        self.vlines = []
        self.line = None
        self.iq_cut = []
        self.span_fill = None
        self.span_vline = None
        self.fres_point = []
        self.iq_scatter = None

        self.x_data_previous = self.x_datas_previous[0] if self.x_datas_previous is not None else None
        self.setup_plot()
        self.set_resonator_index()
        plt.show()

    def set_resonator_index(self):
        """Sets all of the variables for the current resonator index
           and updates the plot
        """
        ri = self.resonator_index
        self.ax.set_title(f'Resonator: {int(ri)}')
        self.x_data = np.array(self.x_datas[ri])
        self.y_data = np.array(self.y_datas[ri])
        self.dB_data = 20 * np.log10(np.abs(self.y_data))
        self.fres_outpath = self.fres_outpaths[ri]
        self.span_outpath = self.span_outpaths[ri]
        self.fres = float(self.fress[ri])
        self.span0 = self.span0s[ri]
        self.fmin = self.fres - self.span0 / 2
        self.fmax = self.fres + self.span0 / 2
        self.res_index = self.res_indices[ri]

        if self.x_datas_previous is not None:
            self.x_data_previous = np.array(self.x_datas_previous[ri])
            self.y_data_previous = np.array(self.y_datas_previous[ri])
            self.dB_data_previous = 20 * np.log10(np.abs(self.y_data_previous))
            self.fres_previous = float(self.fress_previous[ri])

        self.other_vlines = []
        self.vlines = []
        self.line = None

        self.other_fres = np.array(self.other_fress[ri])
        ix = (self.other_fres >= min(self.x_data)) & (self.other_fres <= max(self.x_data))
        self.other_fres = self.other_fres[ix]
        self.ax.cla()
        self.ax_iq.cla()
        if self.x_datas_previous is not None:
            self.ax_p.cla()
            self.ax_iq_p.cla()
        self.initialize_plot_axes()
        self.initialize_plot()

        self.ax.set_title(f'Fn {self.res_index}')
        if self.titles is not None:
            self.ax_iq.set_title(self.titles[ri])
        if self.x_datas_previous is not None and self.titles_previous is not None:
            self.ax_iq_p.set_title(self.titles_previous[ri])

        # Add annotations of commands
        ax = self.ax_iq_p if self.x_datas_previous is not None else self.ax_iq
        annotation_text = (
            "shift + right click:\nplace resonance\n\n"
            "control + right click:\nchange span\n\n"
            "a/enter:\nsave resonance\n\n"
            "space:\ngo back")
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(1.05, 0.8, annotation_text, transform=ax.transAxes, fontsize=5,
                verticalalignment='top', bbox=props)

        # adjustment box
        if self.ares is not None:
            axbox = self.fig.add_axes([0.5, 0.2, 0.1, 0.05])  # Position of the text box (left, bottom, width, height)
            text_box = TextBox(axbox, '', initial="25", label_pad=0.05)
            ax_title = self.fig.add_axes([0.5, 0.25, 0.2, 0.05])  # Position for the title
            ax_title.axis('off')  # Hide the axis
            ax_title.text(0.5, 0.5, self.ares_title, horizontalalignment='center', verticalalignment='center')
            # Connect the text box to the submit function
            # text_box.on_submit(submit)
        plt.draw()

    def _go_back(self):
        """
        Goes back to the previous resonator
        """
        if self.resonator_index != 0:
            self.resonator_index -= 1
            self.set_resonator_index()

    def _on_done(self, event):
        """
        Saves the data and moves on to the next plot. Disconnects when
           finished with all the resonators
        """
        np.save(self.fres_outpath, np.array(self.fres))
        np.save(self.span_outpath, np.array([self.fmin, self.fmax]))
        if self.resonator_index == len(self.fress) - 1:
            self.fig.canvas.mpl_disconnect(self.cid)
            plt.close(self.fig)
        else:
            self.resonator_index += 1
            self.set_resonator_index()


def run_qres_opt(out_directory, f, z, fres, qres, ares, fcal_indices, res_indices, bypass_indices,
                 f_previous, z_previous, fres_previous, ares_previous, delete_temp_data = False):
    """
    runs qres optimization

    Parameters:
    out_directory (str): directory to save the file
    fcal_indices (array-like): calibration tone indices
    res_indices (array-like): resonator indices
    bypass_indices (array-like): resonator indices to bypass
    delete_temp_data (bool): if True, deletes temporary data that was created while running
    The following parameters are lists of lists, where the first index is the resonator index and
        the second index is the data
    f (M X N array-like, float): frequency data in Hz
    z (M X N array-like, complex): complex S21 data in Hz
    fres (M X 1 array-like, float): resonance frequencies in Hz
    qres (M X 1 array-like, float): q-factors
    ares (M X 1 array-like, float): parameter that was varied from the last dataset
    parameters with _previous suffix: same as above, but from the previous value of ares
    """
    path = out_directory + 'fres.npy'
    if os.path.exists(path):
        response = ''
        while response != 'y':
            response = input(f'{path} already exists!! Overwrite (y/n)?')
            if response == 'n':
                raise FileExistsError(f'{path} already exists!!')
    os.makedirs(out_directory, exist_ok = True)
    f, z = np.asarray(f), np.asarray(z)
    fres, qres, ares = np.asarray(fres), np.asarray(qres), np.asarray(ares)
    fcal_indices, bypass_indices = np.asarray(fcal_indices), np.asarray(bypass_indices)

    onres_indices = [i for i in range(len(fres)) if i not in fcal_indices and res_indices[i] not in bypass_indices]
    fres_outpaths = [out_directory + f'fres_{ri:04d}.npy' for ri in res_indices[onres_indices]]
    span_outpaths = [out_directory + f'span_{ri:04d}.npy' for ri in res_indices[onres_indices]]
    titles = np.array([f'{round(a, 2)} dB' for a in ares])

    if f_previous is not None:
        f_previous, z_previous = np.asarray(f_previous), np.asarray(z_previous)
        fres_previous, ares_previous = np.asarray(fres_previous), np.asarray(ares_previous)
        x_datas_previous = f_previous[onres_indices]
        y_datas_previous = z_previous[onres_indices]
        titles_previous = np.array([f'{round(a, 2)} dB' for a in ares_previous])
        titles_previous = titles_previous[onres_indices]
        fres_previous = fres_previous[onres_indices]
    else:
        x_datas_previous = None
        y_datas_previous = None
        titles_previous = None

    qresFinder(f[onres_indices], z[onres_indices], fres[onres_indices],
               fres[onres_indices] / qres[onres_indices], fres_outpaths, span_outpaths,
               x_datas_previous = x_datas_previous,
               y_datas_previous = y_datas_previous,
               res_indices = res_indices[onres_indices],
               fress_previous = fres_previous,
               titles_previous = titles_previous, titles = titles[onres_indices],
               ares = None, ares_title = 'power (dBm)')

    fres = fres.copy()
    qres = qres.copy()
    for fres_path, span_path, index in zip(fres_outpaths, span_outpaths, onres_indices):
        fres[index] = np.load(fres_path)
        span = np.load(span_path, allow_pickle = True)
        if None not in span:
            qres[index] = fres[index] / abs(np.diff(span)[0])
        if delete_temp_data:
            os.remove(fres_path)
            os.remove(span_path)
    np.save(out_directory + 'fres.npy', fres)
    np.save(out_directory + 'qres.npy', qres)

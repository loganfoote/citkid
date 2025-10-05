import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from citkid.res.fitter import fit_nonlinear_iq_with_gain
from ipywidgets import IntSlider, Dropdown, VBox, HBox
from ipywidgets import Output, Label, ToggleButton, Button, FloatText, Layout
from IPython.display import display
from citkid.res.data_io import nonlinear_iq_names, nonlinear_iq_labels
from citkid.util import fix_path

class resSweepFitter:
    def __init__(self, get_res_data, resfit_param, nres, x_name, x_unit,
                 x_df_name, out_directory, fig_dpi = 80, start_data_ix = 0):
        """
        Interactive code for fitting resonances over a parameter sweep
            (temperature, blackbody power, microwave power, etc.).

        Parameters:
        get_res_data (func): function that takes parameter <data_ix (int)> and
            returns sweep data for a single resonator:
            res_ix (int): resonator index.
            x (array-like (N,)): values (numeric) are the sweep parameter. N is
                the number of sweep points.
            y (array-like (N,)): values (numeric) are the resonance fit
                parameter for each dataset, corresponding to the values in x.
                For example, one may set x to temperature and y to fr.
            ffs (array-like, (N, M)): values (array-like) are arrays of
                frequencies for the fine S21 sweeps. M is the length of the S21
                sweep.
            zfs (array-like, (N, M)): values (array-like) are arrays of
                complex S21 for the fine S21 sweeps.
            fgs (array-like, (N, K)): values (array-like) are arrays of
                frequencies for the gain S21 sweeps. K is the length of the S21
                sweep.
            zgs (array-like, (N, K)): values (array-like) are arrays of
                complex S21 for the gain S21 sweeps.
            fres_alls (array-like, (N, L)): values (array-like) are arrays of
                resonant frequencies consisting of all resonant frequencies in
                the array, for gain sweep removal. L is the number of resonances
                in the array.
            qres_alls (array-like, (N, L)): values (array-like) are arrays of
                quality-factor-like parameters for removing resonances from the
                gain sweep, corresponding to the resonant frequencies in
                fres_alls.
        resfit_param (str): resonator fit parameter name corresponding to the
            y data. Must be in ['fr', 'Qr', 'amp', 'phi', 'a', 'i0', 'q0',
                                'tau'].
        nres (int): number of resonators in the dataset (such that
            get_res_data(nres) errors).
        x_name (str): name of the x variable.
        x_unit (str): unit of the x variable.
        x_df_name (str): DataFrame column name for inserting the x data into
            the output csv files.
        out_directory (str): directory to save the output csv files for new
            IQ fits.
        fig_dpi (int): figure DPI. Use this to scale figures for your display.
        start_data_ix (int): data index from which the code starts.
        """
        # Type and rangechecks
        if not callable(get_res_data):
            raise ValueError('get_res_data must be a function')
        lbl_ix = np.where([name == resfit_param
                           for name in nonlinear_iq_names])[0]
        if not len(lbl_ix):
            raise ValueError(f'resfit_param must be in {nonlinear_iq_names}')
        if type(fig_dpi) != int or fig_dpi < 1 or fig_dpi > 500:
            raise ValueError('fig_dpi must be an integer in [1, 500]')
        if type(start_data_ix) != int:
            raise ValueError('start_data_ix must be an int')
        if (type(nres) != int) or (nres <= 0):
            raise ValueError('nres must be an int that is >= 1')
        if start_data_ix >= nres or start_data_ix < 0:
            raise ValueError('start_data_ix must be in range [0, nres)')
        if type(x_name) != str or len(x_name) > 25:
            raise ValueError('x_name must be a str with len < 25')
        if type(x_unit) != str or len(x_unit) > 10:
            raise ValueError('x_unit must be a str with len < 10')
        if type(x_df_name) != str or len(x_df_name) > 25:
            raise ValueError('x_df_name must be a str with len < 25')
        # Set attributes
        self.get_res_data = get_res_data
        self.y_name = resfit_param # name for text display
        self.y_label = nonlinear_iq_labels[lbl_ix[0]] # label for plots
        self.dpi = fig_dpi
        self.data_ix = start_data_ix
        self.nres = nres
        self.x_name, self.x_label, self.x_unit = x_name, x_name, x_unit
        self.x_df_name = x_df_name
        # initialize sweep length to 500 until data is loaded
        self.s21_data_lens = [500]
        self.out_directory = fix_path(out_directory)
        os.makedirs(self.out_directory, exist_ok = True)
        self.fig_fit = None
        
    def run_fitter(self):
        """
        Run the main fitting code by setting up and displaying widgets.
        """
        vbox = self.setup_widgets()
        display(vbox)

    def load_data_ix(self):
        """
        Loads data for the current data index and updates plots.
        """
        lbl = f"Dataset {self.data_ix + 1:d} / {self.nres:d}"
        self.data_ix_label.value = lbl
        res_ix, x, y, ffs, zfs, fgs, zgs,\
        fres_alls, qres_alls = self.get_res_data(self.data_ix)
        res_ix = int(res_ix) # just in case
        self.s21_data_lens = [len(ff) for ff in ffs]
        self.sweep_data_len = len(x)

        # Format data as arrays and sort
        self.res_ix = res_ix
        self.x, self.y = np.asarray(x), np.asarray(y)
        ix = np.argsort(x)
        x, y = x[ix], y[ix]
        self.ffs = [np.asarray(ffs[i]) for i in ix]
        self.zfs = [np.asarray(zfs[i]) for i in ix]
        self.fgs = [np.asarray(fgs[i]) for i in ix]
        self.zgs = [np.asarray(zgs[i]) for i in ix]
        self.fres_alls = [np.asarray(fres_alls[i]) for i in ix]
        self.qres_alls = [np.asarray(qres_alls[i]) for i in ix]
        self.bad_iq_flags = np.zeros(len(self.ffs), dtype = int)
        # # update dropdown options
        options = [("Select", None)]
        options += [(f"{xi:.0f} {self.x_unit}", i)
                    for i, xi in enumerate(self.x)]
        self.dataset_selector.options = options
        self.dataset_selector.value = None  # reset selection

        # reset sliders
        self.start_ix_slider.value = 0
        self.end_ix_slider.value = self.s21_data_lens[0]
        self.start_ix_slider.max = self.s21_data_lens[0]
        self.end_ix_slider.max = self.s21_data_lens[0]

        # draw initial y vs x plot
        self.draw_sweep_plot()
        with self.out_swp:
            self.out_swp.clear_output(wait = True)
            d = display(self.fig_sweep, display_id = 'fig_sweep')

        # clear fit plot
        with self.out_fit:
            self.out_fit.clear_output()

    def draw_sweep_plot(self):
        """
        Draws a sweep plot of y vs x, and plots of |S21| vs frequency for each
        dataset in the sweep.
        """
        # Set up plots
        if self.y_name == 'fr':
            ylbl = '$x$ (kHz / GHz)'
            y = (self.y - self.y[0]) / self.y[0] * 1e6
        elif self.y_name == 'Qr':
            ylbl = r'$Q_r$ / 1,000'
            y = self.y * 1e-3
        else:
            ylbl = self.y_label
            y = self.y
        self.fig_sweep, self.axs = plt.subplots(1, 2, figsize = (9, 4),
                                                layout = "tight",
                                                dpi = self.dpi)
        self.axs[0].set(ylabel = ylbl,
                        xlabel = f'{self.x_name} ({self.x_unit})')
        self.axs[0].plot([], [], 'sk', label = 'fit data')
        self.axs[0].plot([], [], '--k', label = 'flagged bad data')
        self.axs[0].legend(framealpha = 1)
        f0 = np.mean(self.ffs[0])
        self.axs[1].set(ylabel = r'$|S_{21}|$ offset (dB)',
                   xlabel = f'(f - {f0 / 1e6:.4f} MHz) (kHz)')
        # Plot data
        offset = 0
        self.xy_scatter = [None] * len(self.ffs)
        for ix, (xi, yi, f, z) in enumerate(zip(self.x, y,
                                                 self.ffs, self.zfs)):
            color = plt.cm.viridis(ix / (len(self.ffs) - 1))
            # Scatter or vline on ax 0 
            if self.bad_iq_flags[ix]:
                self.xy_scatter[ix] = self.axs[0].axvline(xi, linestyle = '--', 
                                                          color = color)
            else:
                self.xy_scatter[ix] = self.axs[0].scatter(xi, yi, 
                                            color = color, marker = 's')
            # Plot |S21| with offset on ax 1
            dB = 20 * np.log10(np.abs(z))
            dB += offset - min(dB)
            offset = max(dB)
            self.axs[1].plot((f - f0) / 1e3, dB, '.',
                                color = color)

    def update_sweep_plot(self):
        """
        Update the sweep plot with new data and redraw the figure.
        """
        # Update scatter data
        if self.y_name == 'fr':
            y = (self.y - self.y[0]) / self.y[0] * 1e6
        elif self.y_name == 'Qr':
            y = self.y * 1e-3
        else:
            y = self.y 
        for ix, (xi, yi, scatter) in enumerate(zip(self.x, y, self.xy_scatter)):
            color = plt.cm.viridis(ix / (len(self.ffs) - 1))
            scatter.remove()
            if not self.bad_iq_flags[ix]:
                self.xy_scatter[ix] = self.axs[0].scatter(xi, yi, 
                                            color = color, marker = 's')
            else:
                self.xy_scatter[ix] = self.axs[0].axvline(xi, linestyle = '--', 
                                                          color = color)
 
        # redraw figure
        with self.out_swp:
            self.out_swp.clear_output(wait = True)
            self.fig_sweep.canvas.draw_idle()
            display(self.fig_sweep, display_id = 'fig_sweep')

    def setup_widgets(self):
        """
        Sets up and initializes the UI.
        """
        npoints = self.s21_data_lens[0]
        # --- Widgets ---
        self.out_swp = Output(layout = Layout(width = 'auto')) # sweep plot out
        self.out_fit = Output(layout = Layout(width = 'auto')) # fit plot out

        self.dataset_selector = Dropdown(options=[("Select", None)],
                                         description = self.x_name,
                                         tooltip = 'Select sweep to modify',
                                         layout = Layout(width = 'auto'),
                                         style = {'description_width': 'auto'})
        tt = 'Select S21 sweep start index for fitting'
        self.start_ix_slider = IntSlider(description = "Start", min = 0,
                                         max = npoints - 1, step = 1,
                                         value = 0, continuous_update = False,
                                         tooltip = tt,
                                         layout = Layout(width = 'auto'),
                                         style = {'description_width': 'auto'})
        tt = 'Select S21 sweep end index for fitting'
        self.end_ix_slider = IntSlider(description = "End", min = 1,
                                       max = npoints, step = 1,
                                       value = npoints,
                                       continuous_update = False,
                                       tooltip = tt,
                                       layout = Layout(width = 'auto'),
                                       style = {'description_width': 'auto'})
        tt = 'value by which qres_all is multiplied '
        tt += 'before fitting the gain sweep'
        self.q_mult_text = FloatText(description = "qres multiplier",
                                     value = 1.0, step = 0.1,
                                     tooltip = tt,
                                     layout = Layout(width = '200px'),
                                     style = {'description_width': 'auto'})
        tt = 'Flag the current sweep data as bad'
        self.bad_iq_flag = ToggleButton(value = False,
                                        description = 'Bad data flag',
                                        button_style = '',
                                        tooltip = tt,
                                        layout = Layout(width = 'auto'),
                                        style = {'description_width': 'auto'})
        # single handler that updates color and reruns fit/plot
        self.bad_iq_flag.observe(self._bad_iq_flag_change, 'value')
        tt = 'Apply current configuration to all S21 datasets in the sweep'
        self.apply_all_btn = Button(description = 'Apply to all',
                                    button_style = '',
                                    tooltip = tt,
                                    style = {'description_width': 'auto'},
                                    layout = Layout(width = 'auto'))
        self.status = Label(value = "Idle", layout = Layout(width = 'auto'),
                            style = {'description_width': 'auto'})

        # connect widgets
        self.dataset_selector.observe(self._select_dataset, names = "value")
        self.start_ix_slider.observe(self._rerun_fit, names = "value")
        self.end_ix_slider.observe(self._rerun_fit,   names = "value")
        self.q_mult_text.observe(self._rerun_fit,     names = "value")
        self.apply_all_btn.on_click(self._rerun_all_fits)

        # Buttons and label
        self.btn_prev = Button(description = "Previous Resonator",
                               style = {'description_width': 'auto'},
                               layout = Layout(width = 'auto'))
        self.btn_next = Button(description = "Next Resonator",
                               style = {'description_width': 'auto'},
                               layout = Layout(width = 'auto'))
        lbl = f"Dataset {self.data_ix + 1:d} / {self.nres:d}"
        self.data_ix_label = Label(value = lbl)
        self.btn_prev.on_click(lambda b: _on_prev_clicked(b, rsf = self))
        self.btn_next.on_click(lambda b: _on_next_clicked(b, rsf = self))

        # initial load
        self.load_data_ix()

        # create top-level vbox
        vbox = VBox([
            HBox([self.btn_prev, self.btn_next, self.data_ix_label],
                 align_items = 'flex-start'),
            HBox([self.dataset_selector, self.bad_iq_flag, self.apply_all_btn],
                 align_items = 'flex-start'),
            VBox([self.start_ix_slider, self.end_ix_slider,
                  self.q_mult_text, self.status],
                  layout = Layout(width = '500px'),
                  align_items = 'flex-start') ,
            HBox([self.out_swp, self.out_fit],
                 align_items = 'flex-start')
        ], align_items = 'flex-start')
        return vbox

    def _rerun_all_fits(self, button):
        """
        Callback function to run fitter and update plots for all datasets in
        the sweep.

        change (dict or None): widget change event dictionary (ignored, but
            required for ipywidgets.observe). Defaults to None.
        """
        if not all([di == self.s21_data_lens[0] for di in self.s21_data_lens]):
            s = 'Cannot apply all unless all S21 sweeps are the same length!!!'
            self.status.value = s
            return
        for sweep_ix in range(self.sweep_data_len):
            self.dataset_selector.value = sweep_ix
            self.select_dataset()

    def _rerun_fit(self, change = None):
        """
        Callback function to run fitter and update plots.

        change (dict or None): widget change event dictionary (ignored, but
            required for ipywidgets.observe). Defaults to None.
        """
        sweep_ix = self.dataset_selector.value
        start_ix, end_ix = self.start_ix_slider.value, self.end_ix_slider.value
        q_mult = self.q_mult_text.value
        bypass_fit = self.bad_iq_flag.value

        self.status.value = "Fitting..."
        if sweep_ix is None:
            self.status.value = 'Idle'
            return
        if bypass_fit:
            out_row = pd.Series({key: np.nan for key in fitrow_keys})
            fit_fig = None
            status = 'Flagged bad data, idle...'
        else:
            # extract and sort data
            fg, zg = self.fgs[sweep_ix], self.zgs[sweep_ix]
            ff, zf = self.ffs[sweep_ix], self.zfs[sweep_ix]
            fres_all = self.fres_alls[sweep_ix]
            qres_all = self.qres_alls[sweep_ix]
            ix = np.argsort(fg)
            fg, zg = fg[ix], zg[ix]
            ix = np.argsort(ff)
            ff, zf = ff[ix], zf[ix]
            # Apply ix cut
            ff_cut, zf_cut = ff[start_ix:end_ix], zf[start_ix:end_ix]

            # Run fitter
            try:
                out_row, fit_fig = fit_nonlinear_iq_with_gain(
                    fg, zg, ff_cut, zf_cut,
                    frs = fres_all, Qrs = qres_all * q_mult,
                    plotq = True, return_dataframe = True
                )
                fit_fig.set_dpi(self.dpi)
                status = 'Fit successful, idle...'
            except Exception as e:
                out_row = pd.Series({key: np.nan for key in fitrow_keys})
                fit_fig = None
                status = 'Fit failed, idle...'

        # Save fit output as CSV
        self.bad_iq_flags[self.data_ix] = int(self.bad_iq_flag.value)
        data_out = pd.DataFrame(out_row).T
        data_out['resIndex'] = self.res_ix
        data_out[self.x_df_name] = self.x[self.data_ix]
        data_out['resSweepFitterIndex'] = sweep_ix
        data_out['badS21dataFlag'] = int(self.bad_iq_flag.value)
        data_out['iqFitStartIx'] = start_ix
        data_out['iqFitEndIx'] = end_ix
        data_out['iqFitQresMult'] = q_mult
        path = os.path.join(os.path.join(self.out_directory,
                            f'fitdata_SI{sweep_ix:d}_Fn{self.res_ix:d}.csv'))
        data_out.to_csv(path, index = False)

        # Redraw sweep plot with updated row
        self.y[self.data_ix] = out_row[f'iq_{self.y_name}']
        # Update and display plots
        self.update_sweep_plot()

        with self.out_fit:
            self.out_fit.clear_output(wait=True)
            d = display(fit_fig)
        self.status.value = status

    def _select_dataset(self, change = None):
        """
        Callback function to select the next dataset.

        change (dict or None): widget change event dictionary (ignored, but
            required for ipywidgets.observe). Defaults to None.
        """
        sweep_ix = self.dataset_selector.value
        data_len = self.s21_data_lens[sweep_ix]
        self.start_ix_slider.max = data_len
        self.end_ix_slider.max = data_len
        self.start_ix_slider.value = 0
        self.end_ix_slider.value = data_len
        self._rerun_fit()

    def _bad_iq_flag_change(self, change = None):
        """
        Helper function for bad_iq_flag toggle button change.

        Parameters:
        change (dict or None): widget change event dictionary (ignored, but
            required for ipywidgets.observe). Defaults to None.
        """
        _update_bad_iq_color(change, self.bad_iq_flag)
        self._rerun_fit(change)

################################################################################
############################## Callback functions ##############################
################################################################################
def _on_prev_clicked(b, rsf):
    """
    Callback for the 'Previous' button.

    Parameters:
    b (ipywidgets.Button): the button widget that triggered the callback.
    rsf (resSweepFitter): resonator sweep fitter class instance.
    """
    rsf.data_ix = max(0, rsf.data_ix - 1)
    if rsf.fig_fit is not None:
        plt.close(rsf.fig_fit)
    plt.close(rsf.fig_sweep)
    rsf.load_data_ix()

def _on_next_clicked(b, rsf):
    """
    Callback for the 'Next' button.

    Parameters:
    b (ipywidgets.Button): the button widget that triggered the callback.
    rsf (resSweepFitter): resonator sweep fitter class instance.
    """
    rsf.data_ix = min(rsf.nres - 1, rsf.data_ix + 1)
    if rsf.fig_fit is not None:
        plt.close(rsf.fig_fit)
    plt.close(rsf.fig_sweep)
    rsf.load_data_ix()

def _update_bad_iq_color(change, toggle):
    """
    Callback function for changing the toggle button color.

    Parameters:
    change (dict): widget change event dictionary.
    toggle (ipywidgets.ToggleButton): toggle button to change.
    """
    if change['new']:
        toggle.button_style = 'danger'
    else:
        toggle.button_style = ''

################################################################################
################################### Variables ##################################
################################################################################
fitrow_keys = ['iq_fr_guess', 'iq_Qr_guess', 'iq_amp_guess', 'iq_phi_guess',
       'iq_a_guess', 'iq_i0_guess', 'iq_q0_guess', 'iq_tau_guess', 'iq_fr',
       'iq_Qr', 'iq_amp', 'iq_phi', 'iq_a', 'iq_i0', 'iq_q0', 'iq_tau',
       'iq_Qc', 'iq_Qi', 'iq_fr_err', 'iq_Qr_err', 'iq_amp_err', 'iq_phi_err',
       'iq_a_err', 'iq_i0_err', 'iq_q0_err', 'iq_tau_err', 'iq_pamp_00',
       'iq_pamp_01', 'iq_pamp_02', 'iq_pphase_00', 'iq_pphase_01',
       'iq_sweep_direction', 'iq_res', 'iq_plotpath']

import matplotlib.pyplot as plt
import os
from io import BytesIO
import warnings
import itertools
import numpy as np
import time
import threading
from tqdm.auto import tqdm

def save_fig(fig, filename, plot_directory, ftype = 'png',
             tight_layout = False, close_fig = True):
    """
    Save a matplotlib figure to disk with standard settings.

    Parameters:
    fig (plt.figure): Figure to save.
    filename (str): File name without extension.
    plot_directory (str): Directory to save the file.
    ftype (str): File type to save (e.g., 'png', 'eps').
    tight_layout (bool): If True, sets the figure layout to 'tight'.
    close_fig (bool): If True, closes the figure after saving.

    Returns:
    None
    """
    # Input validation
    if not isinstance(filename, str):
        raise TypeError('filename must be a string')
    if not isinstance(plot_directory, str):
        raise TypeError('plot_directory must be a string')
    if not isinstance(ftype, str):
        raise TypeError('ftype must be a string')
    if not isinstance(tight_layout, bool):
        raise TypeError('tight_layout must be a boolean')
    if not isinstance(close_fig, bool):
        raise TypeError('close_fig must be a boolean')
    
    # Save figure 
    if fig is None:
        return 
    if plot_directory:
        plot_directory = os.path.normpath(
            os.path.expanduser(plot_directory)
        )
        output_path = os.path.join(plot_directory, f'{filename}.{ftype}')
    else:
        output_path = f'{filename}.{ftype}'
    fig.set_facecolor('white')
    if tight_layout:
        fig.tight_layout()
    plt.figure(fig.number)
    try:
        plt.savefig(output_path, bbox_inches='tight', pad_inches = 0.05)
    except Exception as e:
        plt.savefig(output_path, pad_inches = 0.05)
    if close_fig:
        plt.close(fig)

def save_figure_to_memory(fig):
    """
    Save a matplotlib figure to a memory buffer.

    Parameters:
    fig (pyplot.figure): Figure to save.

    Returns:
    buf (BytesIO): Memory buffer containing the saved figure.
    """
    # Input validation
    if fig is None:
        raise ValueError('fig cannot be None')
    if not hasattr(fig, 'savefig'):
        raise TypeError('fig must be a matplotlib figure object')
    
    buf = BytesIO()
    fig.set_facecolor('white')
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        fig.tight_layout()
    fig.savefig(buf, format='png', bbox_inches = 'tight', pad_inches = 0.05)
    buf.seek(0)
    return buf

def combine_figures_vertically(fig1, fig2, dpi = 200):
    """
    Combine two matplotlib figures vertically into a single figure.

    Parameters:
    fig1 (pyplot.figure): First figure to combine.
    fig2 (pyplot.figure): Second figure to combine.
    dpi (int): Dots per inch for the combined figure.

    Returns:
    fig (pyplot.figure): Combined figure.
    """
    # Input validation
    if not isinstance(dpi, (int, np.integer)):
        raise TypeError('dpi must be an integer')
    if dpi <= 0:
        raise ValueError('dpi must be positive')
    
    with save_figure_to_memory(fig1) as buf1, \
        save_figure_to_memory(fig2) as buf2:
        plt.close(fig1)
        plt.close(fig2)
        fig, axs = plt.subplots(2, 1, dpi = dpi, layout = 'tight')
        for ax in axs:
            ax.set_axis_off()
        axs[0].imshow(plt.imread(buf1))
        axs[1].imshow(plt.imread(buf2))
        fig.tight_layout()
    return fig

def combine_figures_horizontally(fig1, fig2, dpi = 200):
    """
    Combine two matplotlib figures horizontally into a single figure.

    Parameters:
    fig1 (pyplot.figure): First figure to combine.
    fig2 (pyplot.figure): Second figure to combine.
    dpi (int): Dots per inch for the combined figure.

    Returns:
    fig (pyplot.figure): Combined figure.
    """
    # Input validation
    if not isinstance(dpi, (int, np.integer)):
        raise TypeError('dpi must be an integer')
    if dpi <= 0:
        raise ValueError('dpi must be positive')
    
    with save_figure_to_memory(fig1) as buf1, \
        save_figure_to_memory(fig2) as buf2:
        plt.close(fig1)
        plt.close(fig2)
        fig, axs = plt.subplots(1, 2, dpi = dpi, layout = 'tight')
        for ax in axs:
            ax.set_axis_off()
        axs[0].imshow(plt.imread(buf1))
        axs[1].imshow(plt.imread(buf2))
        fig.tight_layout()
    return fig

def to_scientific_notation(number):
    """
    Converts a number to scientific notation and returns the value and exponent.

    Parameters:
    number (float): Number to convert.

    Returns:
    mantissa (float): Mantissa in scientific notation.
    exponent (int): Exponent in scientific notation.
    """
    # Input validation
    if not isinstance(number, (int, float, np.integer, np.floating)):
        raise TypeError('number must be a numeric type')
    
    if number == 0:
        return (0.0, 0)  # Handle the special case where the number is zero

    exponent = int(np.floor(np.log10(abs(number))))
    mantissa = number / (10 ** exponent)

    # Ensure the mantissa is in the range [1, 10)
    if not (1 <= abs(mantissa) < 10):
        mantissa *= 10
        exponent -= 1

    return (mantissa, exponent)

def format_str_scientific_with_err(p, perr, for_plotting = True):
    r"""
    Formats a value and its uncertainty as a string in scientific notation,
    where the values are rounded to the appropriate number of significant
    figures. e.g. (1.54 ± 0.04) X 10^-4

    Parameters:
    p (float): Parameter value.
    perr (float): Parameter uncertainty.
    for_plotting (bool): If True, formats the string in LaTeX for plotting.
        If False, formats the string for plain text.

    Returns:
    formatted_str (str): Formatted string.
    """
    # Input validation
    if not isinstance(p, (int, float, np.integer, np.floating)):
        raise TypeError('p must be a numeric type')
    if not isinstance(perr, (int, float, np.integer, np.floating)):
        raise TypeError('perr must be a numeric type')
    if not isinstance(for_plotting, bool):
        raise TypeError('for_plotting must be a boolean')
    
    p_mantissa, p_exponent = to_scientific_notation(p)
    perr_mantissa, perr_exponent = to_scientific_notation(perr)
    exp_diff = p_exponent - perr_exponent
    perr_mantissa /= 10 ** (exp_diff)
    perr_mantissa = round(perr_mantissa, exp_diff)
    p_mantissa = round(p_mantissa, exp_diff)
    if for_plotting:
        formatted_str = f"$({p_mantissa} ± {perr_mantissa})  "
        formatted_str += r"\times10^{" + f"{p_exponent}" + r"}$"
    else:
        formatted_str = f"({p_mantissa} ± {perr_mantissa}) X 10^{p_exponent}"
    return formatted_str

def get_fit_bound_curves(x, popt, perr, model):
    """
    Gets the best fit model and upper/lower bound curves given
    optimal fit parameters and uncertainties

    Parameters:
    x (array-like): Sample x data.
    popt (array-like): Fit parameters.
    perr (array-like): Fit parameter uncertainties.
    model (callable): Model function that takes parameters (x, *popt).

    Returns:
    y_best_fit (np.array): Best-fit data corresponding to x.
    y_lower (np.array): Lower bound curve corresponding to x.
    y_upper (np.array): Upper bound curve corresponding to x.
    """
    # Input validation
    if not callable(model):
        raise TypeError('model must be callable')
    popt = np.asarray(popt)
    perr = np.asarray(perr)
    x = np.asarray(x)
    if popt.shape != perr.shape:
        raise ValueError('popt and perr must have the same shape')
    
    y_best_fit = model(x, *popt)

    param_combinations = list(itertools.product(*zip(popt - np.array(perr),
                                                  popt, popt + np.array(perr))))
    y_combinations = [model(x, *params) for params in param_combinations]
    y_combinations = [yi for yi in y_combinations if not any(np.isnan(yi))]
    y_upper = np.nanmax(y_combinations, axis=0)
    y_lower = np.nanmin(y_combinations, axis=0)
    return y_best_fit, y_lower, y_upper

def run_with_time_bar(fn, duration_s, desc, *args, **kwargs):
    """
    Runs a function while displaying a time progress bar for the specified 
    duration.

    Parameters:
    fn (callable): Function to run.
    duration_s (float): Duration in seconds for the progress bar.
    desc (str): Description for the progress bar.

    *args: Positional arguments to pass to fn.
    **kwargs: Keyword arguments to pass to fn.
    
    Returns:
    The return value of fn.
        """
    # Input validation
    if not callable(fn):
        raise TypeError('fn must be callable')
    if not isinstance(duration_s, (int, float, np.integer, np.floating)):
        raise TypeError('duration_s must be a numeric type')
    if duration_s < 0:
        raise ValueError('duration_s must be non-negative')
    if not isinstance(desc, str):
        raise TypeError('desc must be a string')
    
    stop = threading.Event()

    def progress():
        start = time.monotonic()
        with tqdm(
            total = duration_s, 
            unit = "s", 
            desc = desc, 
            bar_format = "{l_bar}{bar}| {remaining}",
            leave = False
            ) as pbar:
            while not stop.is_set():
                elapsed = time.monotonic() - start
                pbar.n = min(elapsed, duration_s)
                pbar.refresh()
                time.sleep(0.1)

            pbar.n = duration_s 
            pbar.refresh()

    t = threading.Thread(target=progress, daemon=True)
    t.start()

    try:
        return fn(*args, **kwargs)
    finally:
        stop.set()
        t.join()
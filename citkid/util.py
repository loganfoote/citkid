import matplotlib.pyplot as plt
from io import BytesIO
import warnings
import itertools
import numpy as np

def fix_path(path):
    r"""
    Given a path, convert \ to / and ensure the path ends in / if it is a folder

    Parameters:
    path (str): path to a folder or file

    Returns:
    fixed_path (str): fixed path
    """
    if path == '':
        return path
    fixed_path = path.replace('\\', '/')
    if fixed_path[-1] != '/' and ('.' not in fixed_path.split('/')[-1]):
        fixed_path += '/'
    return fixed_path

def save_fig(fig, filename, plot_directory, ftype = 'png',
             tight_layout = False, close_fig = True):
    """
    Saves a pyplot figure with standard settings

    Parameters:
    fig (plt.figure): figure to save
    filename (str): name of the file to save, without extension
    plot_directory (str): directory to save the file
    ftype (str): file type to save. Common types are 'png' and 'eps'
    tight_layout (bool): if True, sets the figure layout to 'tight'
    """
    if fig is not None:
        plot_directory = fix_path(plot_directory)
        fig.set_facecolor('white')
        if tight_layout:
            fig.tight_layout()
        plt.figure(fig.number)
        try:
            plt.savefig(plot_directory + filename + '.' + ftype,
                        bbox_inches='tight', pad_inches = 0.05)
        except Exception as e:
            plt.savefig(plot_directory + filename + '.' + ftype,
                        pad_inches = 0.05)
        if close_fig:
            plt.close(fig)

def save_figure_to_memory(fig):
    """
    Saves a matplotlib figure to memory. Use this to easily stitch together
    multiple figures without saving extra files

    Parameters:
    fig (pyplot.figure): figure to save

    Returns:
    buf (BytesIO): memory buffer of saved figure
    """
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
    Combine two matplotlib figures vertically for saving as a single file

    Parameters:
    fig1, fig2 (pyplot.figure): figures to combine

    Returns:
    fig (pyplot.figure): combined figure
    """
    with save_figure_to_memory(fig1) as buf1, save_figure_to_memory(fig2) as buf2:
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
    Combine two matplotlib figures horizontally for saving as a single file

    Parameters:
    fig1, fig2 (pyplot.figure): figures to combine

    Returns:
    fig (pyplot.figure): combined figure
    """
    with save_figure_to_memory(fig1) as buf1, save_figure_to_memory(fig2) as buf2:
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
    number (float): The number to convert.

    Returns:
    tuple: A tuple containing the value in scientific notation and the exponent.
    """
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
    p (float): parameter value
    perr (float) parameter uncertainty
    for_plotting (bool): if True, formats the string in latex for plotting. If
        False, formats the string for printing

    Returns:
    formatted_str (str): formatted string
    """
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
    x (array-like): x sample data
    popt (array-like): fit parameters
    perr (array-like): fit parameter uncertainties
    model (func): model function that takes parameters (x, *popt)

    Returns:
    y_best_fit (np.array): best fit data corresponding to x
    y_lower (np.array): lower bound on fit data corresponding to x
    y_upper (np.array): upper bound on fit data corresponding to x
    """
    y_best_fit = model(x, *popt)

    param_combinations = list(itertools.product(*zip(popt - np.array(perr),
                                                  popt, popt + np.array(perr))))
    y_combinations = [model(x, *params) for params in param_combinations]
    y_combinations = [yi for yi in y_combinations if not any(np.isnan(yi))]
    y_upper = np.nanmax(y_combinations, axis=0)
    y_lower = np.nanmin(y_combinations, axis=0)
    return y_best_fit, y_lower, y_upper

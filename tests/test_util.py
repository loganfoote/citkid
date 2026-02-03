import pytest
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for testing
import matplotlib.pyplot as plt
import os
import tempfile
from io import BytesIO
from citkid import util


################################################################################
################################### save_fig ###################################
################################################################################
def test_save_fig_saves_to_file():
    with tempfile.TemporaryDirectory() as tmpdir:
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])
        filename = 'test_fig'
        
        util.save_fig(fig, filename, tmpdir, ftype = 'png')
        
        expected_path = os.path.join(tmpdir, f'{filename}.png')
        assert os.path.exists(expected_path)


def test_save_fig_with_different_ftypes():
    with tempfile.TemporaryDirectory() as tmpdir:
        for ftype in ['png', 'pdf', 'eps']:
            fig, ax = plt.subplots()
            ax.plot([1, 2, 3], [1, 2, 3])
            filename = f'test_fig_{ftype}'
            
            util.save_fig(fig, filename, tmpdir, ftype = ftype)
            
            expected_path = os.path.join(tmpdir, f'{filename}.{ftype}')
            assert os.path.exists(expected_path)


def test_save_fig_with_empty_plot_directory():
    # Use actual temp file rather than changing directories
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 2, 3])
    filename = 'test_fig_empty_dir'
    
    try:
        util.save_fig(fig, filename, '', ftype = 'png')
        
        expected_path = f'{filename}.png'
        assert os.path.exists(expected_path)
    finally:
        if os.path.exists(expected_path):
            os.remove(expected_path)


def test_save_fig_expands_tilde():
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 2, 3])
    filename = 'test_fig'
    plot_dir = '~/test_citkid_dir'
    expanded_dir = os.path.expanduser(plot_dir)
    
    try:
        os.makedirs(expanded_dir, exist_ok = True)
        util.save_fig(fig, filename, plot_dir, ftype = 'png')
        
        expected_path = os.path.join(expanded_dir, f'{filename}.png')
        assert os.path.exists(expected_path)
    finally:
        if os.path.exists(expected_path):
            os.remove(expected_path)
        if os.path.exists(expanded_dir):
            os.rmdir(expanded_dir)


def test_save_fig_closes_figure_when_close_fig_true():
    with tempfile.TemporaryDirectory() as tmpdir:
        fig, ax = plt.subplots()
        fig_num = fig.number
        ax.plot([1, 2, 3], [1, 2, 3])
        
        util.save_fig(fig, 'test', tmpdir, close_fig = True)
        
        assert fig_num not in plt.get_fignums()


def test_save_fig_keeps_figure_when_close_fig_false():
    with tempfile.TemporaryDirectory() as tmpdir:
        fig, ax = plt.subplots()
        fig_num = fig.number
        ax.plot([1, 2, 3], [1, 2, 3])
        
        util.save_fig(fig, 'test', tmpdir, close_fig = False)
        
        assert fig_num in plt.get_fignums()
        plt.close(fig)


def test_save_fig_with_tight_layout():
    with tempfile.TemporaryDirectory() as tmpdir:
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])
        
        util.save_fig(fig, 'test', tmpdir, tight_layout = True)
        
        expected_path = os.path.join(tmpdir, 'test.png')
        assert os.path.exists(expected_path)


def test_save_fig_with_none_fig():
    with tempfile.TemporaryDirectory() as tmpdir:
        util.save_fig(None, 'test', tmpdir)
        
        expected_path = os.path.join(tmpdir, 'test.png')
        assert not os.path.exists(expected_path)


def test_save_fig_invalid_filename_type():
    with tempfile.TemporaryDirectory() as tmpdir:
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])
        
        with pytest.raises(TypeError):
            util.save_fig(fig, 123, tmpdir)
        
        plt.close(fig)


def test_save_fig_invalid_plot_directory_type():
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 2, 3])
    
    with pytest.raises(TypeError):
        util.save_fig(fig, 'test', 123)
    
    plt.close(fig)


def test_save_fig_invalid_ftype_type():
    with tempfile.TemporaryDirectory() as tmpdir:
        fig, ax = plt.subplots()
        ax.plot([1, 2, 3], [1, 2, 3])
        
        with pytest.raises(TypeError):
            util.save_fig(fig, 'test', tmpdir, ftype = 123)
        
        plt.close(fig)


################################################################################
########################### save_figure_to_memory ##############################
################################################################################
def test_save_figure_to_memory_returns_bytesio():
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 2, 3])
    
    buf = util.save_figure_to_memory(fig)
    
    assert isinstance(buf, BytesIO)
    assert buf.tell() == 0  # Buffer should be at start
    plt.close(fig)


def test_save_figure_to_memory_buffer_contains_png_data():
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 2, 3])
    
    buf = util.save_figure_to_memory(fig)
    
    # PNG files start with signature bytes
    png_signature = b'\x89PNG\r\n\x1a\n'
    data = buf.read(8)
    assert data == png_signature
    plt.close(fig)


def test_save_figure_to_memory_buffer_is_readable():
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 2, 3])
    
    buf = util.save_figure_to_memory(fig)
    
    # Should be able to read image from buffer
    img = plt.imread(buf)
    assert img is not None
    assert img.ndim == 3  # RGB image
    plt.close(fig)


def test_save_figure_to_memory_invalid_fig_none():
    with pytest.raises(ValueError):
        util.save_figure_to_memory(None)


def test_save_figure_to_memory_invalid_fig_type():
    with pytest.raises(TypeError):
        util.save_figure_to_memory("not a figure")


def test_save_figure_to_memory_invalid_fig_type_int():
    with pytest.raises(TypeError):
        util.save_figure_to_memory(123)


################################################################################
######################## combine_figures_vertically ############################
################################################################################
def test_combine_figures_vertically_returns_figure():
    fig1, ax1 = plt.subplots()
    ax1.plot([1, 2, 3], [1, 2, 3])
    fig2, ax2 = plt.subplots()
    ax2.plot([1, 2, 3], [3, 2, 1])
    
    combined = util.combine_figures_vertically(fig1, fig2)
    
    assert isinstance(combined, plt.Figure)
    plt.close(combined)


def test_combine_figures_vertically_has_two_rows():
    fig1, ax1 = plt.subplots()
    ax1.plot([1, 2, 3], [1, 2, 3])
    fig2, ax2 = plt.subplots()
    ax2.plot([1, 2, 3], [3, 2, 1])
    
    combined = util.combine_figures_vertically(fig1, fig2)
    
    # Should have 2 subplots arranged vertically
    assert len(combined.axes) == 2
    plt.close(combined)


def test_combine_figures_vertically_creates_new_figure():
    fig1, ax1 = plt.subplots()
    ax1.plot([1, 2, 3], [1, 2, 3])
    fig2, ax2 = plt.subplots()
    ax2.plot([1, 2, 3], [3, 2, 1])
    
    combined = util.combine_figures_vertically(fig1, fig2)
    
    # Combined figure should exist
    assert combined.number in plt.get_fignums()
    plt.close(combined)
    plt.close('all')  # Clean up any remaining figures


def test_combine_figures_vertically_with_custom_dpi():
    fig1, ax1 = plt.subplots()
    ax1.plot([1, 2, 3], [1, 2, 3])
    fig2, ax2 = plt.subplots()
    ax2.plot([1, 2, 3], [3, 2, 1])
    
    combined = util.combine_figures_vertically(fig1, fig2, dpi = 300)
    
    assert combined.dpi == 300
    plt.close(combined)


################################################################################
####################### combine_figures_horizontally ###########################
################################################################################
def test_combine_figures_horizontally_returns_figure():
    fig1, ax1 = plt.subplots()
    ax1.plot([1, 2, 3], [1, 2, 3])
    fig2, ax2 = plt.subplots()
    ax2.plot([1, 2, 3], [3, 2, 1])
    
    combined = util.combine_figures_horizontally(fig1, fig2)
    
    assert isinstance(combined, plt.Figure)
    plt.close(combined)


def test_combine_figures_horizontally_has_two_columns():
    fig1, ax1 = plt.subplots()
    ax1.plot([1, 2, 3], [1, 2, 3])
    fig2, ax2 = plt.subplots()
    ax2.plot([1, 2, 3], [3, 2, 1])
    
    combined = util.combine_figures_horizontally(fig1, fig2)
    
    # Should have 2 subplots arranged horizontally
    assert len(combined.axes) == 2
    plt.close(combined)


def test_combine_figures_horizontally_creates_new_figure():
    fig1, ax1 = plt.subplots()
    ax1.plot([1, 2, 3], [1, 2, 3])
    fig2, ax2 = plt.subplots()
    ax2.plot([1, 2, 3], [3, 2, 1])
    
    combined = util.combine_figures_horizontally(fig1, fig2)
    
    # Combined figure should exist
    assert combined.number in plt.get_fignums()
    plt.close(combined)
    plt.close('all')  # Clean up any remaining figures


def test_combine_figures_horizontally_with_custom_dpi():
    fig1, ax1 = plt.subplots()
    ax1.plot([1, 2, 3], [1, 2, 3])
    fig2, ax2 = plt.subplots()
    ax2.plot([1, 2, 3], [3, 2, 1])
    
    combined = util.combine_figures_horizontally(fig1, fig2, dpi = 150)
    
    assert combined.dpi == 150
    plt.close(combined)


def test_combine_figures_vertically_invalid_dpi_type():
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    
    with pytest.raises(TypeError):
        util.combine_figures_vertically(fig1, fig2, dpi = "not an int")
    
    plt.close('all')


def test_combine_figures_vertically_invalid_dpi_negative():
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    
    with pytest.raises(ValueError):
        util.combine_figures_vertically(fig1, fig2, dpi = 0)
    
    plt.close('all')


def test_combine_figures_horizontally_invalid_dpi_type():
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    
    with pytest.raises(TypeError):
        util.combine_figures_horizontally(fig1, fig2, dpi = 3.14)
    
    plt.close('all')


def test_combine_figures_horizontally_invalid_dpi_negative():
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    
    with pytest.raises(ValueError):
        util.combine_figures_horizontally(fig1, fig2, dpi = -100)
    
    plt.close('all')


################################################################################
######################### to_scientific_notation ###############################
################################################################################
def test_to_scientific_notation_zero():
    mantissa, exponent = util.to_scientific_notation(0)
    assert mantissa == 0.0
    assert exponent == 0


def test_to_scientific_notation_positive_numbers():
    test_cases = [
        (1.0, (1.0, 0)),
        (10.0, (1.0, 1)),
        (100.0, (1.0, 2)),
        (1.54e-4, (1.54, -4)),
        (3.14159, (3.14159, 0)),
        (0.001, (1.0, -3)),
    ]
    for number, (exp_mantissa, exp_exponent) in test_cases:
        mantissa, exponent = util.to_scientific_notation(number)
        assert np.isclose(mantissa, exp_mantissa, rtol = 1e-10)
        assert exponent == exp_exponent


def test_to_scientific_notation_negative_numbers():
    test_cases = [
        (-1.0, (-1.0, 0)),
        (-10.0, (-1.0, 1)),
        (-0.001, (-1.0, -3)),
        (-1.54e-4, (-1.54, -4)),
    ]
    for number, (exp_mantissa, exp_exponent) in test_cases:
        mantissa, exponent = util.to_scientific_notation(number)
        assert np.isclose(mantissa, exp_mantissa, rtol = 1e-10)
        assert exponent == exp_exponent


def test_to_scientific_notation_mantissa_range():
    # Mantissa should be in range [1, 10)
    test_numbers = [1e-10, 1e-5, 1e-1, 1, 10, 100, 1e5, 1e10]
    for number in test_numbers:
        mantissa, exponent = util.to_scientific_notation(number)
        assert 1 <= abs(mantissa) < 10


def test_to_scientific_notation_invalid_type():
    with pytest.raises(TypeError):
        util.to_scientific_notation("not a number")


def test_to_scientific_notation_invalid_type_none():
    with pytest.raises(TypeError):
        util.to_scientific_notation(None)


################################################################################
##################### format_str_scientific_with_err ###########################
################################################################################
def test_format_str_scientific_with_err_for_plotting():
    result = util.format_str_scientific_with_err(
        1.54e-4, 0.04e-4, for_plotting = True
    )
    assert '$' in result
    assert r'\times10^{' in result
    assert '±' in result


def test_format_str_scientific_with_err_for_text():
    result = util.format_str_scientific_with_err(
        1.54e-4, 0.04e-4, for_plotting = False
    )
    assert '$' not in result
    assert '±' in result
    assert 'X 10^' in result


def test_format_str_scientific_with_err_various_magnitudes():
    test_cases = [
        (1.0, 0.1),
        (100.0, 5.0),
        (0.001, 0.0001),
        (1e6, 1e4),
    ]
    for p, perr in test_cases:
        result = util.format_str_scientific_with_err(p, perr)
        assert isinstance(result, str)
        assert len(result) > 0


def test_format_str_scientific_with_err_invalid_p_type():
    with pytest.raises(TypeError):
        util.format_str_scientific_with_err("not a number", 0.1)


def test_format_str_scientific_with_err_invalid_perr_type():
    with pytest.raises(TypeError):
        util.format_str_scientific_with_err(1.0, "not a number")


def test_format_str_scientific_with_err_invalid_for_plotting_type():
    with pytest.raises(TypeError):
        util.format_str_scientific_with_err(1.0, 0.1, for_plotting = "yes")


################################################################################
########################### get_fit_bound_curves ###############################
################################################################################
def test_get_fit_bound_curves_linear_model():
    # Linear model: y = a*x + b
    def linear_model(x, a, b):
        return a * x + b
    
    x = np.linspace(0, 10, 50)
    popt = [2.0, 1.0]  # a = 2, b = 1
    perr = [0.1, 0.1]
    
    y_best, y_lower, y_upper = util.get_fit_bound_curves(
        x, popt, perr, linear_model
    )
    
    # Best fit should match model with optimal params
    assert np.allclose(y_best, linear_model(x, *popt))
    # Upper bound should be >= best fit
    assert np.all(y_upper >= y_best - 1e-10)
    # Lower bound should be <= best fit
    assert np.all(y_lower <= y_best + 1e-10)


def test_get_fit_bound_curves_returns_correct_shapes():
    def model(x, a):
        return a * x
    
    x = np.linspace(0, 10, 100)
    popt = [2.0]
    perr = [0.2]
    
    y_best, y_lower, y_upper = util.get_fit_bound_curves(
        x, popt, perr, model
    )
    
    assert y_best.shape == x.shape
    assert y_lower.shape == x.shape
    assert y_upper.shape == x.shape


def test_get_fit_bound_curves_quadratic_model():
    def quad_model(x, a, b, c):
        return a * x**2 + b * x + c
    
    x = np.linspace(-5, 5, 50)
    popt = [1.0, 0.0, 0.0]
    perr = [0.1, 0.1, 0.1]
    
    y_best, y_lower, y_upper = util.get_fit_bound_curves(
        x, popt, perr, quad_model
    )
    
    assert isinstance(y_best, np.ndarray)
    assert isinstance(y_lower, np.ndarray)
    assert isinstance(y_upper, np.ndarray)


def test_get_fit_bound_curves_invalid_model_not_callable():
    with pytest.raises(TypeError):
        util.get_fit_bound_curves([1, 2, 3], [1.0], [0.1], "not callable")


def test_get_fit_bound_curves_invalid_popt_perr_shape_mismatch():
    def model(x, a, b):
        return a * x + b
    
    with pytest.raises(ValueError):
        util.get_fit_bound_curves(
            [1, 2, 3], [1.0, 2.0], [0.1], model
        )


################################################################################
############################ run_with_time_bar #################################
################################################################################
def test_run_with_time_bar_executes_function():
    def simple_func(x, y):
        return x + y
    
    result = util.run_with_time_bar(simple_func, 0.1, "Test", 2, 3)
    
    assert result == 5


def test_run_with_time_bar_with_kwargs():
    def func_with_kwargs(a, b = 10):
        return a * b
    
    result = util.run_with_time_bar(func_with_kwargs, 0.1, "Test", 5, b = 3)
    
    assert result == 15


def test_run_with_time_bar_propagates_exceptions():
    def failing_func():
        raise ValueError("Test error")
    
    with pytest.raises(ValueError, match = "Test error"):
        util.run_with_time_bar(failing_func, 0.1, "Test")


def test_run_with_time_bar_returns_none_for_none_function():
    def returns_none():
        pass
    
    result = util.run_with_time_bar(returns_none, 0.1, "Test")
    
    assert result is None


def test_run_with_time_bar_invalid_fn_not_callable():
    with pytest.raises(TypeError):
        util.run_with_time_bar("not callable", 1.0, "Test")


def test_run_with_time_bar_invalid_duration_type():
    def dummy():
        pass
    
    with pytest.raises(TypeError):
        util.run_with_time_bar(dummy, "not a number", "Test")


def test_run_with_time_bar_invalid_duration_negative():
    def dummy():
        pass
    
    with pytest.raises(ValueError):
        util.run_with_time_bar(dummy, -1.0, "Test")


def test_run_with_time_bar_invalid_desc_type():
    def dummy():
        pass
    
    with pytest.raises(TypeError):
        util.run_with_time_bar(dummy, 0.1, 123)

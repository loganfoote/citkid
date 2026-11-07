"""Tests for citkid.primecam.figures."""

from io import BytesIO

import matplotlib.pyplot as plt
import pytest

from citkid import util
from citkid.primecam import figures


def test_combine_figs_vert_returns_bytesio_and_png():
    fig1, ax1 = plt.subplots()
    ax1.plot([1, 2, 3], [1, 2, 3])
    fig2, ax2 = plt.subplots()
    ax2.plot([1, 2, 3], [3, 2, 1])

    buf = figures.combine_figs_vert([fig1, fig2])
    assert isinstance(buf, BytesIO)
    # PNG signature
    sig = buf.read(8)
    assert sig == b'\x89PNG\r\n\x1a\n'
    buf.seek(0)


def test_combine_figs_vert_accepts_mixed_inputs_and_preserves_buffers():
    fig1, ax1 = plt.subplots()
    ax1.plot([1, 2, 3], [1, 2, 3])
    fig2, ax2 = plt.subplots()
    ax2.plot([1, 2, 3], [3, 2, 1])

    buf1 = util.save_fig_to_memory(fig1)
    # pass a BytesIO and a Figure
    out = figures.combine_figs_vert([buf1, fig2])
    assert isinstance(out, BytesIO)
    # original buffer should still be readable by caller
    buf1.seek(0)
    assert buf1.read(8) == b'\x89PNG\r\n\x1a\n'
    buf1.seek(0)


def test_combine_figs_horz_returns_bytesio_and_png():
    fig1, ax1 = plt.subplots()
    ax1.plot([1, 2, 3], [1, 2, 3])
    fig2, ax2 = plt.subplots()
    ax2.plot([1, 2, 3], [3, 2, 1])

    buf = figures.combine_figs_horz([fig1, fig2])
    assert isinstance(buf, BytesIO)
    sig = buf.read(8)
    assert sig == b'\x89PNG\r\n\x1a\n'
    buf.seek(0)


def test_combine_figs_horz_accepts_mixed_inputs():
    fig1, ax1 = plt.subplots()
    ax1.plot([1, 2, 3], [1, 2, 3])
    fig2, ax2 = plt.subplots()
    ax2.plot([1, 2, 3], [3, 2, 1])

    buf2 = util.save_fig_to_memory(fig2)
    out = figures.combine_figs_horz([fig1, buf2])
    assert isinstance(out, BytesIO)
    buf2.seek(0)
    assert buf2.read(8) == b'\x89PNG\r\n\x1a\n'
    buf2.seek(0)


def test_combine_figures_vertically_stacks_two_figures():
    """
    The two-figure helper returns one figure with both images, one above
    the other, and closes the inputs.
    """
    fig1, ax1 = plt.subplots()
    ax1.plot([1, 2], [1, 2])
    fig2, ax2 = plt.subplots()
    ax2.plot([1, 2], [2, 1])

    fig = figures.combine_figures_vertically(fig1, fig2, dpi = 50)

    assert len(fig.axes) == 2
    # Figure numbers are reused, so compare the open figures themselves
    open_figs = [plt.figure(n) for n in plt.get_fignums()]
    assert fig1 not in open_figs and fig2 not in open_figs
    plt.close(fig)


def test_save_figure_to_memory_returns_png_buffer():
    """
    The buffer holds a PNG image and starts at position 0.
    """
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3])
    buf = figures.save_figure_to_memory(fig)
    assert isinstance(buf, BytesIO)
    assert buf.tell() == 0
    assert buf.read(8) == b'\x89PNG\r\n\x1a\n'
    plt.close(fig)


def test_combine_figures_vertically_invalid_dpi_type():
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    
    with pytest.raises(TypeError):
        figures.combine_figs_vert([fig1, fig2], dpi = "not an int")
    
    plt.close('all')


def test_combine_figures_vertically_invalid_dpi_negative():
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    
    with pytest.raises(ValueError):
        figures.combine_figs_vert([fig1, fig2], dpi = 0)
    
    plt.close('all')


def test_combine_figures_horizontally_invalid_dpi_type():
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    
    with pytest.raises(TypeError):
        figures.combine_figs_horz([fig1, fig2], dpi = 3.14)
    
    plt.close('all')


def test_combine_figures_horizontally_invalid_dpi_negative():
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    
    with pytest.raises(ValueError):
        figures.combine_figs_horz([fig1, fig2], dpi = -100)
    
    plt.close('all')

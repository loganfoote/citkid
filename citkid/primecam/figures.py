"""
Combine matplotlib figures into one image, for the Prime-Cam plots.

These helpers were in ``citkid.util``; primecam is their only user.
"""
from io import BytesIO
import warnings
import numpy as np
from ..util import save_fig_to_memory


def combine_figs_vert(figs, dpi = 200, rescale_to = 'max'):
    """
    Combine a list of matplotlib figures vertically into a single matplotlib
    figure.

    Parameters:
    figs (list of plt.Figure): Figures to combine (any length >= 1).
    dpi (int): Dots per inch for the combined figure.
    rescale_to (str): 'max' to scale images up to the max width, 'min' to scale 
        down to min width.

    Returns:
    plt.Figure: Combined figure.
    """
    import matplotlib.pyplot as plt
    from PIL import Image
    if not isinstance(dpi, (int, np.integer)):
        raise TypeError('dpi must be an integer')
    if dpi <= 0:
        raise ValueError('dpi must be positive')
    if not isinstance(figs, (list, tuple)):
        raise TypeError('figs must be a list or tuple of matplotlib figures')
    if len(figs) == 0:
        raise ValueError('figs must contain at least one figure')

    # If only one figure, return it unchanged
    if len(figs) == 1:
        return figs[0]

    bufs = []
    created_bufs = []
    try:
        for f in figs:
            if f is None:
                raise ValueError('figure in figs cannot be None')
            # Accept either a matplotlib Figure (has savefig) or a BytesIO with
            # PNG data
            if hasattr(f, 'savefig'):
                b = save_fig_to_memory(f)
                created_bufs.append(b)
                bufs.append(b)
                plt.close(f)
            elif isinstance(f, BytesIO):
                f.seek(0)
                bufs.append(f)
            else:
                raise TypeError(
                    'each item in figs must be a matplotlib Figure '
                    'or a BytesIO PNG buffer'
                    )

        images = [Image.open(b).convert('RGBA') for b in bufs]

        widths = [im.width for im in images]
        if rescale_to == 'max':
            target_w = max(widths)
        elif rescale_to == 'min':
            target_w = min(widths)
        else:
            raise ValueError("rescale_to must be 'max' or 'min'")

        resized = []
        for im in images:
            if im.width != target_w:
                new_h = int(im.height * (target_w / im.width))
                resized.append(im.resize((target_w, new_h), Image.LANCZOS))
            else:
                resized.append(im)

        total_h = sum(im.height for im in resized)
        combined = Image.new('RGBA', (target_w, total_h), (255, 255, 255, 255))
        y = 0
        for im in resized:
            combined.paste(im, (0, y), im)
            y += im.height

        # Save combined image to a PNG bytes buffer and return it
        out_buf = BytesIO()
        combined.save(out_buf, format='PNG')
        out_buf.seek(0)
        return out_buf
    finally:
        for im in locals().get('images', []):
            try:
                im.close()
            except Exception:
                pass
        for b in created_bufs:
            try:
                b.close()
            except Exception:
                pass


def combine_figs_horz(figs, dpi = 200, rescale_to = 'max'):
    """
    Combine a list of matplotlib figures horizontally into a single matplotlib 
    figure.

    Parameters:
    figs (list of plt.Figure): Figures to combine (any length >= 1).
    dpi (int): Dots per inch for the combined figure.
    rescale_to (str): 'max' to scale images up to the max height, 'min' to
        scale down to min height.

    Returns:
    plt.Figure: Combined figure.
    """
    import matplotlib.pyplot as plt
    from PIL import Image
    if not isinstance(dpi, (int, np.integer)):
        raise TypeError('dpi must be an integer')
    if dpi <= 0:
        raise ValueError('dpi must be positive')
    if not isinstance(figs, (list, tuple)):
        raise TypeError('figs must be a list or tuple of matplotlib figures')
    if len(figs) == 0:
        raise ValueError('figs must contain at least one figure')

    if len(figs) == 1:
        return figs[0]

    bufs = []
    created_bufs = []
    try:
        for f in figs:
            if f is None:
                raise ValueError('figure in figs cannot be None')
            if hasattr(f, 'savefig'):
                b = save_fig_to_memory(f)
                created_bufs.append(b)
                bufs.append(b)
                plt.close(f)
            elif isinstance(f, BytesIO):
                f.seek(0)
                bufs.append(f)
            else:
                raise TypeError(
                    'each item in figs must be a matplotlib Figure '
                    'or a BytesIO PNG buffer'
                    )

        images = [Image.open(b).convert('RGBA') for b in bufs]

        heights = [im.height for im in images]
        if rescale_to == 'max':
            target_h = max(heights)
        elif rescale_to == 'min':
            target_h = min(heights)
        else:
            raise ValueError("rescale_to must be 'max' or 'min'")

        resized = []
        for im in images:
            if im.height != target_h:
                new_w = int(im.width * (target_h / im.height))
                resized.append(im.resize((new_w, target_h), Image.LANCZOS))
            else:
                resized.append(im)

        total_w = sum(im.width for im in resized)
        combined = Image.new('RGBA', (total_w, target_h), (255, 255, 255, 255))
        x = 0
        for im in resized:
            combined.paste(im, (x, 0), im)
            x += im.width

        out_buf = BytesIO()
        combined.save(out_buf, format='PNG')
        out_buf.seek(0)
        return out_buf
    finally:
        for im in locals().get('images', []):
            try:
                im.close()
            except Exception:
                pass
        for b in created_bufs:
            try:
                b.close()
            except Exception:
                pass


def combine_figures_vertically(fig1, fig2, dpi = 200):
    """
    Combine two matplotlib figures vertically for saving as a single file

    Parameters:
    fig1, fig2 (pyplot.figure): figures to combine

    Returns:
    fig (pyplot.figure): combined figure
    """
    import matplotlib.pyplot as plt
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


def save_figure_to_memory(fig):
    """
    Save a matplotlib figure to memory. Use this to easily stitch together
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

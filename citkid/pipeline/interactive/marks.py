"""
Dialogs for the tone-marks registry (``daq_layout`` tone marks), shared
by every window that works tone by tone: mark the tone bad with a
reason, add a note (the tone is kept), or, in the tuning window, mark
every fit point bad and choose whether to keep the tone.

The dialogs return what the user chose; the windows write the registry
with ``daq_layout.add_tone_mark`` and keep their own local effect (NaN
outputs, a tone left out of an average, ...). Each window keeps the
dialog in an attribute (``_ask_reason``, ``_ask_note``, ...) so tests
can stand it in.
"""
from pyqtgraph.Qt import QtWidgets
from ...daq_layout import BAD_REASONS

def ask_reason(parent, title, reasons = BAD_REASONS, label = 'Reason:'):
    """
    Ask why a tone is bad: one of some reasons, or typed.

    Parameters:
    parent (QWidget or None): the dialog's parent.
    title (str): window title, e.g. 'Mark res 12 bad'.
    reasons (sequence of str): the choices offered. Default
        ``daq_layout.BAD_REASONS``.
    label (str): the prompt. Default 'Reason:'.

    Returns:
    reason (str or None): the reason, or None if cancelled.
    """
    dialog = QtWidgets.QDialog(parent)
    dialog.setWindowTitle(title)
    lay = QtWidgets.QVBoxLayout(dialog)
    lay.addWidget(QtWidgets.QLabel(label))
    box = QtWidgets.QComboBox()
    box.setEditable(True)
    box.addItems(list(reasons))
    lay.addWidget(box)
    box.setFocus()
    box.lineEdit().selectAll()
    buttons = QtWidgets.QDialogButtonBox(
        QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
    buttons.accepted.connect(dialog.accept)
    buttons.rejected.connect(dialog.reject)
    lay.addWidget(buttons)
    if dialog.exec_() != QtWidgets.QDialog.Accepted:
        return None
    return box.currentText().strip()

def ask_note(parent, title, text = ''):
    """
    Ask for a note about a tone (free text, several lines).

    Parameters:
    parent (QWidget or None): the dialog's parent.
    title (str): window title, e.g. 'Note on res 12'.
    text (str): text to start from. Default ''.

    Returns:
    note (str or None): the note, or None if cancelled or left blank.
    """
    dialog = QtWidgets.QDialog(parent)
    dialog.setWindowTitle(title)
    lay = QtWidgets.QVBoxLayout(dialog)
    lay.addWidget(QtWidgets.QLabel('Note (the tone is kept):'))
    edit = QtWidgets.QPlainTextEdit()
    edit.setPlainText(text)
    lay.addWidget(edit)
    buttons = QtWidgets.QDialogButtonBox(
        QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
    buttons.accepted.connect(dialog.accept)
    buttons.rejected.connect(dialog.reject)
    lay.addWidget(buttons)
    if dialog.exec_() != QtWidgets.QDialog.Accepted:
        return None
    note = edit.toPlainText().strip()
    return note or None

def ask_all_bad(parent, title, reasons = BAD_REASONS):
    """
    Ask, when every fit point of a tone is marked bad, whether to keep
    the tone (at the fallback value) or remove it from the tuned set,
    with a note or reason.

    Parameters:
    parent (QWidget or None): the dialog's parent.
    title (str): window title, e.g. 'Mark every point of res 12 bad'.
    reasons (sequence of str): reasons offered (any text can be typed).
        Default ``daq_layout.BAD_REASONS``.

    Returns:
    choice (str or None): 'keep' or 'remove', or None if cancelled.
    text (str): the note or reason ('' if none).
    """
    dialog = QtWidgets.QDialog(parent)
    dialog.setWindowTitle(title)
    lay = QtWidgets.QVBoxLayout(dialog)
    lay.addWidget(QtWidgets.QLabel(
        'Every fit point of this tone is marked bad, so it gets the '
        'fallback value (the median of the other tones).\n'
        'Keep the tone in the tuned set, or remove it?'))
    keep = QtWidgets.QRadioButton('Keep the tone (a note is saved)')
    remove = QtWidgets.QRadioButton(
        'Remove the tone from the tuned set (marked bad)')
    keep.setChecked(True)
    lay.addWidget(keep)
    lay.addWidget(remove)
    lay.addWidget(QtWidgets.QLabel('Note or reason:'))
    box = QtWidgets.QComboBox()
    box.setEditable(True)
    box.addItems([''] + list(reasons))
    lay.addWidget(box)
    box.setFocus()
    box.lineEdit().selectAll()
    buttons = QtWidgets.QDialogButtonBox(
        QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
    buttons.accepted.connect(dialog.accept)
    buttons.rejected.connect(dialog.reject)
    lay.addWidget(buttons)
    if dialog.exec_() != QtWidgets.QDialog.Accepted:
        return None, ''
    return ('remove' if remove.isChecked() else 'keep'), \
        box.currentText().strip()

def registry(grp):
    """
    Open the tone-marks registry's store root for a window working in a
    zarr group of the store.

    Parameters:
    grp (zarr.Group or other): the group the window writes to (e.g. a
        DataSet's ``root``, or a tuning's ``fq_sweep``). Anything else
        (a mock, None) means no registry.

    Returns:
    root (zarr.Group or None): the store root, writable, or None if the
        group is not a zarr group of a writable store.
    stage (str): the group's path in the store ('' without a registry).
    """
    import zarr
    from ... import daq_layout as dl
    if not isinstance(grp, zarr.Group):
        return None, ''
    try:
        root = dl.store_root(grp, mode = 'r+')
    except Exception:
        return None, ''
    return root, str(grp.path)

def record(root, res_idx, kind, text, stage):
    """
    Write a mark to the registry unless it repeats the resonator's last
    entry (the same kind, text and stage), so a window that saves on
    every move does not pile up copies.

    Parameters:
    root (zarr.Group or None): the store root (None: nothing written).
    res_idx (int or None): resonator index (None: nothing written).
    kind (str): 'bad', 'note' or 'clear'.
    text (str): the reason or note.
    stage (str): the group the user was working in.

    Returns:
    written (bool): True if an entry was written.
    """
    from ... import daq_layout as dl
    if root is None or res_idx is None:
        return False
    marks = dl.tone_marks(root, res_idx)
    text = str(text).strip()
    if marks and marks[-1]['kind'] == kind and marks[-1]['text'] == text \
            and marks[-1]['stage'] == stage:
        return False
    if kind == 'clear' and int(res_idx) not in dl.bad_tones(root):
        return False
    dl.add_tone_mark(root, res_idx, kind, text, stage)
    return True

def mark_label(root, res_idx):
    """
    Describe a resonator's registry state in a few words, for a status
    line: its bad reason, or how many notes it has.

    Parameters:
    root (zarr.Group or None): the store root (None: no registry).
    res_idx (int): resonator index.

    Returns:
    text (str): e.g. 'BAD: double resonance (tune0/power_fits)',
        '2 notes', or ''.
    """
    from ... import daq_layout as dl
    if root is None:
        return ''
    bad = dl.bad_tones(root).get(int(res_idx))
    notes = [e for e in dl.tone_marks(root, res_idx) if e['kind'] == 'note']
    parts = []
    if bad is not None:
        parts.append('BAD: ' + dl.bad_reason(bad))
    if notes:
        parts.append(f'{len(notes)} note' + ('s' if len(notes) > 1 else ''))
    return ', '.join(parts)

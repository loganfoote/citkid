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

GLOBAL_LABEL = 'Also leave the tone out of every later step (global)'

def ask_reason(
    parent, title, reasons = BAD_REASONS, label = 'Reason:', scope = None,
    forced = False, current = None
):
    """
    Ask why a tone is bad: one of some reasons, or typed; with a scope,
    also whether the mark is global (the tone leaves every later step
    of the tuning) or local to this step's results.

    Parameters:
    parent (QWidget or None): the dialog's parent.
    title (str): window title, e.g. 'Mark res 12 bad'.
    reasons (sequence of str): the choices offered. Default
        ``daq_layout.BAD_REASONS``.
    label (str): the prompt. Default 'Reason:'.
    scope (str or None): the default scope, 'global' or 'local', shown
        as a box. Default None: no box, and only the reason is returned.
    forced (bool): If True, the box shows ``scope`` and cannot be
        changed, e.g. global where the tone's outputs become NaN, or
        local for a step no later stage needs. Default False.
    current (str or None): the tone's reason so far, shown as the
        starting text (a tone marked bad already). Default None: the
        first of ``reasons``.
    current (str or None): the tone's reason so far, shown as the
        starting text (a tone marked bad already). Default None: the
        first of ``reasons``.

    Returns:
    if scope is None:
        reason (str or None): the reason, or None if cancelled.
    else:
        choice (tuple or None): (reason, scope), or None if cancelled.
    """
    dialog = QtWidgets.QDialog(parent)
    dialog.setWindowTitle(title)
    lay = QtWidgets.QVBoxLayout(dialog)
    lay.addWidget(QtWidgets.QLabel(label))
    box = QtWidgets.QComboBox()
    box.setEditable(True)
    box.addItems(list(reasons))
    if current:
        box.setCurrentText(str(current))
    lay.addWidget(box)
    box.setFocus()
    box.lineEdit().selectAll()
    global_box = None
    if scope is not None:
        global_box = QtWidgets.QCheckBox(GLOBAL_LABEL)
        global_box.setChecked(scope == 'global')
        global_box.setEnabled(not forced)
        if forced and scope == 'global':
            global_box.setToolTip('The outputs become NaN here, so the '
                                  'later steps cannot use the tone')
        elif forced:
            global_box.setToolTip('No later step uses these outputs, so '
                                  'the tone stays good elsewhere')
        lay.addWidget(global_box)
    buttons = QtWidgets.QDialogButtonBox(
        QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
    buttons.accepted.connect(dialog.accept)
    buttons.rejected.connect(dialog.reject)
    lay.addWidget(buttons)
    if dialog.exec_() != QtWidgets.QDialog.Accepted:
        return None
    reason = box.currentText().strip()
    if global_box is None:
        return reason
    return reason, 'global' if global_box.isChecked() else 'local'

def reason_and_scope(answer, default):
    """
    Split a reason dialog's answer into the reason and the scope: the
    answer is (reason, scope) from ``ask_reason`` with a scope, or a
    bare reason (e.g. a stand-in in tests), which gets the default.

    Parameters:
    answer (tuple, str or None): the dialog's answer.
    default (str): the scope of a bare reason.

    Returns:
    reason (str or None): the reason (None if cancelled).
    scope (str): the scope.
    """
    if answer is None:
        return None, default
    if isinstance(answer, tuple):
        return answer[0], answer[1]
    return answer, default

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

def record(root, res_idx, kind, text, stage, scope = None):
    """
    Write a mark to the registry unless it repeats the resonator's last
    entry (the same kind, text and stage), so a window that saves on
    every move does not pile up copies. A 'clear' is written only if
    the stage has an active bad mark.

    Parameters:
    root (zarr.Group or None): the store root (None: nothing written).
    res_idx (int or None): resonator index (None: nothing written).
    kind (str): 'bad', 'note' or 'clear'.
    text (str): the reason or note.
    stage (str): the group the user was working in.
    scope (str or None): 'global' or 'local'. Default None: the
        stage's default; a clear takes the active mark's.

    Returns:
    written (bool): True if an entry was written.
    """
    from ... import daq_layout as dl
    if root is None or res_idx is None:
        return False
    marks = dl.tone_marks(root, res_idx)
    text = str(text).strip()
    if marks and marks[-1]['kind'] == kind and marks[-1]['text'] == text \
            and marks[-1]['stage'] == stage and \
            (scope is None or marks[-1].get('scope') == scope):
        return False
    if kind == 'clear':
        if not any(e.get('stage', '') == stage for e in
                   dl.active_bad_marks(root, res_idx)):
            return False
        dl.clear_tone_mark(root, res_idx, text, stage, scope)
        return True
    dl.add_tone_mark(root, res_idx, kind, text, stage, scope)
    return True

def current_reason(root, res_idx, stage = None):
    """
    Return the reason a tone is marked bad with, for a stage, or None.

    Parameters:
    root (zarr.Group or None): the store root (``registry``).
    res_idx (int): the resonator.
    stage (str or None): the stage. Default None.

    Returns:
    reason (str or None): the mark's text, or None if not marked bad.
    """
    from ... import daq_layout as dl
    if root is None:
        return None
    entry = dl.bad_tones(root, stage).get(int(res_idx))
    return None if entry is None else str(entry.get('text') or '')

def current_reason(root, res_idx, stage = None):
    """
    Return the reason a tone is marked bad with, for a stage, or None.

    Parameters:
    root (zarr.Group or None): the store root (``registry``).
    res_idx (int): the resonator.
    stage (str or None): the stage. Default None.

    Returns:
    reason (str or None): the mark's text, or None if not marked bad.
    """
    from ... import daq_layout as dl
    if root is None:
        return None
    entry = dl.bad_tones(root, stage).get(int(res_idx))
    return None if entry is None else str(entry.get('text') or '')

def mark_label(root, res_idx, stage = None):
    """
    Describe a resonator's registry state in a few words, for a status
    line: its bad reason (the mark that applies at the stage), or how
    many notes it has.

    Parameters:
    root (zarr.Group or None): the store root (None: no registry).
    res_idx (int): resonator index.
    stage (str or None): the stage asking (``daq_layout.bad_tones``).
        Default None: any global mark.

    Returns:
    text (str): e.g. 'BAD: double resonance (tune0/power_fits)',
        '2 notes', or ''.
    """
    from ... import daq_layout as dl
    if root is None:
        return ''
    bad = dl.bad_tones(root, stage).get(int(res_idx))
    notes = [e for e in dl.tone_marks(root, res_idx) if e['kind'] == 'note']
    parts = []
    if bad is not None:
        parts.append('BAD: ' + dl.bad_reason(bad))
    if notes:
        parts.append(f'{len(notes)} note' + ('s' if len(notes) > 1 else ''))
    return ', '.join(parts)

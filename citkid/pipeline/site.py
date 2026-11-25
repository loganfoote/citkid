"""
Browsable report site: a static HTML site (no server, no extra
dependency) with one page per device store, built from what the store
holds (7_report_site.py).

Layout of the site::

    <out_root>/index.html                      cooldowns and devices
    <out_root>/<cooldown>/<device>/index.html  the device page
    <out_root>/<cooldown>/<device>/figures/    the report figures (PNG)
    <out_root>/<cooldown>/<device>/data/       the report tables (CSV)
    <out_root>/<cooldown>/<device>/res/        one page and figure per
                                               resonator
    <out_root>/<cooldown>/<device>/manifest.json  what the page was built
                                               from (``inventory``)

The device page shows what was taken (sweeps, tunings, runs,
calibrations, results) with the store path of each, the VNA, tuning,
lifetime, multiplicity and noise figures (``report`` and
``vna.report``; every version, chosen with a dropdown; the latest first),
a sortable table of the resonators linking to their pages, and the tone
marks. A ``notes.md`` in the device's folder is shown at the top and
never overwritten. Only reduced results are copied: timestreams and
pulses stay in the store, referenced by path.
"""
import html
import json
import os
import re
import numpy as np
import pandas as pd
from .. import daq_layout as dl
from ..parallel import map_in_pool, resolve_workers
from ..vna import report as vreport
from . import hee_finder as cf
from . import photon_finder as pf
from . import psds
from . import report as treport

FIG_DPI = 110
RES_DPI = 90
NOTES_FILE = 'notes.md'
MANIFEST = 'manifest.json'
ATTACHMENTS = 'attachments'
ATTACHMENTS_INDEX = 'attachments.json'
COMMENTS_FILE = 'comments.json'
COMMENTS_PHP = 'comments.php'
KEEP_ON_OVERWRITE = (COMMENTS_FILE,)

################################################################################
################################# Attachments ##################################
################################################################################
def device_dir(out_root, root):
    """
    Return a store's folder in the site.

    Parameters:
    out_root (str): the site's root folder.
    root (zarr.Group): the store root.

    Returns:
    out_dir (str): ``<out_root>/<cooldown>/<device>``.
    """
    return os.path.join(os.path.expanduser(out_root),
                        str(root.attrs.get('cooldown', 'cooldown')),
                        str(root.attrs.get('device', 'device')))

def load_comments(out_dir):
    """
    Read the comments added on the served site to a device's report
    (``comments.json`` in its folder, written by ``comments.php``;
    the build never writes it).

    Parameters:
    out_dir (str): the device's folder in the site.

    Returns:
    comments (list of dict): ``id``, ``who``, ``when``, ``text``,
        ``edited`` (when it was last edited, or ''), oldest first ([]
        without the file or when it is not a list).
    """
    path = os.path.join(out_dir, COMMENTS_FILE)
    if not os.path.exists(path):
        return []
    try:
        with open(path, encoding = 'utf-8') as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return []
    return [dict(id = str(c.get('id') or ''), who = str(c.get('who') or ''),
                 when = str(c.get('when') or ''),
                 text = str(c.get('text') or ''),
                 edited = str(c.get('edited') or ''))
            for c in data if isinstance(c, dict)] \
        if isinstance(data, list) else []

def comments_summary(comments, width = 80):
    """
    Write the comments of a device as one cell of the top page.

    Parameters:
    comments (list of dict): ``load_comments`` output.
    width (int): characters of the latest comment shown. Default 80.

    Returns:
    text (str): '', or 'n; latest <when> <who>: <text>'.
    """
    if not comments:
        return ''
    last = comments[-1]
    text = ' '.join(last['text'].split())
    if len(text) > width:
        text = text[:width - 3].rstrip() + '...'
    who = f" {last['who']}" if last['who'] else ''
    return f"{len(comments)}; latest {last['when']}{who}: {text}"

def comments_html(comments):
    """
    Render the comments of a device page.

    Parameters:
    comments (list of dict): ``load_comments`` output.

    Returns:
    html (str): one block per comment, or a line saying there are
        none.
    """
    if not comments:
        return '<p class="small">No comments yet.</p>'
    return ''.join(
        f'<div class="comment"><span class="small">{_esc(c["when"])}'
        + (f' {_esc(c["who"])}' if c['who'] else '')
        + (f' (edited {_esc(c["edited"])})' if c.get('edited') else '')
        + '</span><br>' + _esc(c['text']).replace('\n', '<br>') + '</div>'
        for c in comments)

COMMENTS_SCRIPT = '''
(function () {
  if (!location.protocol.startsWith('http')) { return; }
  var form = document.getElementById('comment-form');
  form.hidden = false;
  var esc = function (s) { var d = document.createElement('div');
    d.textContent = s; return d.innerHTML; };
  var field = function (name) { return form.elements[name]; };
  var byId = {};
  var reset = function () {
    field('action').value = 'add'; field('id').value = '';
    field('text').value = ''; form.querySelector('button').textContent =
      'Add comment'; document.getElementById('comment-cancel').hidden = true;
  };
  document.getElementById('comment-cancel').onclick = function (e) {
    e.preventDefault(); reset(); };
  var render = function (list) {
    byId = {};
    document.getElementById('comments-count').textContent =
      'Comments (' + list.length + ')';
    document.getElementById('comments-list').innerHTML = list.length
      ? list.map(function (c, i) {
          var id = c.id || String(i); byId[id] = c;
          return '<div class="comment"><span class="small">'
            + esc(c.when || '') + (c.who ? ' ' + esc(c.who) : '')
            + (c.edited ? ' (edited ' + esc(c.edited) + ')' : '')
            + ' &middot; <a href="#comments" data-edit="' + esc(id)
            + '">edit</a> &middot; <a href="#comments" data-delete="'
            + esc(id) + '">delete</a></span><br>'
            + esc(c.text || '').replace(/\\n/g, '<br>') + '</div>'; }).join('')
      : '<p class="small">No comments yet.</p>';
  };
  document.getElementById('comments-list').onclick = function (e) {
    var t = e.target;
    if (t.dataset && t.dataset.edit !== undefined) {
      e.preventDefault();
      var c = byId[t.dataset.edit] || {};
      field('action').value = 'edit'; field('id').value = t.dataset.edit;
      field('who').value = c.who || ''; field('text').value = c.text || '';
      form.querySelector('button').textContent = 'Save edit';
      document.getElementById('comment-cancel').hidden = false;
      field('text').focus();
    } else if (t.dataset && t.dataset.delete !== undefined) {
      e.preventDefault();
      if (!confirm('Delete this comment?')) { return; }
      field('action').value = 'delete'; field('id').value = t.dataset.delete;
      field('text').required = false;
      form.submit();
    }
  };
  fetch('comments.json', {cache: 'no-store'})
    .then(function (r) { return r.ok ? r.json() : []; })
    .then(function (list) { if (Array.isArray(list)) { render(list); } })
    .catch(function () {});
})();
'''

def comments_section(out_dir, cooldown, device):
    """
    Render the Comments section of a device page: the comments as of
    the build, and, when the page is served (the NAS), the live list
    from ``comments.json`` with edit and delete links, and a form
    posting to ``comments.php`` (add, or save an edit).

    Parameters:
    out_dir (str): the device's folder in the site.
    cooldown (str), device (str): its folders (the form's fields).

    Returns:
    html (str): the section.
    """
    comments = load_comments(out_dir)
    form = (
        '<form id="comment-form" method="post" '
        f'action="../../{COMMENTS_PHP}" hidden>'
        f'<input type="hidden" name="cooldown" value="{_esc(cooldown)}">'
        f'<input type="hidden" name="device" value="{_esc(device)}">'
        '<input type="hidden" name="action" value="add">'
        '<input type="hidden" name="id" value="">'
        '<input name="who" maxlength="100" placeholder="your name"> '
        '<textarea name="text" required maxlength="5000" rows="3" '
        'placeholder="comment"></textarea> '
        '<button type="submit">Add comment</button> '
        '<a href="#comments" id="comment-cancel" hidden>cancel</a>'
        '</form>')
    return (
        '<details open id="comments"><summary><h2 id="comments-count">'
        f'Comments ({len(comments)})</h2></summary>'
        f'<div id="comments-list">{comments_html(comments)}</div>{form}'
        '<p class="small">Comments are added on the served site and '
        f'kept in {COMMENTS_FILE} next to this page, which rebuilding '
        'the report leaves alone.</p>'
        f'<script>{COMMENTS_SCRIPT}</script></details>')

def clear_device_dir(out_dir, keep = KEEP_ON_OVERWRITE):
    """
    Empty a device's folder for an overwrite, keeping the files the
    build does not own (the comments).

    Parameters:
    out_dir (str): the device's folder in the site.
    keep (sequence of str): file names kept. Default
        ``KEEP_ON_OVERWRITE``.
    """
    import shutil
    if not os.path.isdir(out_dir):
        return
    for name in os.listdir(out_dir):
        if name in keep:
            continue
        path = os.path.join(out_dir, name)
        if os.path.isdir(path) and not os.path.islink(path):
            shutil.rmtree(path)
        else:
            os.remove(path)

def load_attachments(out_dir):
    """
    Read a device folder's list of attached files.

    Parameters:
    out_dir (str): the device's folder in the site.

    Returns:
    attachments (dict): item path ('' for the store, a group's path, or
        a run's first stream path): list of file paths relative to the
        device folder.
    """
    path = os.path.join(out_dir, ATTACHMENTS_INDEX)
    if not os.path.exists(path):
        return {}
    with open(path, encoding = 'utf-8') as fh:
        return json.load(fh)

def add_attachment(out_dir, item, source):
    """
    Copy a file into the device folder's attachments and record it with
    an item, so the pages link it next to the item's notes. The file
    never goes into the store.

    Parameters:
    out_dir (str): the device's folder in the site.
    item (str): the item ('' for the store, a group's path, or a run's
        first stream path, as ``daq_layout.notes_entries``).
    source (str): the file to copy.

    Returns:
    rel (str): the copy's path relative to the device folder.
    """
    import shutil
    folder = os.path.join(out_dir, ATTACHMENTS, item.replace('/', '__')
                          or 'store')
    os.makedirs(folder, exist_ok = True)
    name = os.path.basename(source)
    target = os.path.join(folder, name)
    # The same file again is kept once; another file of the same name
    # gets a numbered name
    import filecmp
    stem, ext = os.path.splitext(name)
    k = 1
    while os.path.exists(target) and not filecmp.cmp(source, target,
                                                     shallow = False):
        target = os.path.join(folder, f'{stem}_{k}{ext}')
        k += 1
    if not os.path.exists(target):
        shutil.copy2(source, target)
    rel = _rel(target, out_dir)
    attachments = load_attachments(out_dir)
    files = attachments.setdefault(item, [])
    if rel not in files:
        files.append(rel)
    with open(os.path.join(out_dir, ATTACHMENTS_INDEX), 'w',
              encoding = 'utf-8') as fh:
        json.dump(attachments, fh, indent = 1)
    return rel

def remove_attachment(out_dir, item, rel):
    """
    Remove an attached file from an item: the record goes, and the copy
    is deleted when no other item lists it.

    Parameters:
    out_dir (str): the device's folder in the site.
    item (str): the item ('' for the store).
    rel (str): the file's path relative to the device folder, as
        ``load_attachments`` lists it.

    Returns:
    removed (bool): True if the item listed the file.
    """
    attachments = load_attachments(out_dir)
    files = attachments.get(item, [])
    if rel not in files:
        return False
    files.remove(rel)
    if not files:
        del attachments[item]
    with open(os.path.join(out_dir, ATTACHMENTS_INDEX), 'w',
              encoding = 'utf-8') as fh:
        json.dump(attachments, fh, indent = 1)
    if not any(rel in v for v in attachments.values()):
        path = os.path.join(out_dir, rel)
        if os.path.exists(path):
            os.remove(path)
    return True

def attachment_links(attachments, item):
    """
    Return an item's attachments as links for the pages.

    Parameters:
    attachments (dict): ``load_attachments`` output.
    item (str): the item.

    Returns:
    links (list of tuple): (file name, path relative to the device
        folder).
    """
    return [(os.path.basename(p), p) for p in attachments.get(item, [])]

################################################################################
################################## Inventory ###################################
################################################################################
def _attr(grp, key, default = None):
    """
    Return one attribute of a group, as a plain value.

    Parameters:
    grp (zarr.Group): the group.
    key (str): attribute name.
    default: value if missing. Default None.

    Returns:
    value: the attribute.
    """
    return grp.attrs.get(key, default)

def _settings_of(grp):
    """
    Pick the readout settings recorded on an acquisition group.

    Parameters:
    grp (zarr.Group): a calibration sweep, target sweep or stream group.

    Returns:
    settings (dict): the per-module full scale (dBm), ADC attenuation
        (dB) and Nyquist zone, and ``nsamps``, ``sample_freq``,
        ``dec_stage`` where recorded.
    """
    out = {}
    for key, value in grp.attrs.items():
        for stem in ('full_scale_dbm', 'adc_attenuation_db', 'nyquist_zone'):
            if key.startswith(stem):
                out[key] = value
    for key in ('nsamps', 'sample_freq', 'dec_stage', 'timestamp'):
        if key in grp.attrs:
            out[key] = grp.attrs[key]
    return out

def device_parts(device):
    """
    Split a device name into the wafer name and the die.

    Parameters:
    device (str): e.g. 'L261022p1-die7'.

    Returns:
    name (str): the name before '-die', with a 'p' between digits read
        as a period: 'L261022.1'.
    die (str): the die number, '' if none.
    """
    m = re.match(r'^(.*?)(?:-die(\w+))?$', str(device))
    name = re.sub(r'(?<=\d)p(?=\d)', '.', m.group(1))
    return name, m.group(2) or ''

def store_suffix(root):
    """
    Return a store's file name and suffix (``<device>_<suffix>.zarr``).

    Parameters:
    root (zarr.Group): the store root.

    Returns:
    name (str): the file name without '.zarr' ('' if unknown).
    suffix (str): what follows the device name; when the file name
        does not start with the device (a device attribute typed
        differently), what follows its last '_' ('' if none).
    """
    path = str(getattr(root.store, 'root', '') or '')
    name = os.path.basename(os.path.normpath(path)) if path else ''
    if name.endswith('.zarr'):
        name = name[:-5]
    device = str(root.attrs.get('device', ''))
    if device and name.startswith(device + '_'):
        suffix = name[len(device) + 1:]
    elif '_' in name and not (device and name == device):
        suffix = name.rsplit('_', 1)[1]
    else:
        suffix = ''
    return name, suffix

def store_newest(root):
    """
    Return the newest creation time recorded in a store, to tell whether
    a device page is out of date.

    Parameters:
    root (zarr.Group): the store root.

    Returns:
    newest (str): the latest ``created`` attribute ('YYYY-MM-DD
        HH:MM:SS'), or '' if none.
    """
    stamps = ('created', 'tau_refit', 'notes_updated')
    times = [str(_attr(root, key, '')) for key in stamps]
    for name in root.group_keys():
        g = root[name]
        times += [str(_attr(g, key, '')) for key in stamps]
        for sub in g.group_keys():
            times += [str(_attr(g[sub], key, '')) for key in stamps]
    for entries in dl.tone_marks(root).values():
        for e in entries:
            times.append(str(e.get('created', '')))
    return max(times)

def inventory(root):
    """
    List what a store holds, for the device page and its manifest.

    Parameters:
    root (zarr.Group): the store root.

    Returns:
    inv (dict): ``device``, ``cooldown``, ``attrs`` (the root
        attributes), ``newest`` (``store_newest``), ``sweeps``,
        ``tunings``, ``runs``, ``calibrations``, ``photons``, ``hee``,
        ``psds`` (lists of dict, in store order, each with its ``path``,
        ``created``, ``notes`` and a few numbers), and ``marks`` (list of
        dict: ``res_idx``, ``kind``, ``text``, ``stage``, ``created``).
    """
    name, suffix = store_suffix(root)
    wafer, die = device_parts(_attr(root, 'device', 'device'))
    inv = dict(device = str(_attr(root, 'device', 'device')),
               cooldown = str(_attr(root, 'cooldown', 'cooldown')),
               wafer = wafer, die = die, store_name = name, suffix = suffix,
               attrs = dl.to_jsonable(dict(root.attrs)),
               newest = store_newest(root))
    sweeps = []
    for k in dl.list_indices(root, 'vna'):
        g = root[dl.set_name(k, 'vna')]
        fit = g['full_sweep_fit'] if 'full_sweep_fit' in g else None
        res = root.get(dl.set_name(k, 'resonators'))
        sweeps.append(dict(
            k = k, path = g.path, created = _attr(g, 'created', ''),
            notes = _attr(g, 'notes', ''), amplitude = _attr(g, 'amplitude'),
            setup = _attr(g, 'setup'),
            fmin = _attr(g, 'fmin'), fmax = _attr(g, 'fmax'),
            n_points = int(g['f'].shape[0]) if 'f' in g else None,
            fit = fit is not None,
            n_resonators = int(res['res_idxs_all'].shape[0])
            if res is not None else None,
            report = _attr(fit, 'report', '') if fit is not None else '',
            in_report = dl.in_report(g)))
    inv['sweeps'] = sweeps
    tunings = []
    for k in dl.list_indices(root, 'tune'):
        t = root[dl.set_name(k, 'tune')]
        tones = dl.load_tones(t['tones']) if 'tones' in t else None
        ps = t.get('power_sweep')
        pc = t.get('power_choice')
        # The tuning's own notes, else those its steps were taken with
        notes = next((_attr(g, 'notes', '') for g in
                      (t, t.get('tones'), ps) if g is not None
                      and _attr(g, 'notes', '')), '')
        tunings.append(dict(
            k = k, path = t.path, created = _attr(t, 'created', ''),
            in_report = dl.in_report(t), name = dl.tuning_name(t),
            notes = notes, resonators = _attr(t, 'resonators'),
            setup = _attr(t, 'setup'),
            n_tones = int(len(tones['res_idxs'])) if tones else None,
            n_tone_sets = int(len(np.unique(tones['tone_set'])))
            if tones else None,
            powers = list(_attr(ps, 'powers', [])) if ps is not None else [],
            power_choice = {key: _attr(pc, key) for key in
                            ('anl', 'limits', 'fallback')}
            if pc is not None else {},
            fit = 'power_fits' in t and 'xy_fit' in t['power_fits'],
            cal_sweeps = [dict(path = t[c].path, tone_set =
                               _attr(t[c], 'tone_set'),
                               created = _attr(t[c], 'created', ''),
                               notes = _attr(t[c], 'notes', ''),
                               in_report = dl.in_report(t[c]),
                               n_tones = int(t[c]['res_idxs'].shape[0])
                               if 'res_idxs' in t[c] else None,
                               settings = _settings_of(t[c]))
                          for c in sorted(t.group_keys())
                          if c.startswith('cal')]))
    inv['tunings'] = tunings
    runs = []
    for run in dl.stream_runs(root):
        rec = {key: run.get(key) for key in
               ('run', 'group', 'name', 'notes', 'dec_stage', 'duration_s',
                'tone_set', 'calibration', 'tuning', 'complete', 'created',
                'paths', 'nsamples', 'setup')}
        settings = {}
        if run.get('paths'):
            s = root[run['paths'][0]]
            settings.update(_settings_of(s))
            w = root['/'.join(run['paths'][0].split('/')[:-1])]
            rec['n_tones'] = int(w['res_idxs'].shape[0]) \
                if 'res_idxs' in w else None
        if run.get('calibration') and run['calibration'] in root:
            cal = root[run['calibration']]
            for key, value in _settings_of(cal).items():
                settings.setdefault(key, value)
        rec['settings'] = settings
        rec['in_report'] = dl.in_report(run)
        runs.append(rec)
    inv['runs'] = runs
    cals = []
    for c in dl.list_indices(root, 'ts_cal'):
        g = root[dl.set_name(c, 'ts_cal')]
        sets = []
        for name in sorted(g.group_keys()):
            sg = g[name]
            n = None
            n_ok = None
            if 'poly_x' in sg and 'row_exists' in sg['poly_x']:
                exists = np.asarray(sg['poly_x/row_exists'][:], bool)
                n, n_ok = int(len(exists)), int(exists.sum())
            sets.append(dict(name = name, n_tones = n, n_calibrated = n_ok,
                             resonators = _attr(sg, 'resonators'),
                             iq_fit = 'iq_popt' in sg))
        cals.append(dict(c = c, path = g.path,
                         created = _attr(g, 'created', ''),
                         in_report = dl.in_report(g),
                         notes = _attr(g, 'notes', ''),
                         selection = _attr(g, 'selection', []), sets = sets))
    inv['calibrations'] = cals
    for kind, names in (('photons', pf.list_results(root)),
                        ('hee', cf.list_results(root))):
        items = []
        for name in names:
            g = root[f'{kind}/{name}']
            item = dict(name = name, path = g.path,
                        created = _attr(g, 'created', ''),
                        in_report = dl.in_report(g),
                        notes = _attr(g, 'notes', ''),
                        run = _attr(g, 'run'), run_name = _attr(g, 'name'),
                        tone_set = _attr(g, 'tone_set'),
                        calibration = _attr(g, 'calibration')
                        or _attr(g, 'source'),
                        complete = 'res_idxs' in g and 'tau' in g)
            if item['complete']:
                tau = np.asarray(g['tau'][:], float)
                bad = np.asarray(g['bad'][:], bool) if 'bad' in g \
                    else np.zeros(len(tau), bool)
                ok = np.isfinite(tau) & ~bad
                item.update(n_tones = int(len(tau)), n_bad = int(bad.sum()),
                            tau_median_us = float(np.median(tau[ok]) * 1e6)
                            if ok.any() else None)
                if 'rate' in g:
                    rate = np.asarray(g['rate'][:], float)[ok]
                    item['rate_median'] = float(np.median(rate)) \
                        if len(rate) else None
            items.append(item)
        inv[kind] = items
    groups = []
    for p in dl.list_indices(root, psds.PREFIX):
        g = root[dl.set_name(p, psds.PREFIX)]
        names = [n for n in g.group_keys() if n.startswith('timestream')]
        complete = 'f_reduced' in g and any('res_idxs' in g[n] for n in names)
        item = dict(p = p, path = g.path, created = _attr(g, 'created', ''),
                    in_report = dl.in_report(g),
                    notes = _attr(g, 'notes', ''), name = _attr(g, 'name'),
                    timestreams = _attr(g, 'timestreams', []),
                    settings = _attr(g, 'settings', {}), complete = complete,
                    n_timestreams = len(names))
        if complete:
            item['description'] = psds.describe_group(g)
        groups.append(item)
    inv['psds'] = groups
    marks = []
    for r, entries in sorted(dl.tone_marks(root).items()):
        for e in entries:
            marks.append(dict(res_idx = int(r), kind = e.get('kind'),
                              scope = '' if e.get('kind') == 'note'
                              else dl.mark_scope(e),
                              text = e.get('text', ''),
                              stage = e.get('stage', ''),
                              tuning = e.get('tuning') or '',
                              created = e.get('created', '')))
    inv['marks'] = marks
    return dl.to_jsonable(inv)

################################################################################
################################### Tuning #####################################
################################################################################
def tuning_table(root, tune):
    """
    Tabulate a tuning's tones and their chosen powers. The tuning's own
    fits (made to choose the powers) are not reported: the user checks
    the chosen powers, not every fit.

    Parameters:
    root (zarr.Group): the store root.
    tune (zarr.Group): ``tune{k}``.

    Returns:
    table (pd.DataFrame): per tone: ``res_idx``, ``fres`` (Hz),
        ``tone_set``, ``power_dbm`` (the tone's power), ``set`` (the
        power-sweep set nearest the chosen power, or -1), ``user_chosen``,
        ``fallback`` (how the power was chosen), ``bad`` (marked bad in
        the store).
    """
    tones = dl.load_tones(tune['tones'])
    res_idxs = np.asarray(tones['res_idxs'], int)
    n = len(res_idxs)
    bad_marks = dl.bad_tones(root, f'{tune.path}/tones')
    out = dict(res_idx = res_idxs, fres = np.asarray(tones['fres'], float),
               tone_set = np.asarray(tones['tone_set'], int),
               power_dbm = np.asarray(tones['ares'], float),
               set = np.full(n, -1),
               user_chosen = np.zeros(n, bool), fallback = np.zeros(n, bool),
               bad = np.array([int(r) in bad_marks for r in res_idxs]))
    pc = tune.get('power_choice')
    if pc is not None and 'res_idxs' in pc:
        pos = {int(r): i for i, r in enumerate(pc['res_idxs'][:])}
        for key in ('user_chosen', 'fallback'):
            if key in pc:
                v = np.asarray(pc[key][:], bool)
                out[key] = np.array([v[pos[int(r)]] if int(r) in pos
                                     else False for r in res_idxs])
    fits = tune.get('power_fits')
    ps = tune.get('power_sweep')
    if fits is None or ps is None or 'xy_fit' not in fits or \
            'set0' not in ps or 'res_idxs' not in ps['set0']:
        return pd.DataFrame(out)
    powers = np.asarray(ps.attrs.get('powers', []), float)
    rows = {int(r): i for i, r in enumerate(ps['set0/res_idxs'][:])}
    x_chosen = np.asarray(fits['xy_fit/x_chosen'][:], float)
    for i, r in enumerate(res_idxs):
        row = rows.get(int(r))
        if row is None or row >= len(x_chosen) or \
                not np.isfinite(x_chosen[row]) or not len(powers):
            continue
        out['set'][i] = int(np.argmin(np.abs(powers - x_chosen[row])))
    return pd.DataFrame(out)

def plot_tuning_summary(table, title = ''):
    """
    Plot the chosen power against resonant frequency (MHz), with a
    histogram at the side.

    Parameters:
    table (pd.DataFrame): ``tuning_table`` output.
    title (str): figure title. Default ''.

    Returns:
    fig (Figure): the figure.
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    rows = [('power_dbm', 'Chosen power (dBm)')]
    fig = plt.figure(figsize = (12, 3. * len(rows)))
    gs = GridSpec(len(rows), 2, width_ratios = [4, 1], figure = fig,
                  wspace = 0.05, hspace = 0.3)
    for k, (col, label) in enumerate(rows):
        ax = treport._scatter_with_hist(fig, gs, k, table, col, label)
        if k == 0:
            n_bad = int(table['bad'].sum())
            ax.set_title(title + (f' ({n_bad} tones marked bad left out)'
                                  if n_bad else ''))
    ax.set_xlabel('Resonant frequency (MHz)')
    return fig

################################################################################
############################# Stream calibrations ##############################
################################################################################
def calibration_tables(root, cal):
    """
    Tabulate the IQ fits of a stream calibration's DataSets, where
    5_calibrate_streams.py fit them (``FIT_IQ``).

    Parameters:
    root (zarr.Group): the store root.
    cal (zarr.Group): ``ts_cal{c}``.

    Returns:
    tables (dict): DataSet name: pd.DataFrame per tone: ``res_idx``,
        ``fres`` (Hz, the resonator list's), ``fr`` (Hz, fitted), ``Qr``,
        ``Qc``, ``Qi``, ``a_nl``, ``nrmse`` (NaN where the fit is
        missing), ``bad`` (marked bad in the store). Only the DataSets
        with fits.
    """
    from .stream_sources import open_calibration
    out = {}
    names = [n for n in cal.group_keys() if 'iq_popt' in cal[n]]
    if not names:
        return out
    try:
        datasets = open_calibration(root, cal)
    except KeyError:
        return out
    for name in names:
        if name not in datasets:
            continue
        bad_marks = dl.bad_tones(root, f'{cal.path}/{name}')
        DS = datasets[name]['DS']
        res_idxs = np.asarray(DS.res_idxs[:], int)
        n = len(res_idxs)
        res = root[datasets[name]['resonators']]
        fres_all = np.asarray(res['fres_all'][:], float)
        table = dict(res_idx = res_idxs,
                     fres = np.where((res_idxs >= 0) &
                                     (res_idxs < len(fres_all)),
                                     fres_all[np.clip(res_idxs, 0,
                                                      len(fres_all) - 1)],
                                     np.nan),
                     fr = np.full(n, np.nan), Qr = np.full(n, np.nan),
                     Qc = np.full(n, np.nan), Qi = np.full(n, np.nan),
                     a_nl = np.full(n, np.nan), nrmse = np.full(n, np.nan),
                     bad = np.array([int(r) in bad_marks for r in res_idxs]))
        saved = np.flatnonzero(DS.saved_row_mask('iq_popt'))
        if len(saved):
            popt = np.asarray(DS.iq_popt[saved.tolist()], float)
            fr, Qr, amp, a = popt[:, 0], popt[:, 1], popt[:, 2], popt[:, 4]
            with np.errstate(divide = 'ignore', invalid = 'ignore'):
                Qc = Qr / amp
                Qi = 1 / (1 / Qr - 1 / Qc)
            table['fr'][saved], table['Qr'][saved] = fr, Qr
            table['Qc'][saved], table['Qi'][saved] = Qc, Qi
            table['a_nl'][saved] = a
            if DS.saved_row_mask('iq_nrmse').any():
                ok = np.flatnonzero(DS.saved_row_mask('iq_nrmse'))
                table['nrmse'][ok] = np.asarray(DS.iq_nrmse[ok.tolist()],
                                                float).ravel()
        out[name] = pd.DataFrame(table)
    return out

def plot_calibration_fits(table, title = ''):
    """
    Plot a calibration's IQ fits (Qr, Qc, Qi, nonlinearity) against
    resonant frequency (MHz), each with a histogram at the side.

    Parameters:
    table (pd.DataFrame): one of ``calibration_tables``.
    title (str): figure title. Default ''.

    Returns:
    fig (Figure): the figure.
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    rows = [(col, label) for col, label in
            (('Qr', 'Qr'), ('Qc', 'Qc'), ('Qi', 'Qi'),
             ('a_nl', 'Nonlinearity a_nl'))]
    fig = plt.figure(figsize = (12, 3. * len(rows)))
    gs = GridSpec(len(rows), 2, width_ratios = [4, 1], figure = fig,
                  wspace = 0.05, hspace = 0.3)
    for k, (col, label) in enumerate(rows):
        ax = treport._scatter_with_hist(fig, gs, k, table, col, label)
        if k == 0:
            n_bad = int(table['bad'].sum())
            ax.set_title(title + (f' ({n_bad} tones marked bad left out)'
                                  if n_bad else ''))
    ax.set_xlabel('Resonant frequency (MHz)')
    return fig

################################################################################
################################# Resonators ###################################
################################################################################
class DeviceData:
    """
    The parts of a store the resonator pages read, opened once.

    Attributes:
    root (zarr.Group): the store root.
    inv (dict): ``inventory`` output.
    tunings (list of tuple): (tune group, ``tuning_table``) per tuning.
    calibrations (list of tuple): (cal group, ``calibration_tables``)
        per stream calibration with IQ fits.
    vna_tables (dict): sweep index: ``vna.report.resonator_table``.
    results (dict): 'photons' / 'hee': list of (name, group, table).
    psd_groups (list): the complete ``ts_psd{p}`` groups.
    """

    def __init__(self, root, inv):
        """
        Open the store's results.

        Parameters:
        root (zarr.Group): the store root.
        inv (dict): ``inventory`` output.
        """
        self.root, self.inv = root, inv
        self.tunings = []
        for t in inv['tunings']:
            tune = root[t['path']]
            if 'tones' in tune:
                self.tunings.append((tune, tuning_table(root, tune)))
        self.calibrations = []
        for c in inv['calibrations']:
            if any(s.get('iq_fit') for s in c['sets']):
                cal = root[c['path']]
                tables = calibration_tables(root, cal)
                if tables:
                    self.calibrations.append((cal, tables))
        self.vna_tables = {}
        for s in inv['sweeps']:
            if s['fit']:
                self.vna_tables[s['k']] = vreport.resonator_table(
                    root[s['path']]['full_sweep_fit'])
        self.results = {}
        for kind in ('photons', 'hee'):
            items = []
            for item in inv[kind]:
                if item['complete']:
                    grp = root[item['path']]
                    items.append((item['name'], grp,
                                  treport.lifetime_table(root, grp, kind)))
            self.results[kind] = items
        self.psd_groups = [root[g['path']] for g in inv['psds']
                           if g['complete']]
        self.marks = dl.tone_marks(root)

    def res_idxs(self):
        """
        Return the resonators that have a page: every tone of every
        tuning, in order.

        Returns:
        res_idxs (list of int): resonator indices.
        """
        seen = []
        for _, table in self.tunings:
            for r in table['res_idx']:
                if int(r) >= 0 and int(r) not in seen:
                    seen.append(int(r))
        return seen

def resonator_rows(data, r):
    """
    Gather one resonator's numbers from every step and result.

    Parameters:
    data (DeviceData): the store's results.
    r (int): resonator index.

    Returns:
    sections (list of tuple): (title, pd.DataFrame) per step that has
        the resonator: 'VNA fit', 'Tuning', 'Photons', 'High-energy events',
        'Noise', 'Marks'.
    """
    out = []
    for k, table in sorted(data.vna_tables.items()):
        row = table[table['res_idx'] == r]
        if len(row):
            out.append((f'VNA fit (vna{k})', row.assign(
                sweep = f'vna{k}').reset_index(drop = True)))
    rows = []
    for tune, table in data.tunings:
        row = table[table['res_idx'] == r]
        if len(row):
            rows.append(row.assign(tuning = tune.basename))
    if rows:
        out.append(('Tuning', pd.concat(rows).reset_index(drop = True)))
    rows = []
    for cal, tables in data.calibrations:
        for name, table in tables.items():
            row = table[table['res_idx'] == r]
            if len(row):
                rows.append(row.assign(calibration = f'{cal.path}/{name}'))
    if rows:
        out.append(('Stream calibration (IQ fit)',
                    pd.concat(rows).reset_index(drop = True)))
    for kind, title in (('photons', 'Photons'),
        ('hee', 'High-energy events')):
        rows = []
        for name, grp, table in data.results[kind]:
            row = table[table['res_idx'] == r]
            if len(row):
                rows.append(row.assign(result = f'{kind}/{name}'))
        if rows:
            out.append((title, pd.concat(rows).reset_index(drop = True)))
    rows = []
    for g in data.psd_groups:
        f_reduced = np.asarray(g['f_reduced'][:], float)
        for name in sorted(n for n in g.group_keys()
                           if n.startswith('timestream')):
            t = g[name]
            idx = np.flatnonzero(np.asarray(t['res_idxs'][:], int) == r)
            if not len(idx):
                continue
            i = int(idx[0])
            rec = dict(group = f"{g.path} ({g.attrs.get('name') or ''})",
                       timestream = f"{t.attrs.get('run')} "
                       f"({t.attrs.get('name')})",
                       status = str(t['status'][:][i]),
                       bad = bool(t['bad'][:][i]) if 'bad' in t else False)
            for key, label in (('sxx_reduced', 'sxx'),
                               ('sxx_reduced_cn', 'sxx_cn'),
                               ('sfactor_reduced', 'sfactor')):
                if key in t:
                    v = np.asarray(t[key][:][i], float)
                    for fr, vi in zip(f_reduced, v):
                        rec[f'{label}_{fr:g}'] = vi
            rows.append(rec)
    if rows:
        out.append(('Noise', pd.DataFrame(rows)))
    marks = data.marks.get(r) or data.marks.get(str(r))
    if marks:
        out.append(('Marks', pd.DataFrame(marks)))
    return out

def _sweep_row(grp, r):
    """
    Return a tone's row in a target sweep group, by resonator index.

    Parameters:
    grp (zarr.Group): a sweep group with ``res_idxs``.
    r (int): resonator index.

    Returns:
    row (int or None): the row, or None if the sweep lacks the tone.
    """
    if 'res_idxs' not in grp:
        return None
    idx = np.flatnonzero(np.asarray(grp['res_idxs'][:], int) == r)
    return int(idx[0]) if len(idx) else None

def resonator_figure(data, r):
    """
    Draw one resonator's figure: its IQ loop at the chosen power, its
    calibration sweep, its mean photon and HEE pulses (every
    result), and its noise spectra (every set).

    Parameters:
    data (DeviceData): the store's results.
    r (int): resonator index.

    Returns:
    fig (Figure): the figure.
    """
    import matplotlib.pyplot as plt
    root = data.root
    fig, axes = plt.subplots(2, 3, figsize = (13, 7.5))
    ax = axes[0, 0]
    drawn = False
    for tune, table in data.tunings:
        row = table[table['res_idx'] == r]
        if not len(row) or 'power_sweep' not in tune:
            continue
        row = row.iloc[0]
        s = int(row['set'])
        g = tune['power_sweep'].get(dl.set_name(max(s, 0)))
        if g is None or 'fine_sweep' not in g:
            continue
        i = _sweep_row(g, r)
        if i is None:
            continue
        z = np.asarray(g['fine_sweep/z'][i])
        ax.plot(z.real, z.imag, '.-', ms = 3, lw = 0.8, color = 'C0')
        ax.set_title(f"{tune.basename}: {row['power_dbm']:.1f} dBm, tone "
                     f"set {int(row['tone_set'])}", fontsize = 9)
        drawn = True
        break
    ax.set_xlabel('I')
    ax.set_ylabel('Q')
    ax.set_aspect('equal', adjustable = 'datalim')
    if not drawn:
        ax.set_title('no power sweep', fontsize = 9)
    ax = axes[0, 1]
    drawn = False
    for t in data.inv['tunings']:
        for c in t['cal_sweeps']:
            g = root[c['path']]
            i = _sweep_row(g, r)
            if i is None or 'fine_sweep' not in g:
                continue
            f = np.asarray(g['fine_sweep/f'][i], float)
            z = np.asarray(g['fine_sweep/z'][i])
            with np.errstate(divide = 'ignore'):
                ax.plot((f - f.mean()) / 1e3, 20 * np.log10(np.abs(z)),
                        lw = 0.8, label = f"{c['path']} ({f.mean() / 1e6:,.3f}"
                        ' MHz)')
            drawn = True
    ax.set_xlabel('Frequency offset (kHz)')
    ax.set_ylabel('|S21| (dB)')
    fit_text = ''
    for cal, tables in data.calibrations:
        for name, table in tables.items():
            row = table[table['res_idx'] == r]
            if len(row) and np.isfinite(row.iloc[0]['Qr']):
                row = row.iloc[0]
                fit_text = (f"; {cal.path} fit: Qr {row['Qr']:,.0f}, Qc "
                            f"{row['Qc']:,.0f}, Qi {row['Qi']:,.0f}, a_nl "
                            f"{row['a_nl']:.2f}")
    ax.set_title(('Calibration sweeps' if drawn else 'no calibration sweep')
                 + fit_text, fontsize = 8)
    if drawn:
        ax.legend(fontsize = 7)
    ax.grid(alpha = 0.3)
    ax = axes[0, 2]
    drawn = False
    for name, grp, table in data.results['photons']:
        i = _sweep_row(grp, r)
        if i is None or 'avg_pulse' not in grp:
            continue
        avg = np.asarray(grp['avg_pulse'][i], float)
        if not np.isfinite(avg).any():
            continue
        dt = float(grp.attrs['dt'])
        pre = int(grp.attrs.get('pre', int(np.nanargmax(avg))))
        t = (np.arange(len(avg)) - pre) * dt * 1e3
        row = table[table['res_idx'] == r].iloc[0]
        ax.plot(t, avg * treport.PPM, lw = 1,
                label = f"{name}: tau {row['tau'] * 1e6:,.0f} us, "
                f"{row['rate']:.3g} /s")
        drawn = True
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Mean photon pulse, -x (kHz/GHz)')
    ax.set_title('Photons' if drawn else 'no photon result', fontsize = 9)
    if drawn:
        ax.legend(fontsize = 7)
    ax.grid(alpha = 0.3)
    ax = axes[1, 0]
    drawn = False
    for name, grp, table in data.results['hee']:
        i = _sweep_row(grp, r)
        if i is None or 'avg_pulse' not in grp:
            continue
        avg = np.asarray(grp['avg_pulse'][i], float)
        if not np.isfinite(avg).any():
            continue
        dt = float(grp.attrs['dt'])
        pre = int(grp.attrs.get('pre', 0))
        t = (np.arange(len(avg)) - pre) * dt * 1e3
        row = table[table['res_idx'] == r].iloc[0]
        ax.plot(t, avg * treport.PPM, lw = 1,
                label = f"{name}: tau {row['tau'] * 1e6:,.0f} us, "
                f"{row['n_used']} pulses")
        drawn = True
    ax.set_xlabel('Time from the tail level (ms)')
    ax.set_ylabel('Mean HEE pulse, -x (kHz/GHz)')
    ax.set_title('High-energy events' if drawn else 'no HEE result',
                 fontsize = 9)
    if drawn:
        ax.legend(fontsize = 7)
    ax.grid(alpha = 0.3)
    ax_s, ax_l = axes[1, 1], axes[1, 2]
    drawn_s = drawn_l = False
    for k, g in enumerate(data.psd_groups):
        for name in sorted(n for n in g.group_keys()
                           if n.startswith('timestream')):
            t = g[name]
            i = _sweep_row(t, r)
            if i is None or 'f_binned' not in t:
                continue
            fb = np.asarray(t['f_binned'][:], float)
            label = f"{g.path} {t.attrs.get('name')}"
            color = f'C{k % 10}'
            if 'sxx_binned' in t:
                ax_s.loglog(fb, t['sxx_binned'][i], lw = 0.8, alpha = 0.5,
                            color = color, label = label + ' raw')
                drawn_s = True
            if 'sxx_binned_cn' in t:
                ax_s.loglog(fb, t['sxx_binned_cn'][i], lw = 1, color = color,
                            label = label + ', correlated noise removed')
            if 'spar_binned' in t and t['spar_binned'].shape[1]:
                with np.errstate(divide = 'ignore', invalid = 'ignore'):
                    spar = 10 * np.log10(t['spar_binned'][i])
                    sper = 10 * np.log10(t['sper_binned'][i])
                ax_l.semilogx(fb, spar, lw = 1, color = color,
                              label = label + ' parallel (dissipation)')
                ax_l.semilogx(fb, sper, lw = 1, ls = '--', color = color,
                              label = label + ' perpendicular (amplitude)')
                drawn_l = True
    ax_s.set_xlabel('Frequency (Hz)')
    ax_s.set_ylabel('Sxx (1/Hz)')
    ax_s.set_title('Noise' if drawn_s else 'no noise spectra', fontsize = 9)
    ax_l.set_xlabel('Frequency (Hz)')
    ax_l.set_ylabel('PSD (dBc/Hz)')
    ax_l.set_title('Amplitude / dissipation noise' if drawn_l
                   else 'no loop spectra', fontsize = 9)
    for ax, drawn in ((ax_s, drawn_s), (ax_l, drawn_l)):
        ax.grid(alpha = 0.3, which = 'both')
        if drawn:
            ax.legend(fontsize = 6)
    fig.suptitle(f"{data.inv['device']}: resonator {r}")
    fig.tight_layout()
    return fig

def resonator_summary(data, sxx_run = None):
    """
    Tabulate every resonator for the device page: the latest value of
    each quantity.

    Parameters:
    data (DeviceData): the store's results.
    sxx_run (str or None): the name of a timestream run whose spectra
        give ``sxx_cn_10hz`` and ``sfactor_10hz`` where it has them
        (the latest set's); the other runs' only fill in the tones it
        lacks. Default None: the latest spectra of each tone, whatever
        the run.

    Returns:
    table (pd.DataFrame): per resonator: ``res_idx``, ``fres_mhz``,
        ``tone_set``, ``power_dbm``, ``Qr``, ``Qc``, ``Qi``, ``a_nl`` (the
        latest stream calibration's IQ fit, if any), ``photon_tau_us``,
        ``photon_height``, ``photon_rate``, ``hee_tau_us``,
        ``sxx_cn_10hz``, ``sfactor_10hz``, ``marks``.
    """
    rows = {}
    for tune, table in data.tunings:
        for _, t in table.iterrows():
            r = int(t['res_idx'])
            if r < 0:
                continue
            rows[r] = dict(res_idx = r, fres_mhz = t['fres'] / 1e6,
                           tone_set = int(t['tone_set']),
                           power_dbm = t['power_dbm'])
    for cal, tables in data.calibrations:
        for name, table in tables.items():
            for _, t in table.iterrows():
                r = int(t['res_idx'])
                if r in rows and np.isfinite(t['Qr']):
                    rows[r].update(Qr = t['Qr'], Qc = t['Qc'], Qi = t['Qi'],
                                   a_nl = t['a_nl'])
    for kind, prefix in (('photons', 'photon'), ('hee', 'hee')):
        for name, grp, table in data.results[kind]:
            for _, t in table.iterrows():
                r = int(t['res_idx'])
                if r not in rows:
                    continue
                rows[r][f'{prefix}_tau_us'] = t['tau'] * 1e6
                if kind == 'photons':
                    rows[r]['photon_height'] = t['height']
                    rows[r]['photon_rate'] = t['rate']
    # Every saved timestream's spectra, in store order, the later ones
    # overwriting; those of ``sxx_run`` moved last so they win
    spectra = [(g, g[name]) for g in data.psd_groups
               if len(g['f_reduced'][:])
               for name in sorted(n for n in g.group_keys()
                                  if n.startswith('timestream'))]
    if sxx_run is not None:
        spectra.sort(key = lambda gt: str(gt[1].attrs.get('name'))
                     == str(sxx_run))
    for g, t in spectra:
        f_reduced = np.asarray(g['f_reduced'][:], float)
        j = int(np.argmin(np.abs(f_reduced - 10.)))
        res = np.asarray(t['res_idxs'][:], int)
        key = 'sxx_reduced_cn' if 'sxx_reduced_cn' in t else 'sxx_reduced'
        sxx = np.asarray(t[key][:], float)[:, j]
        sf = np.asarray(t['sfactor_reduced'][:], float)[:, j] \
            if 'sfactor_reduced' in t else np.full(len(res), np.nan)
        for i, r in enumerate(res):
            if int(r) in rows:
                rows[int(r)][f'sxx_cn_{f_reduced[j]:g}hz'] = sxx[i]
                rows[int(r)][f'sfactor_{f_reduced[j]:g}hz'] = sf[i]
    for r, rec in rows.items():
        marks = data.marks.get(r) or data.marks.get(str(r)) or []
        rec['marks'] = '; '.join(f"{e.get('kind')}: {e.get('text', '')}"
                                 for e in marks)
    return pd.DataFrame(sorted(rows.values(), key = lambda d: d['res_idx']))

################################################################################
##################################### HTML #####################################
################################################################################
CSS = '''
body { font-family: -apple-system, "Segoe UI", Helvetica, Arial, sans-serif;
       margin: 0 auto; max-width: 1400px; padding: 0 24px 48px; color: #222; }
h1 { font-size: 1.6em; margin-top: 24px; }
h2 { font-size: 1.25em; margin-top: 36px; border-bottom: 1px solid #ccc;
     padding-bottom: 4px; }
details > summary { cursor: pointer; list-style: none; }
details > summary h2 { display: inline-block; margin-right: 12px; }
details > summary::before { content: "\\25B8 "; color: #888; }
details[open] > summary::before { content: "\\25BE "; }
tr.filters input { width: 95%; font-size: 0.9em; }
h3 { font-size: 1.05em; margin-top: 24px; }
table { border-collapse: collapse; font-size: 0.85em; margin: 8px 0; }
th, td { border: 1px solid #ddd; padding: 3px 7px; text-align: left;
         vertical-align: top; }
th { background: #f2f2f2; cursor: pointer; white-space: nowrap; }
tr:nth-child(even) td { background: #fafafa; }
img.figure { max-width: 100%; height: auto; border: 1px solid #eee; }
pre { background: #f6f6f6; padding: 8px; font-size: 0.85em;
      white-space: pre-wrap; }
.notes { background: #fffbe6; border: 1px solid #f0e2a0; padding: 8px 12px;
         white-space: pre-wrap; }
.version { margin: 8px 0; }
.version select { font-size: 1em; }
.wrap { overflow-x: auto; }
.nav a { margin-right: 16px; }
.small { font-size: 0.85em; color: #555; }
.comment { border-left: 3px solid #cfd8e3; padding: 4px 10px;
  margin: 6px 0; }
#comment-form input:not([type=hidden]), #comment-form textarea {
  display: block; width: 60%; margin: 4px 0; }
'''

JS = '''
function sortTable(th) {
  const table = th.closest('table'), tbody = table.tBodies[0];
  const col = Array.from(th.parentNode.children).indexOf(th);
  const asc = th.dataset.asc !== 'true';
  th.dataset.asc = asc;
  const rows = Array.from(tbody.rows);
  const num = v => {
    const x = parseFloat(v.replace(/,/g, ''));
    return isNaN(x) ? null : x;
  };
  rows.sort((a, b) => {
    const va = a.cells[col].innerText, vb = b.cells[col].innerText;
    const na = num(va), nb = num(vb);
    let c = (na !== null && nb !== null) ? na - nb : va.localeCompare(vb);
    return asc ? c : -c;
  });
  rows.forEach(r => tbody.appendChild(r));
}
document.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('th:not(.nosort)').forEach(
    th => th.addEventListener('click', () => sortTable(th)));
  const filterRows = (table, test) => {
    Array.from(table.tBodies[0].rows).forEach(r => {
      r.style.display = test(r) ? '' : 'none';
    });
  };
  document.querySelectorAll('input.colfilter').forEach(inp => {
    inp.addEventListener('input', () => {
      const table = inp.closest('table');
      const boxes = Array.from(table.querySelectorAll('input.colfilter'));
      filterRows(table, r => boxes.every(b => !b.value ||
        r.cells[+b.dataset.col].innerText.toLowerCase()
          .includes(b.value.toLowerCase())));
    });
  });
  document.querySelectorAll('input.rowfilter').forEach(inp => {
    inp.addEventListener('input', () => {
      const table = document.getElementById(inp.dataset.table);
      const v = inp.value.toLowerCase();
      filterRows(table, r => !v || r.innerText.toLowerCase().includes(v));
    });
  });
  document.querySelectorAll('select.versions').forEach(sel => {
    const show = () => {
      const box = sel.closest('.version');
      box.querySelectorAll('.item').forEach(d => d.style.display = 'none');
      const item = box.querySelector('#' + sel.value);
      if (item) item.style.display = 'block';
    };
    sel.addEventListener('change', show);
    show();
  });
});
'''

def _esc(text):
    """
    Escape text for HTML.

    Parameters:
    text: the value.

    Returns:
    text (str): escaped.
    """
    return html.escape('' if text is None else str(text))

def _fmt(v):
    """
    Format a table value for display.

    Parameters:
    v: the value.

    Returns:
    text (str): commas in large numbers, 4 significant digits otherwise,
        blank for NaN or None.
    """
    if v is None:
        return ''
    if isinstance(v, (bool, np.bool_)):
        return 'yes' if v else ''
    if isinstance(v, (float, np.floating)):
        if not np.isfinite(v):
            return ''
        if abs(v) >= 1000:
            return f'{v:,.0f}'
        if abs(v) < 0.01 and v != 0:
            return f'{v:.3g}'
        return f'{v:.4g}'
    if isinstance(v, (int, np.integer)):
        return f'{v:,}'
    return str(v)

def table_html(df, links = None, max_rows = None, raw = (),
               filters = False, table_id = None):
    """
    Render a DataFrame as a sortable HTML table.

    Parameters:
    df (pd.DataFrame): the table.
    links (dict or None): column: callable(value) -> href, to make that
        column's cells links. Default None.
    max_rows (int or None): rows shown. Default None: all.
    raw (sequence of str): columns holding HTML, inserted as they are.
        Default none.
    filters (bool): If True, a row of text boxes under the headers
        filters the rows by each column. Default False.
    table_id (str or None): the table's id (for a filter box elsewhere).
        Default None.

    Returns:
    text (str): the table.
    """
    links = links or {}
    head = ''.join(f'<th>{_esc(c)}</th>' for c in df.columns)
    if filters:
        head += '</tr><tr class="filters">' + ''.join(
            f'<th class="nosort"><input class="colfilter" data-col="{i}" '
            f'placeholder="filter"></th>' for i in range(len(df.columns)))
    body = []
    for _, row in (df if max_rows is None else df.head(max_rows)).iterrows():
        cells = []
        for c in df.columns:
            if c in raw:
                text = '' if row[c] is None else str(row[c])
            else:
                text = _esc(_fmt(row[c]))
                if c in links and text:
                    text = f'<a href="{_esc(links[c](row[c]))}">{text}</a>'
            cells.append(f'<td>{text}</td>')
        body.append('<tr>' + ''.join(cells) + '</tr>')
    more = '' if max_rows is None or len(df) <= max_rows else \
        f'<p class="small">{len(df) - max_rows} more rows in the CSV</p>'
    ident = f' id="{_esc(table_id)}"' if table_id else ''
    return (f'<div class="wrap"><table{ident}><thead><tr>{head}</tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table></div>{more}')

def section_html(title, inner, open_ = True, filter_table = None):
    """
    Wrap a part of a page in a collapsible section.

    Parameters:
    title (str): the heading.
    inner (str): the HTML inside.
    open_ (bool): If True (default), open at first; else collapsed.
    filter_table (str or None): the id of a table inside to filter with
        a text box in the heading. Default None.

    Returns:
    text (str): the section.
    """
    box = (f' <input class="rowfilter" data-table="{_esc(filter_table)}" '
           'placeholder="filter rows" onclick="event.stopPropagation()">'
           if filter_table else '')
    return (f'<details{" open" if open_ else ""}><summary><h2>{_esc(title)}'
            f'</h2>{box}</summary>{inner}</details>')

def page_html(title, body, depth, subtitle = ''):
    """
    Wrap a page body in the site's HTML.

    Parameters:
    title (str): page title.
    body (str): HTML body.
    depth (int): folders below the site root (for the link to the
        site's folder, whose ``index.php`` a web server serves).
    subtitle (str): line under the title. Default ''.

    Returns:
    text (str): the page.
    """
    up = '../' * depth
    return (f'<!DOCTYPE html><html><head><meta charset="utf-8">'
            f'<title>{_esc(title)}</title><style>{CSS}</style>'
            f'<script>{JS}</script></head><body>'
            f'<div class="nav"><a href="{up or "./"}">All devices</a></div>'
            f'<h1>{_esc(title)}</h1>'
            + (f'<p class="small">{_esc(subtitle)}</p>' if subtitle else '')
            + body + '</body></html>')

def work_by_html(text):
    """
    Return the line naming who did the work, for the top of a device
    page.

    Parameters:
    text (str): who did the work.

    Returns:
    html (str): the paragraph.
    """
    return f'<p class="small work-by">Work by: {_esc(text)}</p>'

def fill_work_by(out_dir, text):
    """
    Write who did the work into a device's existing report (its
    manifest and the top of its page) without rebuilding it. The
    site's top page is rebuilt by ``build_index``.

    Parameters:
    out_dir (str): the device's folder in the site (``device_dir``).
    text (str): who did the work; blank removes the line.

    Returns:
    out_dir (str): the folder changed.

    Raises:
    FileNotFoundError: if the folder holds no report.
    """
    manifest_path = os.path.join(out_dir, MANIFEST)
    page_path = os.path.join(out_dir, 'index.html')
    if not os.path.exists(manifest_path):
        raise FileNotFoundError(f'{out_dir} holds no report')
    text = (text or '').strip()
    with open(manifest_path, encoding = 'utf-8') as fh:
        inv = json.load(fh)
    attrs = inv.setdefault('attrs', {})
    if text:
        attrs[dl.WORK_BY] = text
    else:
        attrs.pop(dl.WORK_BY, None)
    with open(manifest_path, 'w', encoding = 'utf-8') as fh:
        json.dump(inv, fh, indent = 1, default = str)
    if os.path.exists(page_path):
        with open(page_path, encoding = 'utf-8') as fh:
            page = fh.read()
        page = re.sub(r'<p class="small work-by">.*?</p>', '', page,
                      count = 1, flags = re.S)
        if text:
            page = page.replace('</h1>', '</h1>' + work_by_html(text), 1)
        with open(page_path, 'w', encoding = 'utf-8') as fh:
            fh.write(page)
    return out_dir

def versions_html(key, items, attachments = None):
    """
    Render a set of versions of one section: a dropdown (latest first)
    showing one version's notes, text, figures and attached files.

    Parameters:
    key (str): section id.
    items (list of dict): per version, newest first: ``label``,
        ``notes``, ``text`` (shown in a pre), ``figures`` (list of
        relative paths), ``path`` (store path), ``files`` (list of
        (label, relative path) links).
    attachments (dict or None): ``load_attachments`` output; each
        version's attachments (by its store path) are linked. Default
        None.

    Returns:
    text (str): the HTML, or a note when there are none.
    """
    if not items:
        return '<p class="small">none</p>'
    for it in items:
        it['files'] = list(it.get('files', [])) + attachment_links(
            attachments or {}, it.get('path') or '')
    options = ''.join(
        f'<option value="{key}_{i}">{_esc(it["label"])}</option>'
        for i, it in enumerate(items))
    blocks = []
    for i, it in enumerate(items):
        parts = [f'<p class="small">store path: <code>{_esc(it.get("path"))}'
                 '</code></p>']
        if it.get('notes'):
            parts.append(f'<div class="notes">{_esc(it["notes"])}</div>')
        if it.get('text'):
            parts.append(f'<pre>{_esc(it["text"])}</pre>')
        for fig in it.get('figures', []):
            parts.append(f'<p><img class="figure" src="{_esc(fig)}"></p>')
        if it.get('files'):
            parts.append('<p class="small">' + ' | '.join(
                f'<a href="{_esc(p)}">{_esc(label)}</a>'
                for label, p in it['files']) + '</p>')
        blocks.append(f'<div class="item" id="{key}_{i}">'
                      + ''.join(parts) + '</div>')
    return (f'<div class="version"><select class="versions">{options}'
            f'</select>{"".join(blocks)}</div>')

################################################################################
#################################### Build #####################################
################################################################################
def _rel(path, start):
    """
    Return a path relative to a folder, with forward slashes.

    Parameters:
    path (str): the file.
    start (str): the folder.

    Returns:
    rel (str): the relative path.
    """
    return os.path.relpath(path, start).replace(os.sep, '/')

def setup_note(setup, base):
    """
    Say how an item's setup (readout chain) differs from the store's,
    for the What was taken table.

    Parameters:
    setup (dict or None): the item's ``setup`` attribute.
    base (dict or None): the store's.

    Returns:
    text (str): '; setup: key value, ...' over the keys that differ
        (empty or None values count as unset), or ''.
    """
    if not setup:
        return ''
    base = base or {}
    clean = lambda v: '' if v is None else str(v)
    diff = {k: v for k, v in dict(setup).items()
            if clean(v) != clean(base.get(k)) and clean(v)}
    diff.update({k: '(unset)' for k in base
                 if clean(base[k]) and not clean(dict(setup).get(k))})
    return f'; setup: {dl.setup_text(diff)}' if diff else ''

def _taken_table(inv, attachments = None):
    """
    Tabulate what a store holds, one row per item.

    Parameters:
    inv (dict): ``inventory`` output.
    attachments (dict or None): ``load_attachments`` output: each item's
        attached files, as links. Default None.

    Returns:
    table (pd.DataFrame): ``kind``, ``path`` (in the store), ``created``,
        ``name``, ``notes``, ``details``, ``attached`` (HTML links).
    """
    rows = []
    base = (inv.get('attrs') or {}).get('setup')
    for s in inv['sweeps']:
        rows.append(dict(kind = 'VNA sweep', path = s['path'],
                         created = s['created'], name = '',
                         notes = s['notes'],
                         details = f"{s.get('n_resonators') or 0} resonators"
                         + (', fit' if s['fit'] else ', not fit')
                         + (f", {s['amplitude']} dBm"
                            if s.get('amplitude') is not None else '')
                         + setup_note(s.get('setup'), base)))
    for t in inv['tunings']:
        rows.append(dict(kind = 'Tuning', path = t['path'],
                         created = t['created'], name = '', notes = t['notes'],
                         details = f"{t.get('n_tones') or 0} tones in "
                         f"{t.get('n_tone_sets') or 0} tone sets; powers "
                         f"{t['powers'][0] if t['powers'] else ''} to "
                         f"{t['powers'][-1] if t['powers'] else ''} dBm"
                         + setup_note(t.get('setup'), base)))
        for c in t['cal_sweeps']:
            rows.append(dict(kind = 'Calibration sweep', path = c['path'],
                             created = c['created'],
                             name = f"tone set {c['tone_set']}",
                             notes = c['notes'],
                             details = f"{c.get('n_tones') or 0} tones"))
    for r in inv['runs']:
        s = r.get('settings', {})
        fs = {k: v for k, v in s.items() if k.startswith('full_scale')}
        att = {k: v for k, v in s.items() if k.startswith('adc_att')}
        rows.append(dict(
            kind = 'Timestream run', path = ', '.join(r.get('paths') or []),
            created = r['created'], name = r['name'], notes = r['notes'],
            details = f"{r.get('duration_s')} s, decimation "
            f"{r.get('dec_stage')}, tone set {r.get('tone_set')}, "
            f"{r.get('n_tones') or ''} tones"
            + (f", sample rate {float(s['sample_freq']):,.0f} Hz"
               if s.get('sample_freq') else '')
            + ''.join(f', full scale {float(v):.1f} dBm' for v in fs.values())
            + ''.join(f', ADC attenuation {float(v):g} dB'
                      for v in att.values())
            + setup_note(r.get('setup'), base)))
    for c in inv['calibrations']:
        rows.append(dict(kind = 'Stream calibration', path = c['path'],
                         created = c['created'], name = '', notes = c['notes'],
                         details = '; '.join(
                             f"{s['name']}: {s.get('n_calibrated')} of "
                             f"{s.get('n_tones')} tones" for s in c['sets'])))
    for kind, label in (('photons', 'Photons'),
        ('hee', 'High-energy events')):
        for it in inv[kind]:
            rows.append(dict(
                kind = label, path = it['path'], created = it['created'],
                name = it['name'], notes = it['notes'],
                details = (f"{it.get('n_tones')} tones, {it.get('n_bad')} "
                           f"bad, median lifetime "
                           f"{_fmt(it.get('tau_median_us'))} us"
                           if it['complete'] else 'saved incompletely')))
    for g in inv['psds']:
        rows.append(dict(kind = 'Noise spectra', path = g['path'],
                         created = g['created'], name = g.get('name') or '',
                         notes = g['notes'],
                         details = f"{g['n_timestreams']} timestreams"
                         if g['complete'] else 'saved incompletely'))
    for r, it in zip(rows, _taken_items(inv)):
        if not it.get('in_report', True):
            r['details'] += '; left out of the report'
    for r in rows:
        item = r['path'].split(', ')[0]
        r['attached'] = ' | '.join(
            f'<a href="{_esc(p)}">{_esc(n)}</a>'
            for n, p in attachment_links(attachments or {}, item))
    return pd.DataFrame(rows, columns = ['kind', 'path', 'created', 'name',
                                         'notes', 'details', 'attached'])

def _taken_items(inv):
    """
    List the inventory's items in the order ``_taken_table`` tabulates
    them.

    Parameters:
    inv (dict): ``inventory`` output.

    Returns:
    items (list of dict): the items.
    """
    items = []
    for s in inv['sweeps']:
        items.append(s)
    for t in inv['tunings']:
        items.append(t)
        items.extend(t['cal_sweeps'])
    items.extend(inv['runs'])
    items.extend(inv['calibrations'])
    items.extend(inv['photons'])
    items.extend(inv['hee'])
    items.extend(inv['psds'])
    return items

LEFT_OUT = 'left out of the report (unchecked in the store browser)'

def _left_out(label, notes, path):
    """
    Make the one-line entry of an item left out of the report.

    Parameters:
    label, notes, path (str): the item's label, notes and store path.

    Returns:
    item (dict): a ``versions_html`` item with no figures or files.
    """
    return dict(label = label, notes = notes, path = path, text = LEFT_OUT,
                figures = [], files = [])

def _percentiles(values):
    """
    Return the 1st and 99th percentiles of some values, with their count.

    Parameters:
    values (array-like): the values; NaN is skipped.

    Returns:
    out (list or None): [p1, p99, n], or None if there are no values.
    """
    v = np.asarray(values, dtype = float)
    v = v[np.isfinite(v)]
    if not len(v):
        return None
    lo, hi = np.percentile(v, [1, 99])
    return [float(lo), float(hi), int(len(v))]

def store_overview(inv, summary):
    """
    Reduce a store's results to the numbers the site's top page lists.

    Parameters:
    inv (dict): ``inventory`` output.
    summary (pd.DataFrame): ``resonator_summary`` output.

    Returns:
    overview (dict): ``notes`` (the store's), ``n_tones`` and
        ``n_resonators`` (the latest tuning with tones and the resonator
        list it used), ``yield_pct`` (their ratio), ``fres_mhz`` ([min,
        max] of the tuned tones), and the 1st and 99th percentiles with
        the count ([p1, p99, n], or None) of ``photon_tau_us``,
        ``hee_tau_us`` and ``sxx_cn_10hz`` (Sxx after common-mode
        removal at the reduced frequency nearest 10 Hz).
    """
    out = dict(notes = str((inv.get('attrs') or {}).get('notes') or ''),
               n_tones = None, n_resonators = None, yield_pct = None,
               fres_mhz = None, photon_tau_us = None, hee_tau_us = None,
               sxx_cn_10hz = None)
    tuned = [t for t in inv.get('tunings', []) if t.get('n_tones')]
    if tuned:
        t = tuned[-1]
        out['n_tones'] = int(t['n_tones'])
        res = str(t.get('resonators') or '')
        for s in inv.get('sweeps', []):
            if res == dl.set_name(s['k'], 'resonators') and \
                    s.get('n_resonators'):
                out['n_resonators'] = int(s['n_resonators'])
                out['yield_pct'] = 100. * out['n_tones'] / out['n_resonators']
    if len(summary) and 'fres_mhz' in summary:
        f = summary['fres_mhz'].to_numpy(dtype = float)
        f = f[np.isfinite(f)]
        if len(f):
            out['fres_mhz'] = [float(f.min()), float(f.max())]
    for key in ('photon_tau_us', 'hee_tau_us'):
        if key in summary:
            out[key] = _percentiles(summary[key])
    sxx = [c for c in summary.columns if c.startswith('sxx_cn_')]
    if sxx:
        out['sxx_cn_10hz'] = _percentiles(summary[sxx[0]])
    return out

def _range_text(r, fmt = '{:.3g}', unit = ''):
    """
    Format a [p1, p99, n] range (``_percentiles``) or [lo, hi] pair.

    Parameters:
    r (list or None): the range.
    fmt (str): number format. Default '{:.3g}'.
    unit (str): unit appended. Default ''.

    Returns:
    text (str): 'lo to hi unit (n res)', or ''.
    """
    if not r:
        return ''
    text = f'{fmt.format(r[0])} to {fmt.format(r[1])}{unit}'
    return text + (f' ({r[2]} res)' if len(r) > 2 else '')

def frequency_schedule(root, inv):
    """
    Describe the frequency schedule of the latest tuning: the range of
    the resonances found (``fres_all`` of the resonator list it used),
    the outliers among them (``report.outliers``), and the CRS
    modules, analog bank and NCO frequencies it ran with.

    Parameters:
    root (zarr.Group): the store root.
    inv (dict): ``inventory`` output.

    Returns:
    schedule (dict or None): ``fmin_mhz``, ``fmax_mhz`` (the range
        without the outliers), ``n`` (resonances), ``outliers_mhz``
        (list), ``modules`` (list of int), ``bank`` ('low', 'high'
        or None), ``ncos_mhz`` (list, per tone set); None when no
        tuning has a resonator list.
    """
    tuned = [t for t in inv.get('tunings', []) if t.get('resonators')]
    if not tuned:
        return None
    t = tuned[-1]
    res = root.get(str(t['resonators']))
    if res is None or 'fres_all' not in res:
        return None
    f = np.asarray(res['fres_all'][:], dtype = float) / 1e6
    f = f[np.isfinite(f)]
    out = dict(fmin_mhz = None, fmax_mhz = None, n = int(len(f)),
               outliers_mhz = [], modules = [], bank = None, ncos_mhz = [])
    if len(f):
        odd = treport.outliers(f) if len(f) > 3 else np.zeros(len(f), bool)
        keep = f[~odd] if (~odd).any() else f
        out.update(fmin_mhz = float(keep.min()), fmax_mhz = float(keep.max()),
                   outliers_mhz = sorted(float(v) for v in f[odd]))
    cfg = root.get(f"{t['path']}/crs_config")
    config = dict(cfg.attrs.get('config') or {}) if cfg is not None else {}
    if config:
        out['modules'] = [int(m) for m in config.get('modules') or []]
        if config.get('analog_bank_high') is not None:
            out['bank'] = 'high' if config['analog_bank_high'] else 'low'
        ncos = []
        for ts_ in config.get('tone_sets') or []:
            ncos.extend(float(v) / 1e6 for v in dict(ts_).values())
        if not ncos and config.get('ncos'):
            ncos = [float(v) / 1e6 for v in dict(config['ncos']).values()]
        out['ncos_mhz'] = ncos
    return out

def schedule_text(schedule):
    """
    Write a frequency schedule as one short cell for the site's top
    page: the range of the resonances found to 1 MHz, outliers apart,
    and how many outliers lie below and above it. The CRS setup the
    schedule holds (modules, bank, NCOs) is not written.

    Parameters:
    schedule (dict or None): ``frequency_schedule`` output.

    Returns:
    text (str): e.g. '314 to 2056 MHz, 4 above', or ''.
    """
    if not schedule or schedule.get('fmin_mhz') is None:
        return ''
    text = f"{schedule['fmin_mhz']:.0f} to {schedule['fmax_mhz']:.0f} MHz"
    odd = [float(v) for v in schedule.get('outliers_mhz') or []]
    below = sum(v < schedule['fmin_mhz'] for v in odd)
    above = sum(v > schedule['fmax_mhz'] for v in odd)
    if below:
        text += f', {below} below'
    if above:
        text += f', {above} above'
    return text

def store_location(root):
    """
    Return a store's path on disk, for worker processes to open it.

    Parameters:
    root (zarr.Group): the store root.

    Returns:
    path (str): the path, or '' for a store without one (in memory).
    """
    path = getattr(root.store, 'root', None)
    return '' if path is None else str(path)

def _write_res_pages(store, inv, res_idxs, all_res_idxs, out_dir):
    """
    Write some resonators' pages and figures (in a worker, or here).

    Parameters:
    store (str or zarr.Group): the store's path, or its root (here
        only).
    inv (dict): ``inventory`` output.
    res_idxs (list of int): the resonators to write.
    all_res_idxs (list of int): every resonator with a page, in order
        (the Previous / Next links).
    out_dir (str): the device's folder.

    Returns:
    n (int): pages written.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    data = DeviceData(dl.open_store(store), inv) if isinstance(store, str) \
        else DeviceData(store, inv)
    device = inv['device']
    for r in res_idxs:
        n = all_res_idxs.index(r)
        fig = resonator_figure(data, r)
        fig_path = os.path.join(out_dir, 'res', f'res{r}.png')
        fig.savefig(fig_path, dpi = RES_DPI)
        plt.close(fig)
        parts = [f'<p><img class="figure" src="res{r}.png"></p>']
        for title, table in resonator_rows(data, r):
            parts.append(f'<h3>{_esc(title)}</h3>' + table_html(table))
        prev_r = all_res_idxs[n - 1] if n else None
        next_r = all_res_idxs[n + 1] if n + 1 < len(all_res_idxs) else None
        nav = '<p class="nav"><a href="../index.html">Device page</a>' + (
            f'<a href="res{prev_r}.html">Previous ({prev_r})</a>'
            if prev_r is not None else '') + (
            f'<a href="res{next_r}.html">Next ({next_r})</a>'
            if next_r is not None else '') + '</p>'
        with open(os.path.join(out_dir, 'res', f'res{r}.html'), 'w',
                  encoding = 'utf-8') as fh:
            fh.write(page_html(f'{device}: resonator {r}', nav
                               + ''.join(parts), 3, inv['cooldown']))
    return len(res_idxs)

def build_device(root, out_root, force = False, dpi = FIG_DPI,
                 res_pages = True, progress = None, sxx_run = None,
                 n_workers = 1):
    """
    Build (or refresh) one device's pages.

    Parameters:
    root (zarr.Group or str): the store.
    out_root (str): the site's root folder.
    force (bool): If True, rebuild even if the store has not changed
        since the last build. Default False.
    dpi (int): figure resolution. Default ``FIG_DPI``.
    res_pages (bool): If True (default), one page per resonator.
    n_workers (int or None): processes writing the resonator pages
        (``parallel.map_in_pool``; a store in memory is written here).
        Default 1: here.
    progress (callable or None): called with progress text. Default
        None.
    sxx_run (str or None): the timestream run the resonator table's
        Sxx and s-factor come from where it has them
        (``resonator_summary``). Default None: the latest spectra.

    Returns:
    out_dir (str or None): the device's folder, or None if it was up to
        date and left alone.
    """
    import matplotlib
    import matplotlib.pyplot as plt
    root = dl.open_store(root)
    say = progress or (lambda text: None)
    inv = inventory(root)
    out_dir = os.path.join(os.path.expanduser(out_root), inv['cooldown'],
                           inv['device'])
    manifest_path = os.path.join(out_dir, MANIFEST)
    attachments = load_attachments(out_dir)
    if not force and os.path.exists(manifest_path):
        with open(manifest_path, encoding = 'utf-8') as fh:
            old = json.load(fh)
        if old.get('newest') == inv['newest'] and \
                old.get('res_pages', False) == res_pages and \
                old.get('attachments', {}) == attachments:
            say(f"{inv['cooldown']}/{inv['device']}: up to date")
            return None
    say(f"{inv['cooldown']}/{inv['device']}: building")
    fig_dir, data_dir = os.path.join(out_dir, 'figures'), \
        os.path.join(out_dir, 'data')
    for d in (fig_dir, data_dir, os.path.join(out_dir, 'res')):
        os.makedirs(d, exist_ok = True)
    backend = matplotlib.get_backend()
    matplotlib.use('Agg')
    device = inv['device']
    # The existing reports: VNA sweeps and timestream results
    vna_paths = {}
    for s in inv['sweeps']:
        if s['fit'] and s.get('in_report', True):
            vna_paths[s['k']] = vreport.export_vna_report(root, fig_dir,
                                                         s['k'], dpi = dpi)
    ts_paths = treport.export_timestream_report(
        root, fig_dir, dpi = dpi,
        photons = [it['name'] for it in inv['photons']
                   if it.get('in_report', True)],
        hee = [it['name'] for it in inv['hee'] if it.get('in_report', True)],
        groups = [g['p'] for g in inv['psds'] if g.get('in_report', True)])
    for key, path in list(ts_paths.items()) + [
            (f'vna{k}_{n}', p) for k, d in vna_paths.items()
            for n, p in d.items()]:
        # Tables and text go to data/
        if path.endswith(('.csv', '.txt')):
            new = os.path.join(data_dir, os.path.basename(path))
            os.replace(path, new)
            if key in ts_paths:
                ts_paths[key] = new
            else:
                k, n = key.split('_', 1)
                vna_paths[int(k[3:])][n] = new
    data = DeviceData(root, inv)
    # Tuning summaries
    tune_figs = {}
    for tune, table in data.tunings:
        fig = plot_tuning_summary(
            table, f"{device} {tune.basename}: {len(table)} tones")
        path = os.path.join(fig_dir, f'{device}_{tune.basename}_tuning.png')
        fig.savefig(path, dpi = dpi, bbox_inches = 'tight')
        plt.close(fig)
        csv = os.path.join(data_dir, f'{device}_{tune.basename}_tuning.csv')
        table.to_csv(csv, index = False)
        tune_figs[tune.basename] = (path, csv)
    summary = resonator_summary(data, sxx_run = sxx_run)
    overview = store_overview(inv, summary)
    overview['schedule'] = frequency_schedule(root, inv)
    overview['sxx_run'] = None if sxx_run is None else str(sxx_run)
    summary_csv = os.path.join(data_dir, f'{device}_resonators.csv')
    summary.to_csv(summary_csv, index = False)
    # Resonator pages
    res_idxs = data.res_idxs() if res_pages else []
    if res_idxs:
        store = store_location(root)
        nw = resolve_workers(n_workers) if store else 1
        chunks = [list(map(int, c)) for c in np.array_split(
            np.array(res_idxs), min(nw * 4, len(res_idxs)))]
        say(f'  resonator pages: {len(res_idxs)}'
            + (f' over {nw} processes' if nw > 1 else ''))
        if nw > 1:
            map_in_pool(_write_res_pages,
                        ((k, (store, inv, c, list(map(int, res_idxs)),
                              out_dir)) for k, c in enumerate(chunks)),
                        nw)
        else:
            for n, c in enumerate(chunks):
                say(f'  resonator pages: {sum(map(len, chunks[:n]))} of '
                    f'{len(res_idxs)}')
                _write_res_pages(store or root, inv, c,
                                 list(map(int, res_idxs)), out_dir)
    if backend != matplotlib.get_backend():
        matplotlib.use(backend)
    # The device page
    body = []
    notes_path = os.path.join(out_dir, NOTES_FILE)
    if os.path.exists(notes_path):
        with open(notes_path, encoding = 'utf-8') as fh:
            body.append(f'<div class="notes">{_esc(fh.read())}</div>')
    attrs = inv['attrs']
    if attrs.get(dl.WORK_BY):
        body.insert(0, work_by_html(attrs[dl.WORK_BY]))
    setup = dl.setup_text(attrs.get('setup'))
    body.append('<p class="small">store '
                f"<code>{_esc(inv['store_name'] or device)}.zarr</code>; "
                f"created {_esc(attrs.get('created'))}; CRS "
                f"{_esc(attrs.get('crs_serial_number'))}, firmware "
                f"{_esc(attrs.get('firmware_version'))}"
                + (f'; setup {_esc(setup)}' if setup else '')
                + f"; page built {_esc(dl._now())}</p>")
    if attrs.get('notes'):
        body.append(f'<div class="notes">{_esc(attrs["notes"])}</div>')
    if attachment_links(attachments, ''):
        body.append('<p class="small">attached: ' + ' | '.join(
            f'<a href="{_esc(p)}">{_esc(n)}</a>'
            for n, p in attachment_links(attachments, '')) + '</p>')
    # The comments first: the page's own conversation, above the data
    body.append(comments_section(out_dir, inv['cooldown'],
                                 os.path.basename(out_dir)))
    body.append(section_html('What was taken', table_html(
        _taken_table(inv, attachments), raw = ('attached',))))
    items = []
    for s in reversed(inv['sweeps']):
        if not s.get('in_report', True):
            items.append(_left_out(f"vna{s['k']} ({s['created']})",
                                   s['notes'], s['path']))
        elif s['k'] in vna_paths:
            p = vna_paths[s['k']]
            items.append(dict(
                label = f"vna{s['k']} ({s['created']})", notes = s['notes'],
                text = s['report'], path = s['path'],
                figures = [_rel(p['plot'], out_dir)],
                files = [('resonators CSV', _rel(p['csv'], out_dir)),
                         ('gain CSV', _rel(p['gain_csv'], out_dir)),
                         ('sweep info', _rel(p['info'], out_dir))]))
    body.append(section_html('VNA sweeps', versions_html('vna', items,
                                                          attachments)))
    items = []
    for t in reversed(inv['tunings']):
        name = t['path'].split('/')[-1]
        if not t.get('in_report', True):
            items.append(_left_out(f"{name} ({t['created']})", t['notes'],
                                   t['path']))
        elif name in tune_figs:
            png, csv = tune_figs[name]
            items.append(dict(
                label = f"{name} ({t['created']})", notes = t['notes'],
                path = t['path'], figures = [_rel(png, out_dir)],
                text = f"{t.get('n_tones')} tones in {t.get('n_tone_sets')} "
                f"tone sets; power choice {json.dumps(t['power_choice'])}",
                files = [('tones CSV', _rel(csv, out_dir))]))
    body.append(section_html('Tuning', versions_html('tune', items,
                                                      attachments)))
    items = []
    for c in reversed(inv['calibrations']):
        if not c.get('in_report', True):
            items.append(_left_out(f"{c['path']} ({c['created']})",
                                   c['notes'], c['path']))
            continue
        figs, files = [], []
        for cal, tables in data.calibrations:
            if cal.path != c['path']:
                continue
            for name, table in tables.items():
                fig = plot_calibration_fits(
                    table, f"{device} {cal.path}/{name}: IQ fits of "
                    f'{int(np.isfinite(table["Qr"]).sum())} tones')
                png = os.path.join(fig_dir,
                                   f'{device}_{cal.basename}_{name}_fit.png')
                fig.savefig(png, dpi = dpi, bbox_inches = 'tight')
                plt.close(fig)
                csv = os.path.join(data_dir,
                                   f'{device}_{cal.basename}_{name}_fit.csv')
                table.to_csv(csv, index = False)
                figs.append(_rel(png, out_dir))
                files.append((f'{name} fits CSV', _rel(csv, out_dir)))
        items.append(dict(
            label = f"{c['path']} ({c['created']})", notes = c['notes'],
            path = c['path'], figures = figs, files = files,
            text = 'selection: ' + json.dumps(c['selection']) + '; '
            + '; '.join(f"{s['name']}: {s.get('n_calibrated')} of "
                        f"{s.get('n_tones')} tones calibrated"
                        + (', IQ fit' if s.get('iq_fit') else '')
                        for s in c['sets'])))
    body.append(section_html('Stream calibrations',
                             versions_html('cal', items, attachments)))
    for kind, title in (('photons', 'Photons'),
        ('hee', 'High-energy events')):
        items = []
        for it in reversed(inv[kind]):
            key = f"lifetime_{kind}_{it['name']}"
            if not it.get('in_report', True):
                items.append(_left_out(f"{it['name']} ({it['created']})",
                                       it['notes'], it['path']))
                continue
            if key not in ts_paths:
                continue
            figs = [_rel(ts_paths[key], out_dir)]
            mkey = f"multiplicity_{kind}_{it['name']}"
            if mkey in ts_paths:
                figs.append(_rel(ts_paths[mkey], out_dir))
            files = [('per-tone CSV', _rel(ts_paths[key + '_csv'], out_dir))]
            if mkey + '_csv' in ts_paths:
                files.append(('events CSV',
                              _rel(ts_paths[mkey + '_csv'], out_dir)))
            items.append(dict(
                label = f"{it['name']} ({it['created']})",
                notes = it['notes'], path = it['path'], figures = figs,
                text = f"run {it['run']} ({it['run_name']}), tone set "
                f"{it['tone_set']}, calibration {it['calibration']}; "
                f"{it.get('n_tones')} tones, {it.get('n_bad')} marked bad; "
                f"median lifetime {_fmt(it.get('tau_median_us'))} us"
                + (f", median rate {_fmt(it.get('rate_median'))} /s"
                   if it.get('rate_median') is not None else ''),
                files = files))
        body.append(section_html(title, versions_html(kind, items,
                                                       attachments)))
    items = []
    for g in reversed(inv['psds']):
        key = f"psd{g['p']}"
        if not g.get('in_report', True):
            items.append(_left_out(
                f"ts_psd{g['p']} {g.get('name') or ''} ({g['created']})",
                g['notes'], g['path']))
            continue
        if key not in ts_paths:
            continue
        items.append(dict(
            label = f"ts_psd{g['p']} {g.get('name') or ''} ({g['created']})",
            notes = g['notes'], path = g['path'],
            figures = [_rel(ts_paths[key], out_dir)],
            text = g.get('description', ''),
            files = [('per-tone CSV', _rel(ts_paths[key + '_csv'], out_dir)),
                     ('spectra CSV',
                      _rel(ts_paths[key + '_spectra_csv'], out_dir))]))
    body.append(section_html('Noise spectra', versions_html('psd', items,
                                                             attachments)))
    if 'info' in ts_paths:
        info = _rel(ts_paths['info'], out_dir)
        body.append(f'<p class="small"><a href="{info}">timestream results '
                    'text</a></p>')
    rows = [dict(item = item or '(store)', file = n, link = p)
            for item, files in sorted(attachments.items())
            for n, p in attachment_links(attachments, item)]
    if rows:
        body.append(section_html('Attached files', table_html(
            pd.DataFrame(rows), links = {'link': lambda p: p}),
            open_ = False))
    links = {'res_idx': (lambda r: f'res/res{int(r)}.html')} if res_pages \
        else {}
    body.append(section_html(
        f'Resonators ({len(summary)})',
        f'<p class="small"><a href="{_rel(summary_csv, out_dir)}">CSV</a>; '
        'click a column to sort; type to filter'
        + (f'; Sxx and s-factor from run {_esc(str(sxx_run))} where it '
           'has them, else the latest spectra' if sxx_run is not None
           else '; Sxx and s-factor from the latest spectra of each tone')
        + '</p>'
        + table_html(summary, links = links, table_id = 'resonators'),
        open_ = False, filter_table = 'resonators'))
    marks = pd.DataFrame(inv['marks'], columns = [
        'res_idx', 'kind', 'scope', 'text', 'stage', 'tuning', 'created'])
    body.append(section_html(f'Tone marks ({len(marks)})',
                             table_html(marks, links = links,
                                        table_id = 'marks'),
                             open_ = False, filter_table = 'marks'))
    with open(os.path.join(out_dir, 'index.html'), 'w',
              encoding = 'utf-8') as fh:
        fh.write(page_html(f"{device} ({inv['cooldown']})", ''.join(body),
                           2))
    manifest = dict(inv, built = dl._now(), res_pages = res_pages,
                    attachments = attachments, overview = overview,
                    store = str(getattr(root.store, 'root', root.store)))
    manifest['index_row'] = index_row(manifest, inv['cooldown'],
                                      os.path.basename(out_dir),
                                      load_comments(out_dir))
    with open(manifest_path, 'w', encoding = 'utf-8') as fh:
        json.dump(manifest, fh, indent = 1, default = str)
    return out_dir

INDEX_COLUMNS = [
    'Cooldown', 'Device', 'Die', 'Suffix', 'Work by', 'Notes', 'Comments',
    'Resonators', 'Frequency schedule',
    'Photon lifetime [us]', 'HEE lifetime [us]', 'Sxx at 10 Hz [1/Hz]',
    'Report generated']
INDEX_INTRO = (
    'Type under a heading to filter; click a heading to sort. The devices '
    'listed are the device folders present in this site. Resonators: '
    'found in the VNA sweep the latest tuning used (the other columns '
    'count the tones each measurement has). Frequency schedule: the '
    'range of the resonances found, with the outliers below and above '
    'it counted. Photon and HEE lifetimes and Sxx (after '
    'common-mode removal, at 10 Hz): the 1st to 99th percentile over the '
    'tones of the latest result (count). Comments: added on the device '
    'pages here.')
INDEX_PHP = 'index.php'

def index_row(manifest, cooldown, folder, comments = ()):
    """
    Make a device's row of the site's top page from its manifest: the
    text of every ``INDEX_COLUMNS`` column (the Comments one as of
    now; ``index.php`` reads the live comments itself).

    Parameters:
    manifest (dict): the device's ``manifest.json`` contents.
    cooldown (str): its cooldown folder.
    folder (str): its device folder.
    comments (list of dict): ``load_comments`` output. Default none.

    Returns:
    row (dict): column name: text ('' where unknown).
    """
    wafer, die = device_parts(manifest.get('device', folder))
    ov = manifest.get('overview') or {}
    attrs = manifest.get('attrs') or {}
    sched = ov.get('schedule') or {}
    n_res = sched.get('n') or ov.get('n_resonators')
    resonators = '' if n_res is None else str(int(n_res))
    schedule = schedule_text(ov.get('schedule')) or _range_text(
        ov.get('fres_mhz'), '{:.1f}', ' MHz')
    return {
        'Cooldown': cooldown, 'Device': wafer, 'Die': die,
        'Suffix': manifest.get('suffix', ''),
        'Work by': attrs.get(dl.WORK_BY) or '',
        'Notes': ov.get('notes') or attrs.get('notes') or '',
        'Comments': comments_summary(list(comments)),
        'Resonators': resonators,
        'Frequency schedule': schedule,
        'Photon lifetime [us]': _range_text(ov.get('photon_tau_us'),
                                            '{:.0f}'),
        'HEE lifetime [us]': _range_text(ov.get('hee_tau_us'), '{:.0f}'),
        'Sxx at 10 Hz [1/Hz]': _range_text(ov.get('sxx_cn_10hz'), '{:.2g}'),
        'Report generated': manifest.get('built', '')}

def index_rows(out_root, refresh = False):
    """
    Read every device manifest under the site and return the top page's
    rows, saving a row into each manifest that has none (or into all of
    them with ``refresh``), so ``index.php`` can list the devices
    without computing anything.

    Parameters:
    out_root (str): the site's root folder.
    refresh (bool): If True, compute every row again from its manifest
        and save it. Default False: only manifests without a row.

    Returns:
    rows (list of dict): ``index_row`` of each device, with ``_link``
        (its page, relative to the site root), in folder order.
    """
    out_root = os.path.expanduser(out_root)
    rows = []
    for cooldown in sorted(os.listdir(out_root)):
        cdir = os.path.join(out_root, cooldown)
        if not os.path.isdir(cdir):
            continue
        for device in sorted(os.listdir(cdir)):
            m = os.path.join(cdir, device, MANIFEST)
            if not os.path.exists(m):
                continue
            with open(m, encoding = 'utf-8') as fh:
                manifest = json.load(fh)
            if refresh or not isinstance(manifest.get('index_row'), dict):
                manifest['index_row'] = index_row(
                    manifest, cooldown, device,
                    load_comments(os.path.join(cdir, device)))
                with open(m, 'w', encoding = 'utf-8') as fh:
                    json.dump(manifest, fh, indent = 1, default = str)
            rows.append(dict(manifest['index_row'],
                             _link = f'{cooldown}/{device}/index.html'))
    return rows

PHP_TEMPLATE = r"""<?php
// Top page of the report site, written by citkid (7_report_site.py):
// lists the device folders present here from the row each one's
// manifest.json carries. Nothing is computed; upload a device folder and
// it is listed on the next visit.
header('Content-Type: text/html; charset=utf-8');
$columns = json_decode('@@COLUMNS@@', true);
$rows = array();
foreach (glob(__DIR__ . '/*/*/manifest.json') as $path) {
    $m = json_decode(file_get_contents($path), true);
    if (!is_array($m)) {
        continue;
    }
    $device = basename(dirname($path));
    $cooldown = basename(dirname(dirname($path)));
    $row = (isset($m['index_row']) && is_array($m['index_row']))
        ? $m['index_row']
        : array('Cooldown' => $cooldown, 'Device' => $device);
    $row['_link'] = rawurlencode($cooldown) . '/' . rawurlencode($device)
        . '/index.html';
    $row['Comments'] = comments_summary(dirname($path) . '/comments.json');
    $rows[] = $row;
}
usort($rows, function ($a, $b) {
    return strcmp($a['Cooldown'] . ' ' . $a['Device'],
                  $b['Cooldown'] . ' ' . $b['Device']);
});
function cell($v)
{
    return htmlspecialchars($v === null ? '' : (string) $v, ENT_QUOTES,
                            'UTF-8');
}
function comments_summary($path)
{
    // 'n; latest <when> <who>: <text>' from a device's comments.json
    if (!is_file($path)) {
        return '';
    }
    $list = json_decode(file_get_contents($path), true);
    if (!is_array($list) || count($list) === 0) {
        return '';
    }
    $last = $list[count($list) - 1];
    $text = preg_replace('/\s+/', ' ', trim((string) ($last['text'] ?? '')));
    if (strlen($text) > 80) {
        $text = rtrim(substr($text, 0, 77)) . '...';
    }
    $who = trim((string) ($last['who'] ?? ''));
    return count($list) . '; latest ' . ($last['when'] ?? '')
        . ($who !== '' ? ' ' . $who : '') . ': ' . $text;
}
?>
<!DOCTYPE html><html><head><meta charset="utf-8"><title>Devices</title>
<style>@@CSS@@</style><script>@@JS@@</script></head><body>
<div class="nav"><a href="./">All devices</a></div>
<h1>Devices</h1>
<p class="small"><?php echo count($rows); ?> devices; listed
<?php echo date('Y-m-d H:i:s'); ?></p>
<p class="small">@@INTRO@@</p>
<div class="wrap"><table><thead><tr>@@HEAD@@</tr></thead><tbody>
<?php foreach ($rows as $row) { ?>
<tr><?php foreach ($columns as $c) {
    $v = isset($row[$c]) ? $row[$c] : '';
    if ($c === 'Device') {
        echo '<td><a href="' . cell($row['_link']) . '">' . cell($v)
            . '</a></td>';
    } else {
        echo '<td>' . cell($v) . '</td>';
    }
} ?></tr>
<?php } ?>
</tbody></table></div>
</body></html>
"""

COMMENTS_PHP_SOURCE = r"""<?php
// Adds, edits or deletes a comment in a device's comments.json (the
// report site's own file, which citkid's report build leaves alone).
// Written by citkid (7_report_site.py) next to index.php; the device
// pages post to it: action = add (default), edit or delete, with the
// comment's id for the last two.
header('Content-Type: text/plain; charset=utf-8');
if ($_SERVER['REQUEST_METHOD'] !== 'POST') {
    http_response_code(405);
    echo 'POST only';
    exit;
}
$cooldown = basename(trim((string) ($_POST['cooldown'] ?? '')));
$device = basename(trim((string) ($_POST['device'] ?? '')));
$who = trim((string) ($_POST['who'] ?? ''));
$text = trim(str_replace("\r\n", "\n", (string) ($_POST['text'] ?? '')));
$action = trim((string) ($_POST['action'] ?? 'add'));
$id = trim((string) ($_POST['id'] ?? ''));
$dir = __DIR__ . '/' . $cooldown . '/' . $device;
if ($cooldown === '' || $device === '' || $cooldown[0] === '.'
        || $device[0] === '.' || !is_file($dir . '/manifest.json')) {
    http_response_code(404);
    echo 'no such device';
    exit;
}
if (!in_array($action, array('add', 'edit', 'delete'), true)) {
    http_response_code(400);
    echo 'action must be add, edit or delete';
    exit;
}
if ($action !== 'delete'
        && ($text === '' || strlen($text) > 5000 || strlen($who) > 100)) {
    http_response_code(400);
    echo 'a comment needs text (up to 5000 characters) and a name of up '
        . 'to 100';
    exit;
}
if ($action !== 'add' && $id === '') {
    http_response_code(400);
    echo 'the comment to ' . $action . ' needs its id';
    exit;
}
$path = $dir . '/comments.json';
$fh = fopen($path, 'c+');
if ($fh === false) {
    http_response_code(500);
    echo 'cannot write ' . $path . ': give the web server write '
        . 'permission on the reports folder';
    exit;
}
flock($fh, LOCK_EX);
$list = json_decode(stream_get_contents($fh), true);
if (!is_array($list)) {
    $list = array();
}
if ($action === 'add') {
    $list[] = array('id' => uniqid('c'), 'who' => $who,
                    'when' => date('Y-m-d H:i:s'), 'text' => $text);
} else {
    // The comment by its id; comments saved without one by their position
    $at = -1;
    foreach ($list as $i => $c) {
        $cid = isset($c['id']) ? (string) $c['id'] : (string) $i;
        if ($cid === $id) {
            $at = $i;
            break;
        }
    }
    if ($at < 0) {
        flock($fh, LOCK_UN);
        fclose($fh);
        http_response_code(404);
        echo 'no comment ' . $id . ' (was it deleted?)';
        exit;
    }
    if ($action === 'edit') {
        $list[$at]['who'] = $who;
        $list[$at]['text'] = $text;
        $list[$at]['edited'] = date('Y-m-d H:i:s');
    } else {
        array_splice($list, $at, 1);
    }
}
ftruncate($fh, 0);
rewind($fh);
fwrite($fh, json_encode($list, JSON_PRETTY_PRINT | JSON_UNESCAPED_UNICODE));
fflush($fh);
flock($fh, LOCK_UN);
fclose($fh);
header('Location: ' . rawurlencode($cooldown) . '/' . rawurlencode($device)
    . '/index.html#comments', true, 303);
"""

def build_index(out_root, refresh = False):
    """
    Write the site's top page, ``index.php``: a page the web server
    fills from the device folders present when it is opened, from the
    row saved in each device's manifest (``index_rows``) and the live
    comments. Also ``comments.php``, which the device pages' comment
    forms post to. Served by a web server with PHP (the NAS); not
    pages to open from the drive, where the device pages are opened by
    themselves.

    Parameters:
    out_root (str): the site's root folder.
    refresh (bool): If True, compute every device's row again from its
        manifest and save it. Default False: only the missing rows.

    Returns:
    path (str): the page written.
    """
    out_root = os.path.expanduser(out_root)
    index_rows(out_root, refresh = refresh)
    head = ''.join(f'<th>{_esc(c)}</th>' for c in INDEX_COLUMNS)
    head += '</tr><tr class="filters">' + ''.join(
        f'<th class="nosort"><input class="colfilter" data-col="{i}" '
        f'placeholder="filter"></th>' for i in range(len(INDEX_COLUMNS)))
    columns = json.dumps(INDEX_COLUMNS).replace('\\', '\\\\') \
        .replace("'", "\\'")
    page = (PHP_TEMPLATE.replace('@@COLUMNS@@', columns)
            .replace('@@CSS@@', CSS).replace('@@JS@@', JS)
            .replace('@@INTRO@@', _esc(INDEX_INTRO))
            .replace('@@HEAD@@', head))
    path = os.path.join(out_root, INDEX_PHP)
    with open(path, 'w', encoding = 'utf-8') as fh:
        fh.write(page)
    with open(os.path.join(out_root, COMMENTS_PHP), 'w',
              encoding = 'utf-8') as fh:
        fh.write(COMMENTS_PHP_SOURCE)
    return path

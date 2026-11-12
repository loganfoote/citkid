"""
Build the PDFs in ``documents/`` from their Markdown sources in
``documents/source/`` (needs pandoc and a LaTeX installation with
xelatex).

The procedure guides contain placeholders that are filled from the
scripts in ``procedures/``, so they stay in step with the code:

- ``{{settings:<script>}}``: the script's settings block (the constants
  at its top, with their comments);
- ``{{help:<script>}}``: its command-line help (``--help``).

Usage:
    uv run python documents/build_pdfs.py            # every PDF
    uv run python documents/build_pdfs.py vna        # names containing 'vna'
"""
import contextlib
import importlib.util
import io
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCE = os.path.join(HERE, 'source')
PROCEDURES = os.path.join(os.path.dirname(HERE), 'procedures')
# (source file, output PDF relative to documents/, title)
DOCUMENTS = [
    ('getting_started.md', 'procedures/getting_started.pdf',
     'Getting started with the procedures'),
    ('vna_screening.md', 'procedures/vna_screening.pdf', 'VNA screening'),
    ('tuning_timestreams.md', 'procedures/tuning_timestreams.pdf',
     'Tuning and timestreams'),
    ('crs_streaming_format.md', 'crs_streaming_format.pdf',
     'CRS streaming data format'),
    ('pipeline_guide.md', 'pipeline_guide.pdf', 'Pipeline guide'),
    ('xcal_derivations.md', 'xcal_derivations.pdf', 'x calibration'),
]
BANNER = '#' * 80
PLACEHOLDER = re.compile(r'\{\{(settings|help):([\w.]+)\}\}')

def settings_block(script):
    """
    Return a procedure script's settings block.

    Parameters:
    script (str): file name in ``procedures/``.

    Returns:
    text (str): the lines between the Settings and Script banners.
    """
    with open(os.path.join(PROCEDURES, script), encoding = 'utf-8') as fh:
        lines = fh.read().splitlines()
    start = next(i for i, line in enumerate(lines) if 'Settings' in line
                 and line.startswith('#'))
    stop = next(i for i, line in enumerate(lines) if i > start + 1
                and line.startswith(BANNER))
    return '\n'.join(lines[start + 2:stop]).strip('\n')

def help_text(script):
    """
    Return a procedure script's command-line help.

    Parameters:
    script (str): file name in ``procedures/``.

    Returns:
    text (str): the ``--help`` output.
    """
    path = os.path.join(PROCEDURES, script)
    spec = importlib.util.spec_from_file_location(script[:-3], path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    argv0, sys.argv[0] = sys.argv[0], script
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out):
            module.parse_args(['--help'])
    except SystemExit:
        pass
    finally:
        sys.argv[0] = argv0
    return out.getvalue().rstrip()

def expand(text):
    """
    Fill the placeholders of a source and drop horizontal rules (section
    dividers; LaTeX can't place them everywhere Markdown can).

    Parameters:
    text (str): Markdown source.

    Returns:
    text (str): the expanded source.
    """
    def fill(match):
        """
        Replace one placeholder with a code block.

        Parameters:
        match (re.Match): the placeholder.

        Returns:
        block (str): the code block.
        """
        kind, script = match.groups()
        body = settings_block(script) if kind == 'settings' else \
            help_text(script)
        lang = 'python' if kind == 'settings' else 'text'
        return f'```{lang}\n{body}\n```'

    text = PLACEHOLDER.sub(fill, text)
    lines = text.splitlines()
    head = []
    if lines and lines[0].strip() == '---':
        # Keep the title block (between the first two '---' lines)
        end = next(i for i in range(1, len(lines))
                   if lines[i].strip() in ('---', '...'))
        head, lines = lines[:end + 1], lines[end + 1:]
    return '\n'.join(head + [line for line in lines if line.strip()
                              not in ('---', '***', '___')]) + '\n'

def build(source, output, title):
    """
    Build one PDF.

    Parameters:
    source (str): source file in ``documents/source/``.
    output (str): PDF path relative to ``documents/``.
    title (str): document title (PDF metadata).
    """
    with open(os.path.join(SOURCE, source), encoding = 'utf-8') as fh:
        text = expand(fh.read())
    out = os.path.join(HERE, output)
    os.makedirs(os.path.dirname(out), exist_ok = True)
    with tempfile.TemporaryDirectory() as tmp:
        md = os.path.join(tmp, source)
        with open(md, 'w', encoding = 'utf-8') as fh:
            fh.write(text)
        subprocess.run(
            ['pandoc', md, '-o', out, '--pdf-engine=xelatex',
             '-V', 'geometry:margin=0.9in', '-V', 'fontsize=10pt',
             '-V', 'colorlinks=true', '-V', 'linkcolor=blue',
             # TeX Gyre Pagella (text) and DejaVu Sans Mono (code), both in
             # TeX Live, have the symbols used (≈, ≤, box lines)
             '-V', 'mainfont=TeX Gyre Pagella',
             '-V', 'monofont=DejaVu Sans Mono',
             '-M', f'pagetitle={title}', '--resource-path', SOURCE],
            check = True)
    print(f'built {output}')

if __name__ == '__main__':
    import citkid  # noqa: F401  (Qt and rfmux settings before the scripts)
    wanted = sys.argv[1:]
    for source, output, title in DOCUMENTS:
        if not wanted or any(w in source for w in wanted):
            build(source, output, title)

"""citkid package metadata."""

import os as _os

# citkid's Qt code is written and tested for PyQt5, the binding pinned in
# environment.yml. rfmux also installs PyQt6, which pyqtgraph would
# otherwise prefer. This only takes effect if pyqtgraph is imported after
# citkid (``citkid.qt_compat`` checks).
_os.environ['PYQTGRAPH_QT_LIB'] = 'PyQt5'

try:
	from importlib.metadata import PackageNotFoundError, version
except ImportError:  # pragma: no cover - fallback for older Python
	from importlib_metadata import PackageNotFoundError, version

def _read_pyproject_version():
	try:
		import tomllib
	except ImportError:  # pragma: no cover
		import tomli as tomllib

	from pathlib import Path

	pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
	if not pyproject_path.exists():
		return "0.0"
	with pyproject_path.open("rb") as handle:
		data = tomllib.load(handle)
	return data.get("project", {}).get("version", "0.0")


_pyproject_version = _read_pyproject_version()
if _pyproject_version != "0.0":
	__version__ = _pyproject_version
else:
	try:
		__version__ = version("citkid")
	except PackageNotFoundError:  # pragma: no cover
		__version__ = "0.0"

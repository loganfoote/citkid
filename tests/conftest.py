import os

os.environ["CRS_EMBEDDED"] = "1"
os.environ["MPLBACKEND"] = "Agg"
os.environ["QT_QPA_PLATFORM"] = "offscreen"
import warnings

# Suppress DeprecationWarning raised intentionally by code scheduled for removal.
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
# Also ignore common third-party deprecation/quality warnings we see in tests
import importlib
try:
    import numpy as _np
    warnings.filterwarnings("ignore", category=_np.RankWarning)
except Exception:
    pass

try:
    import zarr as _zarr
    ZDW = getattr(_zarr, 'ZarrDeprecationWarning', None)
    if ZDW is None:
        for _sub in ('errors', 'core', 'core.errors'):
            try:
                m = importlib.import_module('zarr.' + _sub)
                ZDW = getattr(m, 'ZarrDeprecationWarning', None)
                if ZDW:
                    break
            except Exception:
                pass
    if ZDW:
        warnings.filterwarnings("ignore", category=ZDW)
except Exception:
    pass

# Ignore generic runtime warnings exposed during tests (e.g., coroutine not awaited)
warnings.filterwarnings("ignore", category=RuntimeWarning)

def pytest_addoption(parser):
    parser.addoption(
        "--crs_sn",
        action="store",
        default=None,
        help="CRS serial number",
    )
    parser.addoption(
        "--crs_iface",
        action="store",
        default=None,
        help="CRS interface (e.g. 'enp3s0')",
    )
    parser.addoption(
        "--rfsoc_ip",
        action="store",
        default=None,
        help="RFSOC UDP IP address (e.g. '192.168.3.40')",
    )
    parser.addoption(
        "--rfsoc_out_dir",
        action="store",
        default=None,
        help="Output directory for RFSOC hardware tests",
    )


def pytest_report_header(config):
    sn = config.getoption("--crs_sn")
    iface = config.getoption("--crs_iface")
    rfsoc_ip = config.getoption("--rfsoc_ip")
    rfsoc_out_dir = config.getoption("--rfsoc_out_dir")
    return (
        f"CRS options: --crs_sn={sn} --crs_iface={iface}\n"
        f"RFSOC options: --rfsoc_ip={rfsoc_ip} --rfsoc_out_dir={rfsoc_out_dir}"
    )
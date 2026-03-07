import os

os.environ["CRS_EMBEDDED"] = "1"
os.environ["MPLBACKEND"] = "Agg"
os.environ["QT_QPA_PLATFORM"] = "offscreen"

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


def pytest_report_header(config):
    sn = config.getoption("--crs_sn")
    iface = config.getoption("--crs_iface")
    return f"CRS options: --crs_sn={sn} --crs_iface={iface}"
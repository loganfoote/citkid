import os

# Ensure rfmux uses no periscope during test collection/imports.
os.environ["CRS_EMBEDDED"] = "1"

def pytest_addoption(parser):
    parser.addoption(
        "--crs",
        action="store_true",
        default=False,
        help="Enable CRS loopback tests",
    )
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
    crs = config.getoption("--crs")
    sn = config.getoption("--crs_sn")
    iface = config.getoption("--crs_iface")
    return f"CRS options: --crs={crs} --crs_sn={sn} --crs_iface={iface}"
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
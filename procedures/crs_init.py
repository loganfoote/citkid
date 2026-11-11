"""
Configure the CRS board (not a numbered step: run it whenever the
board needs configuring, e.g. after a power cycle).

Run it before the module scripts (vna_quick_check.py, 01_vna_sweep.py,
...), and again whenever the board loses its configuration.
This session has global control: it sets the clock source, analog bank,
DAC full scale (always 7 dBm), extended bandwidth, and decimation (stage
6, streaming every module in the bank, so other sessions can sweep), then
exits. The
board keeps these settings, and the module scripts attach to it without
changing them.

Running this again while other scripts are measuring resets the board
under them.

Usage:
    python crs_init.py
    python crs_init.py --clock SMA --extended-bw

Edit the settings below for your setup; the command-line options override
them for one run.
"""
import argparse
import asyncio
from citkid.crs.instrument import CRS

################################################################################
################################### Settings ###################################
################################################################################
SERIAL_NUMBER = 46          # CRS serial number
INTERFACE = 'enp5s0'        # Ethernet interface connected to the CRS
CLOCK_SOURCE = 'VCXO'       # 'VCXO' (internal) or 'SMA' (external 10 MHz)
ANALOG_BANK_HIGH = False    # False: modules 1-4; True: modules 5-8
EXTENDED_BW = False         # True: 600 MHz module bandwidth (more loss
                            # near the band edges)

################################################################################
################################### Script #####################################
################################################################################
def parse_args(argv = None):
    """
    Parse the command-line options (defaults from the settings above).

    Parameters:
    argv (list of str or None): arguments. None (default): sys.argv.

    Returns:
    args (argparse.Namespace): the options.
    """
    p = argparse.ArgumentParser(description = 'Configure the CRS board.')
    p.add_argument('--serial', type = int, default = SERIAL_NUMBER)
    p.add_argument('--iface', default = INTERFACE)
    p.add_argument('--clock', default = CLOCK_SOURCE,
                   choices = ['VCXO', 'SMA'])
    p.add_argument('--bank-high', action = 'store_true',
                   default = ANALOG_BANK_HIGH)
    p.add_argument('--extended-bw', action = 'store_true',
                   default = EXTENDED_BW)
    return p.parse_args(argv)

async def main(args):
    """
    Configure the board and print its state.

    Parameters:
    args (argparse.Namespace): options from ``parse_args``.

    Returns:
    crs (CRS): the configured instrument (for tests and interactive use).
    """
    crs = CRS(serial_number = args.serial, interface = args.iface)
    await crs.configure_system(
        clock_source = args.clock, full_scale_dbm = 7,
        analog_bank_high = args.bank_high, verbose = True)
    if args.extended_bw:
        await crs.set_extended_bw(True)
    modules = '5-8' if crs.analog_bank_high else '1-4'
    print(f'CRS {args.serial}: clock {crs.clock_source}, modules '
          f'{modules}, bandwidth {crs.bw / 1e6:.0f} MHz per module. '
          'The module scripts can attach now.')
    return crs

if __name__ == '__main__':
    asyncio.run(main(parse_args()))

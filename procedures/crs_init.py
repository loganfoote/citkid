"""
Configure the CRS board (clock, analog bank, DAC full scale, bandwidth,
decimation). Run it before the other scripts and after a power cycle.

Usage:
    python crs_init.py
    python crs_init.py --clock SMA --extended-bw
"""
import argparse
import asyncio
from citkid.crs.instrument import CRS

################################################################################
################################### Settings ###################################
################################################################################
SERIAL_NUMBER = 46
INTERFACE = 'enp5s0'
CLOCK_SOURCE = 'VCXO'
ANALOG_BANK_HIGH = False
EXTENDED_BW = False

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

"""
Settings shared by every script: the board, the device store, the
readout chain and the memory limit. Copy it into the run's scripts
folder with the scripts and set it once.
"""

################################################################################
################################### Settings ###################################
################################################################################
SERIAL_NUMBER = 46
INTERFACE = 'enp5s0'
MODULES = [1]
DATA_DIR = '~/data'
COOLDOWN = 'cooldown'
DEVICE = 'device'
SUFFIX = ''
SETUP = dict(fridge_ch = '', dac_attenuation = '', dac_filters = '',
             adc_filters = '', warm_amps = '', temperature = '')
MEMORY_GB = 12.
LIGHT_MODE = False

################################################################################

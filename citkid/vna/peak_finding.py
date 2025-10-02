 
 
import numpy as np
from scipy.signal import butter, sosfilt, sosfiltfilt, find_peaks

def butter_highpass_filter(data, lowcut, fs, order=1, direction='forward'):
    nyq = 0.5 * fs
    low = lowcut / nyq
    sos = butter(order, low, analog=False, btype='high', output='sos')
    if direction=='forward':
        y = sosfilt(sos, data)
    elif direction=='both':
        y = sosfiltfilt(sos, data)
    return y
 
def find_peaks_highpass(fvna, zvna, fcut, height=0.2, width=10e3, distance=50e3):
    '''
    Highpasses a VNA scan to get rid of gain fluctuations, then finds resonances
    with scipy.signal.find_peaks.

    Parameters:
        fvna: array of vna sweep frequencies
        zvna: complex S21 or absolute value |S21| data of vna sweep
        fcut: cutoff "wavelength" for highpassing the vna scan. Should be in the same frequency units as fvna
        height: peak height for scipy.signal.find_peaks
        width: peak width, in same frequency units as fvna
        distance: distance between neighboring peaks, in same frequency units as fvna
    Returns:
        peaks: array of indices where peaks were found
    '''

    fdiff = np.mean(np.diff(fvna))
    zlog = np.log10(abs(zvna))
    zlog -= zlog[0]
    zfilt = butter_highpass_filter(zlog, lowcut=fdiff/fcut, fs=1, order=1)
    width = int(width/fdiff)
    distance = int(distance/fdiff)
    peaks, _ = find_peaks(-zfilt, height=height, width=width, distance=distance)
    return peaks
#!/usr/bin/env python3

"""
Collect 2T2R voltages from ADALM-pluto, correlate and write out the auto and cross spectra as HDF5 files.
"""

__author__ = "Jishnu N. Thekkeppattu"         

import argparse
import sys
parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("-fc", nargs='?', default=100e6, type=float, help='Set centre frequency in Hz. Default is 100 MHz')
parser.add_argument("-fs", nargs='?', default=30e6, type=float, help='Set sample rate in Hz. Default is 30 MHz')
parser.add_argument("-c",  nargs='?', default=8192, type=int, help='Set chunk size of samples to be collected in each acq. Default is 8192')
parser.add_argument('-g1', nargs='?', default=30, type=int, help='Channel 1 gain. Default is 30.')
parser.add_argument('-g2', nargs='?', default=30, type=int, help='Channel 2 gain. Default is 30.')
parser.add_argument('-N',  nargs='?', type=int, help='Number of acquisitions. Set 1 for single shot. If unspecified, results in an infinite loop.')
parser.add_argument('-s',  nargs='?', help='Suffix to append filename with')
parser.add_argument('-NFFT', nargs='?', default=512, type=int, help='Number of frequency bins, Default is 512')

# args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])
args = parser.parse_args()
    
import adi
import numpy as np
import scipy
import sys
import h5py
import time

if args.N is None:
    Nacq = np.inf #Some number > 1 should work
else:
    Nacq = int(args.N)-1 # As one acq will happen always

def correlate_pluto(buff, NFFT):

    auto11 = np.zeros(NFFT, dtype=np.complex64)
    auto22 = np.zeros(NFFT, dtype=np.complex64)
    cross12 = np.zeros(NFFT, dtype=np.complex64)

    a = buff[0][:]
    b = buff[1][:]
    Nsamp = int(buff.shape[1])

    for j in range(int(Nsamp/NFFT)):
        c1_fft = np.fft.fft(a[j*NFFT:(j+1)*NFFT])
        c2_fft = np.fft.fft(b[j*NFFT:(j+1)*NFFT])
        auto11 = auto11 + c1_fft*np.conj(c1_fft)
        auto22 = auto22 + c2_fft*np.conj(c2_fft)
        cross12 = cross12 + c1_fft*np.conj(c2_fft)

    auto11 = np.fft.fftshift(auto11)
    auto22 = np.fft.fftshift(auto22)
    cross12 = np.fft.fftshift(cross12)
    return auto11, auto22, cross12

try:
    sdr = adi.ad9361("ip:192.168.2.5")
    sdr.rx_enabled_channels = [0, 1]
    sdr.sample_rate = int(args.fs)
    sdr.rx_rf_bandwidth = int(args.fs)
    sdr.rx_lo = int(args.fc)
    sdr.rx_buffer_size = int(args.c)
    sdr.gain_control_mode_chan0 = "manual"
    sdr.gain_control_mode_chan1 = "manual"
    sdr.rx_hardwaregain_chan0 = int(args.g1)
    sdr.rx_hardwaregain_chan1 = int(args.g2)

    NFFT = int(args.NFFT)
    fs = int(args.fs)
    fc = int(args.fc)
    freqs = np.fft.fftshift(np.fft.fftfreq(NFFT, 1/fs)+fc)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    if args.s is None:
        h5_file = str(timestr)+"_plutospec.h5"
    else:
        h5_file = str(timestr)+"_plutospec_"+str(args.s)+".h5"

    hf = h5py.File(h5_file, 'w')
    print ("Writing to",h5_file)
    data_group = hf.create_group('data')
    data_group.attrs["fc"] = int(args.fc)
    data_group.attrs["fs"] = int(args.fs)
    data_group.attrs["c"] = int(args.c)
    data_group.attrs["g1"] = int(args.g1)
    data_group.attrs["g2"] = int(args.g2)
    data_group.attrs["NFFT"] = int(args.NFFT)

    timestamp = time.time()
    samples = np.array(sdr.rx()) # First RX
    print (samples.shape)

    auto11, auto22, cross12 = correlate_pluto(buff=samples, NFFT=NFFT)

    data_group.create_dataset('timestamps', data=np.array([timestamp]), maxshape=(None,))
    data_group.create_dataset('auto11', data=np.array([auto11]), maxshape=(None, int(NFFT)))
    data_group.create_dataset('auto22', data=np.array([auto22]), maxshape=(None, int(NFFT)))
    data_group.create_dataset('cross12', data=np.array([cross12]), maxshape=(None, int(NFFT)))

    acq_index = 0
    while acq_index < Nacq:
        timestamp = time.time()
        print (acq_index, timestamp)
        samples = np.array(sdr.rx())

        auto11, auto22, cross12 = correlate_pluto(buff=samples, NFFT=NFFT)

        len_old_data = data_group['timestamps'].shape[0]
        hf['data/timestamps'].resize((len_old_data + 1), axis=0)
        hf['data/timestamps'][-1:] = timestamp

        hf['data/auto11'].resize((len_old_data + 1), axis=0)
        hf['data/auto11'][-1:] = np.array(auto11)

        hf['data/auto22'].resize((len_old_data + 1), axis=0)
        hf['data/auto22'][-1:] = np.array(auto22)

        hf['data/cross12'].resize((len_old_data + 1), axis=0)
        hf['data/cross12'][-1:] = np.array(cross12)

        acq_index = acq_index+1
    print ("Acq completed, exiting")
except KeyboardInterrupt:
    print ("KeyboardInterrupt, exiting")
    hf.close()
    sys.exit(130)
finally:
    hf.close()
    sys.exit()
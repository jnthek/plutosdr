#!/usr/bin/env python3

"""
Plot waterfall and spectra with ADALM-pluto
"""

__author__ = "Jishnu N. Thekkeppattu"         

import argparse
import sys
parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("-fc", nargs='?', default=100e6, type=float, help='Set centre frequency in Hz. Default is 100 MHz')
parser.add_argument("-fs", nargs='?', default=30e6, type=float, help='Set sample rate in Hz. Default is 30 MHz')
parser.add_argument('-g1', nargs='?', default=30, type=int, help='Channel 1 gain. Default is 30.')
parser.add_argument('-g2', nargs='?', default=30, type=int, help='Channel 2 gain. Default is 30.')
parser.add_argument('-NFFT', nargs='?', default=512, type=int, help='Number of frequency bins, Default is 512')
parser.add_argument('-N',  nargs='?', default=64, type=int, help='Number of spectra to be averaged. Default is 64')
parser.add_argument("-c",  nargs='?', type=int, help='Set chunk size of samples to be collected in each acq. If unspecified, same as NFFT')
parser.add_argument('-W',  nargs='?', default=2, type=int, help='Correlation to plot in waterfall, has to be 1, 2 or 12. Default is 2')
parser.add_argument('-D',  nargs='?', default=100, type=int, help='Waterfall depth. Default is 100')

# args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])
args = parser.parse_args()
    
import adi
import numpy as np
import scipy
import sys
import h5py
import time
import matplotlib.pyplot as plt

if (args.c) is None:
    chunk_size = int(args.NFFT)
else:
    chunk_size = int(args.c)

sdr = adi.ad9361("ip:192.168.2.5")
sdr.rx_enabled_channels = [0, 1]
sdr.sample_rate = int(args.fs)
sdr.rx_rf_bandwidth = int(args.fs)
sdr.rx_lo = int(args.fc)
sdr.rx_buffer_size = int(chunk_size)
sdr.gain_control_mode_chan0 = "manual"
sdr.gain_control_mode_chan1 = "manual"
sdr.rx_hardwaregain_chan0 = int(args.g1)
sdr.rx_hardwaregain_chan1 = int(args.g2)
samples = sdr.rx() # Dummy rx

samp_rate = int(args.fs)
Fc = int(args.fc)
NFFT = int(args.NFFT)
Naver = int(args.N)
N_tstamps = int(args.D)

freq = np.linspace((Fc-samp_rate/2), Fc+(samp_rate/2), NFFT)/1e6

fig, ax = plt.subplots(2,1,figsize=(10,6))
try:
    wfall_buffer = np.zeros([N_tstamps,NFFT]) + 1e-6
    while 1:
        auto11 = np.zeros(NFFT, dtype=np.complex64)
        auto22 = np.zeros(NFFT, dtype=np.complex64)
        cross12 = np.zeros(NFFT, dtype=np.complex64)

        for i in range(Naver):
            x = np.array(sdr.rx())
            a = x[0][:]
            b = x[1][:]
            Nsamp = x.shape[1]
        
            for j in range(int(Nsamp/NFFT)):
                c1_fft = np.fft.fft(a[j*NFFT:(j+1)*NFFT])
                c2_fft = np.fft.fft(b[j*NFFT:(j+1)*NFFT])
                auto11 = auto11 + c1_fft*np.conj(c1_fft)
                auto22 = auto22 + c2_fft*np.conj(c2_fft)
                cross12 = cross12 + c1_fft*np.conj(c2_fft)

        Nnorm = Naver*int(Nsamp/NFFT)

        auto11 = np.fft.fftshift(auto11/Nnorm)
        auto22 = np.fft.fftshift(auto22/Nnorm)
        cross12 = np.fft.fftshift(cross12/Nnorm)

        ax[0].cla()
        ax[0].plot(freq, 20*np.log10(np.abs(auto11)), label="auto11")
        ax[0].plot(freq, 20*np.log10(np.abs(auto22)), label="auto22")
        ax[0].plot(freq, 20*np.log10(np.abs(cross12)), label="cross12")
        ax[0].legend(loc='lower right')
        ax[0].set_xlabel("Freq (MHz)")

        wfall_buffer = np.roll(wfall_buffer, shift=1, axis=0)
        if int(args.W) == 1:
            spec_plot = np.log(np.abs(auto11))
        elif int(args.W) == 12:
            spec_plot = np.log(np.abs(cross12))
        else:
            spec_plot = np.log(np.abs(auto22))
        wfall_buffer[0,:] = spec_plot
        ax[1].cla()
        ax[1].imshow(wfall_buffer, aspect='auto', cmap='jet', vmin=np.min(spec_plot), vmax=np.max(spec_plot))

        plt.draw()
        plt.pause(0.1)

        del(auto11)
        del(auto22)
        del(cross12)

except KeyboardInterrupt:
    sys.exit(130)

plt.show()

#!/usr/bin/python3

import adi
import matplotlib.pyplot as plt
import numpy as np
import scipy
import sys

samp_rate = 30e6
Fc = 130e6
Nsamp = 8192
NFFT = 256
Naver = 16

freq = np.linspace((Fc-samp_rate/2), Fc+(samp_rate/2), NFFT)/1e6

sdr = adi.ad9361("ip:192.168.2.5")
sdr.rx_enabled_channels = [0, 1]
sdr.sample_rate = int(samp_rate)
sdr.rx_rf_bandwidth = int(samp_rate)
sdr.rx_lo = int(Fc)
sdr.rx_buffer_size = Nsamp
sdr.gain_control_mode_chan0 = "manual"
sdr.gain_control_mode_chan1 = "manual"
sdr.rx_hardwaregain_chan0 = int(60)
sdr.rx_hardwaregain_chan1 = int(60)
samples = sdr.rx() # Dummy rx

fig, ax = plt.subplots(2,1,figsize=(8,6))

try:
    while 1:
        auto11 = np.zeros(NFFT, dtype=np.complex64)
        auto22 = np.zeros(NFFT, dtype=np.complex64)
        cross12 = np.zeros(NFFT, dtype=np.complex64)

        for i in range(Naver):
            x = sdr.rx()
            a = x[0][:]
            b = x[1][:]
        
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
        ax[0].set_ylim(80,140)

        ax[1].cla()
        # ax[1].plot(freq, 180*np.unwrap(np.angle(cross12))/np.pi)
        ax[1].plot(freq, 180*(np.angle(cross12))/np.pi)
        ax[1].set_ylim(-360,360)

        plt.draw()
        plt.pause(0.1)

        del(auto11)
        del(auto22)
        del(cross12)

except KeyboardInterrupt:
    sys.exit(130)

plt.show()

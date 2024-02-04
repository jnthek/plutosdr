import numpy as np
import adi
import matplotlib.pyplot as plt

sample_rate = 61.44e6 # Hz
center_freq = 400e6 # Hz
num_samps = int(65536*16) # number of samples returned per call to rx()

sdr = adi.Pluto('ip:192.168.2.5')
sdr.gain_control_mode_chan0 = 'manual'
sdr.rx_hardwaregain_chan0 = 70.0 # dB
sdr.rx_lo = int(center_freq)
sdr.sample_rate = int(sample_rate)
sdr.rx_rf_bandwidth = int(sample_rate) # filter width, just set it to the same as sample rate for now
sdr.rx_buffer_size = num_samps

samples = sdr.rx() # receive samples off Pluto
plt.figure(figsize=(16,9))
plt.psd(samples, NFFT=8192, Fs=sample_rate/1e6, Fc=center_freq/1e6)
plt.show()


#print (samples[0:10])
#print (samples.shape)
#print (samples.dtype)

import adi
import matplotlib.pyplot as plt
import numpy as np
import scipy
 
# Create radio
sdr = adi.ad9361("ip:192.168.2.1")
samp_rate = 20e6
 
'''Configure Rx properties'''
sdr.rx_enabled_channels = [0, 1]
sdr.sample_rate = int(samp_rate)
sdr.rx_rf_bandwidth = int(2e6)
sdr.rx_lo = int(226.5e6)
sdr.rx_buffer_size = 8192
sdr.gain_control_mode_chan0 = "manual"
sdr.gain_control_mode_chan1 = "manual"
sdr.rx_hardwaregain_chan0 = int(40)
sdr.rx_hardwaregain_chan1 = int(40)
samples = sdr.rx() # receive samples off Pluto
while 1:
    x = sdr.rx()
    a = x[0][:]
    b = x[1][:]
    corr = scipy.signal.correlate(a, b, method='fft', mode='same')
    plt.clf()
    plt.plot(np.abs(corr))
    #plt.plot(np.angle(corr))
    # plt.ylim([-20e6, 20e6])
    plt.draw()
    plt.pause(0.1)
    #print(np.argmax(corr))
plt.show()

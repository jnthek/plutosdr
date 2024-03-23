#!/usr/bin/env python3

"""
Collect 2T2R voltages from ADALM-pluto and write them out as HDF5 files.
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

# args = parser.parse_args(args=None if sys.argv[1:] else ['--help'])
args = parser.parse_args()

if args.N is None:
    Nacq = int(100) #Some number > 1 should work
    acq_loop_increment = 0
else:
    Nacq = int(args.N)-1 # As one acq will happen always
    acq_loop_increment = 1
    

import adi
import numpy as np
import scipy
import sys
import h5py
import time

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

try:
    timestr = time.strftime("%Y%m%d-%H%M%S")
    if args.s is None:
        h5_file = str(timestr)+"_pluto.h5"
    else:
        h5_file = str(timestr)+"_pluto_"+str(args.s)+".h5"

    print ("Writing to",h5_file)
    hf = h5py.File(h5_file, 'w')
    data_group = hf.create_group('data')
    data_group.attrs["fc"] = int(args.fc)
    data_group.attrs["fs"] = int(args.fs)
    data_group.attrs["c"] = int(args.c)
    data_group.attrs["g1"] = int(args.g1)
    data_group.attrs["g2"] = int(args.g2)
    timestamp = time.time()
    samples = sdr.rx() # First RX
    print (np.array(samples).shape)
    data_group.create_dataset('timestamps', data=np.array([timestamp]), maxshape=(None,))
    data_group.create_dataset('samples', data=np.array([samples]), maxshape=(None, 2, int(args.c)))
    acq_index = 0
    while acq_index < Nacq:
        timestamp = time.time()
        samples = sdr.rx()

        len_old_data = data_group['timestamps'].shape[0]
        hf['data/timestamps'].resize((len_old_data + 1), axis=0)
        hf['data/timestamps'][-1:] = timestamp

        hf['data/samples'].resize((len_old_data + 1), axis=0)
        hf['data/samples'][-1:] = np.array(samples)

        acq_index = acq_index+acq_loop_increment

except KeyboardInterrupt:
    hf.close()
    sys.exit(130)
finally:
    hf.close()
    sys.exit()
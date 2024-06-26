import matplotlib.pyplot as plt
import numpy as np
import h5py
import glob, os
PATH = "/home/jishnu/codespace/plutosdr/"

plt.rcParams.update({'font.size': 10})

def h5filespec(h5_file):
    print ("Reading",h5_file)
    with h5py.File(h5_file, 'r') as hf: 
        fc = hf["data"].attrs['fc']
        fs = hf["data"].attrs['fs']
        c  = hf["data"].attrs['c']
        NFFT = int(hf["data"].attrs['NFFT'])
        auto11 = hf["data/auto11"][()]
        auto22 = hf["data/auto22"][()]
        cross12 = hf["data/cross12"][()]
        len_data = auto11.shape[0]
        print (len_data)
        timestamps = hf["data/timestamps"][()][0:len_data]
    freqs = np.fft.fftshift(np.fft.fftfreq(NFFT, 1/fs)+fc)
    return freqs, timestamps, auto11, auto22, cross12

os.chdir(PATH)
for h5_filename in glob.glob("*.h5"):
    print(h5_filename)
    freqs, timestamps, auto11, auto22, cross12 = h5filespec(PATH+h5_filename)

    fig, ax = plt.subplots()
    ax.pcolormesh(freqs/1e6, timestamps-timestamps[0], np.log(np.abs(auto11)), cmap='jet')
    ax.set_xlabel("Freq (MHz)")
    ax.set_ylabel("Time (s)")
    plt.savefig(h5_filename.split(".")[0]+".png", dpi=100)
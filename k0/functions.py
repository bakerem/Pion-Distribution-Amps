import numpy as np
import h5py
import pandas as pd

def read_data(Ns, Nt, init_char, Pz, bz):
    """
    Reads in the data from h5 file and pandas dataframe and calculates errors
    and means
    """
    columns = ["t"] + [str(i) for i in range(0,Ns)]
    file = h5py.File(f"k0/TMD_cfgs/64I.qTMDWF.ama.GSRC_W40_k0.{init_char}.eta10.PX0PY0PZ{Pz}.h5")
    DA_data = file[f"b_X/bT0/bz{bz}"]
    c2pt = pd.read_csv(f"k0/c2pt_cfgs/64IGSRC_W40_k0.ama.c2pt.PX0PY0PZ{Pz}.real.cfg.csv", names=columns)
    c2pt_data = np.array(c2pt.drop(axis="columns", labels=["t"])).transpose()
        
    # calculate means and error in ratio of DA/c2pt
    real_samples = np.zeros((Ns,Nt))
    imag_samples = np.zeros((Ns,Nt))
    ratio = DA_data/c2pt_data
    ratio *= -1j
    real_ratio = np.real(ratio)
    imag_ratio = np.imag(ratio)
    
    for i in range(0,Ns):
        real_sample = np.mean(np.delete(real_ratio,i,axis=0), axis=0)
        imag_sample = np.mean(np.delete(imag_ratio,i,axis=0), axis=0)
        real_samples[i,:] = real_sample
        imag_samples[i,:] = imag_sample

    real_ratio_means = np.mean(real_samples, axis=0)
    imag_ratio_means = np.mean(imag_samples, axis=0)

    real_ratio_stds = np.sqrt(Ns-1)*np.std(real_samples, axis = 0)
    imag_ratio_stds = np.sqrt(Ns-1)*np.std(imag_samples, axis = 0)

    return real_ratio_means, imag_ratio_means, real_ratio_stds, imag_ratio_stds



def phys_p(a, n):
    # takes a as an energy in GeV
    return 2 * np.pi * n * a / 64

import numpy as np
import h5py
import pandas as pd
from math import factorial
from scipy.special import gamma, jv
import matplotlib.pyplot as plt


alpha_s = 0.303
C_F = 4/3
gamma_E = 0.57721
a = 2.359



def read_data(Ns: int, Nt: int, init_char: str, Pz: int, bz: int) -> list:
    """
    Reads in the data from h5 file and pandas dataframe and calculates errors
    and means
    """

    columns = ["t"] + [str(i) for i in range(0,Ns)]
    if Pz < 4:
        file = h5py.File(f"0-data/qDA_cfgs/64I.qDA.ama.GSRC_W40_k0_flow05eps01.{init_char}.eta0.PX0PY0PZ{Pz}.h5")
        DA_data = file[f"b_X/bT0/bz{bz}"]
    else:
        file = h5py.File(f"0-data/qDA_cfgs/64I.qDA.ama.GSRC_W40_k6_flow05eps01.{init_char}.eta0.PX0PY0PZ{Pz}.h5")
        DA_data = file[f"b_X/bT0/bz{bz}"]

    # read in 2pt correlation data
    c2pt = pd.read_csv(f"0-data/c2pt_cfgs/64IGSRC_W40_k0.ama.c2pt.PX0PY0PZ{Pz}.real.cfg.csv", names=columns)
    c2pt_data = np.array(c2pt.drop(axis="columns", labels=["t"])).transpose()

    
    # calculate means and error in ratio of DA/c2pt
    real_samples      = np.zeros((Ns,Nt))
    imag_samples      = np.zeros((Ns,Nt))
    real_sample_errs  = np.zeros((Ns,Nt))
    imag_sample_errs = np.zeros((Ns,Nt))
    ratio = DA_data/c2pt_data
    if init_char == "Z5":
        ratio *= -1j
    real_ratio = np.real(ratio)
    imag_ratio = np.imag(ratio)
    
    for i in range(0,Ns):
        real_sample = np.mean(np.delete(real_ratio,i,axis=0), axis=0)
        imag_sample = np.mean(np.delete(imag_ratio,i,axis=0), axis=0)
        real_samples[i,:] = real_sample
        imag_samples[i,:] = imag_sample
        real_sample_errs[i,:] = np.std(np.delete(real_ratio,i,axis=0),axis=0)
        imag_sample_errs[i,:] = np.std(np.delete(imag_ratio,i,axis=0), axis=0)

    real_ratio_means = np.mean(real_samples, axis=0)
    imag_ratio_means = np.mean(imag_samples, axis=0)
    
    real_ratio_stds = np.sqrt(Ns-1)*np.std(real_samples, axis = 0)
    imag_ratio_stds = np.sqrt(Ns-1)*np.std(imag_samples, axis = 0)


    return (
        real_ratio_means, 
        imag_ratio_means, 
        real_ratio_stds, 
        imag_ratio_stds,
        real_samples,
        real_sample_errs,
        imag_samples,
        imag_sample_errs)



def phys_p(a: float, n: int):
    """takes a in GeV^-1 and returns physical momentum assosicated with n """
    return 2 * np.pi * n * a / 64

### Definitions for Mellin OPE ###

def m_ope(bz, mm2, mm4, n_max, mu, Pz):
    """
    Mellin OPE. Eq. (6) from  Gao et al. 2022, arxiv:2206.04084v2
    """
    m_moms = [1, 0, mm2, 0, mm4, 0]
    lam = a*bz*phys_p(a,Pz)
    h_tw2 = 0
    for n in range(0,n_max+1):
        term1 = (-1j*lam/2)**n/factorial(n)
        term2 = 0
        for m in range(0,n+1):
            term2 += m_coeff(n,m, mu, bz)*m_moms[m]
        h_tw2 += term1*term2
    return h_tw2

def m_coeff(n,m,mu, bz):
    """
    Mellin Coefficents
    """
    L = np.log((a*bz)**2*mu**2*np.exp(2*gamma_E)/4)
    # determination of C0
    if n == m:
        C0 = 1
    else:
        C0 = 0
    # determination of C1_mn
    if   n==0:
        C1 = 1.5*L - 3.5
    elif n==1:
        C1 = (17/6)*L - 0.5
    elif n==2:
        C1 = (43/12)*L-(37/12)
    elif n==3:
        C1 = (247/60)*L - (923/180)
    elif n==4:
        C1 = (68/15)*L - (247/36)


    coeff = C0 + alpha_s*C_F/(2*np.pi)*C1
    return coeff

### Definitions for Conformal OPE ###
def c_ope(an, n, mu, bz):
    """
    The conformal OPE for h_tw2. F_n are the conformal partial waves
    and a_n are the Gegenbauer moments. a_n will be the fitting parameters. Eq. 
    (17) from  Gao et al. 2022, arxiv:2206.04084v2
    """
    h_tw2 = 0
    Fn = F_n(n, mu, bz)
    for i in range(0,n,2):
        h_tw2 += an*Fn
    return h_tw2


def c_n(n, mu, bz, init_char):
    if init_char == "Z5":
        delta = 1
    else:
        delta = 0
    Hn = sum(1/i for i in range(1,n+1))
    cn = 1 + (alpha_s*C_F/(2*np.pi))*(((5+2*n)/(3+n+n**2)) 
                                  + (2*delta)/(2+3*n+n**2)
                                  + 2*Hn-Hn**2 - 2*Hn**2
                                  )

    return 

def F_n(n, mu, bz, Pz):
    """
    Expression for the conformal partial waves. Eq. 
    (18) from  Gao et al. 2022, arxiv:2206.04084v2
    """
    cn = c_n(n, mu, bz)
    Hn1 = sum(1/i for i in range(1,n+2))

    gamma_n = alpha_s*C_F/(4*np.pi)*(4*Hn1 - (2/(n**2+3*n+2))-3)
    gamma_O = 3*alpha_s*C_F/(4*np.pi)

    Fn = (cn * (mu**2*bz**2)**(gamma_n+gamma_O) * (gamma(2-gamma_n)*gamma(1+n))/gamma(1+n+gamma_n)
        * 0.375 * 1j**n * np.sqrt(np.pi) * (n**2+3*n+2) * gamma(n+gamma_n+2.5)/gamma(n+2.5)
        * (bz*Pz/2)**(-1.5-gamma_n)*jv(n+gamma_n+1.5,bz*Pz)
    )
    return Fn


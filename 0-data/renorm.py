# Author: Ethan Baker ANL/Haverford College

import numpy as np
import matplotlib.pyplot as plt
import os
from functions import phys_p

Nt = 128
Ns = 55
a = 2.359
bz_max = 10
P0 = 1
init_char = "Z5"
fits = {}
save = False
real = True


# Create save path
save_path = "final_results/2_state_matrix_results"
os.makedirs(save_path, exist_ok=True)


# Calculating stuff for P0
P0_reals = np.load(f"{save_path}/{init_char}/Pz{P0}/real_data.npy")
P0_rerr = np.load(f"{save_path}/{init_char}/Pz{P0}/real_errs.npy")
P0_imags = np.load(f"{save_path}/{init_char}/Pz{P0}/imag_data.npy")
P0_ierr = np.load(f"{save_path}/{init_char}/Pz{P0}/real_errs.npy")

##### Plot for Matrix elements over multiple z ######
plt.figure()
formats = {"1":"bs","2":"ro","3":"gH","4":"m*","4":"cD","9":"y^"}
for Pz in range(1,4):
    real_fit = np.load(f"{save_path}/{init_char}/Pz{Pz}/real_data.npy")
    real_fit_err = np.load(f"{save_path}/{init_char}/Pz{Pz}/real_errs.npy")
    imag_fit = np.load(f"{save_path}/{init_char}/Pz{Pz}/imag_data.npy")
    imag_fit_err = np.load(f"{save_path}/{init_char}/Pz{Pz}/real_errs.npy")

    matrix_els = (((real_fit+1j*imag_fit)/(P0_reals + 1j*P0_imags))
                    *((P0_reals[0]+1j*P0_imags[0])/(real_fit[0] + 1j*imag_fit[0]))
                    *(np.exp(-1j*np.arange(0,33)/a*(phys_p(a,Pz)-phys_p(a,P0))/2))
                 )
    
    m_err1 = (((real_fit+real_fit_err+1j*imag_fit)/(P0_reals + 1j*P0_imags))
                    *((P0_reals[0]+1j*P0_imags[0])/(real_fit[0] + real_fit_err[0] + 1j*imag_fit[0]))
                    *(np.exp(-1j*np.arange(0,33)/a*(phys_p(a,Pz)-phys_p(a,P0))/2))
                 ) - matrix_els
    m_err2 = (((real_fit+1j*(imag_fit+imag_fit_err))/(P0_reals + 1j*P0_imags))
                    *((P0_reals[0]+1j*P0_imags[0])/(real_fit[0] + 1j*(imag_fit[0]+imag_fit_err[0])))
                    *(np.exp(-1j*np.arange(0,33)/a*(phys_p(a,Pz)-phys_p(a,P0))/2))
                 ) - matrix_els
    m_err3 = (((real_fit+1j*imag_fit)/(P0_reals + P0_rerr + 1j*P0_imags))
                    *((P0_reals[0]+P0_rerr[0]+1j*P0_imags[0])/(real_fit[0] + 1j*imag_fit[0]))
                    *(np.exp(-1j*np.arange(0,33)/a*(phys_p(a,Pz)-phys_p(a,P0))/2))
                 ) - matrix_els
    m_err4 = (((real_fit+1j*imag_fit)/(P0_reals + 1j*(P0_imags+P0_ierr)))
                    *((P0_reals[0]+1j*(P0_imags[0]+P0_ierr[0]))/(real_fit[0] + 1j*imag_fit[0]))
                    *(np.exp(-1j*np.arange(0,33)/a*(phys_p(a,Pz)-phys_p(a,P0))/2))
                 ) - matrix_els
    
    m_err = np.sqrt(m_err1**2 + m_err2**2 + m_err3**2 + m_err4**2)

    if save:
        np.save(f"final_results/renorm_results/Pz{Pz}_matrix_els.npy", matrix_els)
        np.save(f"final_results/renorm_results/Pz{Pz}_matrix_errs.npy", m_err)
    if real:
        plt.errorbar(np.arange(0,bz_max),
                    np.real(matrix_els[:bz_max]),
                    # yerr = np.real(m_err[:bz_max]),
                    fmt=formats[str(Pz)],
                    capsize=3,
                    label=f"Pz = {Pz}")
        # plt.text(bz_max-0.8, np.real(matrix_els[bz_max-1]),f"Pz = {Pz}")
    else:
        plt.errorbar(np.arange(0,bz_max),
                    np.imag(matrix_els[:bz_max]),
                    yerr = abs(np.imag(m_err[:bz_max])),
                    fmt=formats[str(Pz)],
                    capsize=3,
                    label=f"Pz = {Pz}")
        # plt.text(bz_max-0.8, np.imag(matrix_els[bz_max-1]),f"Pz = {Pz}")

if real:
    plt.legend()
    plt.xlabel(r"$z_3/a$")
    plt.ylabel(r"Re $\mathcal{M}$")
    # plt.xlim(-1,13)
    plt.title("Renormalized Matrix Elements")
    plt.text(4, 0.8, r"$P_0$ " + "= %.2f GeV" %phys_p(a,P0))
    if save:
        os.makedirs("final_results/renorm_results", exist_ok=True)
        plt.savefig("final_results/renorm_results/real_renorm_multi_p.png")
    plt.show()

else:
    plt.legend()
    plt.xlabel(r"$z_3/a$")
    plt.ylabel(r"Imag $\mathcal{M}$")
    plt.xlim(-1,15)
    plt.title("Renormalized Matrix Elements")
    if save:
        os.makedirs("final_results/renorm_results", exist_ok=True)
        plt.savefig("final_results/renorm_results/imag_renorm_multi_p.png")
    plt.show()





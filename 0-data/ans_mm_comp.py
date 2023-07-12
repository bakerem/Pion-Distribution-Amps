# Author Ethan Baker, ANL/Haverford College

import numpy as np
import os
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from functions import m_ope, phys_p
import scienceplots

plt.style.use("science")

Nt = 128
a = 2.359 # GeV^-1
mu = 2    # GeV
P0 = 3
N_max = 2
Ns = 55

save = True
delta = 20
# initialize empty arrays for saving data
for smear in ["final_results", "final_results_eps10"]:
    for init_char in ["Z5","T5"]:
        plt.figure()
        for fit in [True,False]:
            print(init_char)
            # create save path
            save_path = f"{smear}/2_state_matrix_results_jack/{init_char}/final_mm"
            os.makedirs(save_path, exist_ok=True)
            if fit:
                mm2 = np.load(f"{save_path}/avg2_no_corr.npy")
                mm4 = np.load(f"{save_path}/avg4_no_corr.npy")
            else:
                mm2 = np.load(f"{save_path}/ans_mm2_nocorr_d{delta}.npy")
                mm4 = np.load(f"{save_path}/ans_mm4_nocorr_d{delta}.npy")
            #load in data
            index = 0
    
            # function for fitting calculates ratio and wraps around Mellin-OPE of 
            # the matrix elements
            def ratio(bz, Pz, mm2, mm4,  ):
                num   = m_ope(bz/a, mm2, mm4, 0, 0, 0, 0, 0, 2*N_max, 0, phys_p(a,Pz), init_char)
                denom = m_ope(bz/a, mm2, mm4, 0, 0, 0, 0, 0, 2*N_max, 0, phys_p(a,P0), init_char)
                ratio_result = num/denom
                return np.real(ratio_result)
            bz_range = np.arange(0,9,0.001)
            Pz = 6
            r_err1p1 = ratio(bz_range, Pz, mm2[0] + mm2[1], mm4[0],) - ratio(bz_range, Pz, mm2[0], mm4[0],)
            r_err2p1 = ratio(bz_range, Pz, mm2[0], mm4[0] + mm4[1],) - ratio(bz_range, Pz, mm2[0], mm4[0],)
            r_errtp1 = np.sqrt(r_err1p1**2 + r_err2p1**2)
            r_err1m1 = ratio(bz_range, Pz, mm2[0]  - mm2[1], mm4[0],) - ratio(bz_range, Pz, mm2[0], mm4[0],)
            r_err2m1 = ratio(bz_range, Pz, mm2[0], mm4[0] - mm4[1],) - ratio(bz_range, Pz, mm2[0], mm4[0],)
            r_errtm1 = np.sqrt(r_err1m1**2 + r_err2m1**2)

            r_err1p2 = ratio(bz_range, Pz, mm2[0] + mm2[1] + mm2[2], mm4[0],) - ratio(bz_range, Pz, mm2[0], mm4[0],)
            r_err2p2 = ratio(bz_range, Pz, mm2[0], mm4[0] + mm4[1] + mm4[2],) - ratio(bz_range, Pz, mm2[0], mm4[0],)
            r_errtp2 = np.sqrt(r_err1p2**2 + r_err2p2**2)
            r_err1m2 = ratio(bz_range, Pz, mm2[0]  - mm2[1] - mm2[2], mm4[0],) - ratio(bz_range, Pz, mm2[0], mm4[0],)
            r_err2m2 = ratio(bz_range, Pz, mm2[0], mm4[0] - mm4[1] - mm4[2],) - ratio(bz_range, Pz, mm2[0], mm4[0],)
            r_errtm2 = np.sqrt(r_err1m2**2 + r_err2m2**2)

            if fit:
                color = "blue"
                label = "Moments Fit"
            else:
                color = "red"
                label = "Ansatz Fit"
            init_conv = {"Z5": r"$\gamma_3\gamma_5$", "T5": r"$\gamma_0\gamma_5$"}
            plt.plot(phys_p(a,Pz)*bz_range/a,ratio(bz_range, Pz, mm2[0], mm4[0]),label=label, color=color )
            plt.fill_between(phys_p(a,Pz)*bz_range/a,
                             ratio(bz_range, Pz, mm2[0], mm4[0])-r_errtm1, 
                             ratio(bz_range, Pz, mm2[0], mm4[0])+r_errtp1, color=color, alpha=0.3)
            plt.fill_between(phys_p(a,Pz)*bz_range/a,
                             ratio(bz_range, Pz, mm2[0], mm4[0])-r_errtm2, 
                             ratio(bz_range, Pz, mm2[0], mm4[0])+r_errtp2, color=color, alpha=0.2)
        plt.plot(phys_p(a,Pz)*bz_range/a,ratio(bz_range, Pz, 0.333, 0.2),"-.", color="black", label="Flat")
        plt.plot(phys_p(a,Pz)*bz_range/a,ratio(bz_range, Pz, 0.2, 0.0857), "--", color="black", label="Asymptotic")
        plt.legend()
        plt.xlabel(r"$\lambda$")
        plt.ylabel(r"Re $\mathcal{M}$")
        plt.title(f"Comparison of Mellin Moments, {init_conv[init_char]}")
        if save:
            plt.savefig(f"{save_path}/{init_char}mm_fit_comp.pdf")
        # plt.show()

            
            
        
        





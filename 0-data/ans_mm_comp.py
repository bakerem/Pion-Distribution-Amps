import numpy as np
import os
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as font_manager
from functions import m_ope, phys_p
import scienceplots

plt.style.use("science")

"""
Author Ethan Baker, ANL/Haverford College

Reconstructs matrix elements from mellin_moms_plots.py and x_dep_plots.py. These
are also compared to the reconstructions from the MMs of 2 hypothetical DAs. THis
functions very similarly to "mellin_moms_data_plots.py," just without
plotting the data. 

"""

Nt = 128
a = 1/2.359 # GeV^-1
P0 = 1
N_max = 2
alpha_s = 0.303
gamma_E = 0.57721

save = True
delta = 5
# initialize empty arrays for saving data
for smear in [ "final_results", "final_results_flow10"]:
    for init_char in ["T5"]:
        plt.figure()
        for fit in [False,True]:
            print(init_char)
            # create save path
            save_path = f"{smear}/2_state_matrix_results_jack/{init_char}/final_mm"
            os.makedirs(save_path, exist_ok=True)
            if fit:
                mm2 = np.load(f"{save_path}/avg2_corr_P0{P0}.npy")
                mm4 = np.load(f"{save_path}/avg4_corr_P0{P0}.npy")
                # h0 = np.load(f"{save_path}/avgh0_corr_P0{P0}.npy")

            else:
                mm2 = np.load(f"{save_path}/ans_mm2_nocorr_d{delta}_P0{P0}.npy")
                mm4 = np.load(f"{save_path}/ans_mm4_nocorr_d{delta}_P0{P0}.npy")
                # h0  = np.load(f"{save_path}/ans_h0_nocorr_d{delta}_P0{P0}.npy")
                print(mm2, mm4)

            # function for fitting calculates ratio and wraps around Mellin-OPE of 
            # the matrix elements
            def ratio(bz, Pz, mm2, mm4,):
                mu = 2*np.exp(-gamma_E)/(bz*a)
                alpha_s = 0.303
                num   = m_ope(bz*a, mm2, mm4, 0, 0, 0, 0, 0, 2*N_max, 1, phys_p(a,Pz), alpha_s, mu, init_char)
                denom = m_ope(bz*a, mm2, mm4, 0, 0, 0, 0, 0, 2*N_max, 1, phys_p(a,P0), alpha_s, mu, init_char)
                ratio_result = num/denom
                return np.real(ratio_result)
            bz_range = np.arange(0,9,0.001)
            Pz = 9

            # error calculation by propogating the errors through the calculation of
            # the ratio
            r_err1p1 = ratio(bz_range, Pz, mm2[0] + mm2[1], mm4[0], ) - ratio(bz_range, Pz, mm2[0], mm4[0], )
            r_err2p1 = ratio(bz_range, Pz, mm2[0], mm4[0] + mm4[1], ) - ratio(bz_range, Pz, mm2[0], mm4[0], )
            r_errtp1 = np.sqrt(r_err1p1**2 + r_err2p1**2 )

            r_err1m1 = ratio(bz_range, Pz, mm2[0]  - mm2[1], mm4[0], ) - ratio(bz_range, Pz, mm2[0], mm4[0], )
            r_err2m1 = ratio(bz_range, Pz, mm2[0], mm4[0] - mm4[1], ) - ratio(bz_range, Pz, mm2[0], mm4[0], )

            r_errtm1 = np.sqrt(r_err1m1**2 + r_err2m1**2 )

            r_err1p2 = ratio(bz_range, Pz, mm2[0] + mm2[1] + mm2[2], mm4[0], ) - ratio(bz_range, Pz, mm2[0], mm4[0], )
            r_err2p2 = ratio(bz_range, Pz, mm2[0], mm4[0] + mm4[1] + mm4[2], ) - ratio(bz_range, Pz, mm2[0], mm4[0], )

            r_errtp2 = np.sqrt(r_err1p2**2 + r_err2p2**2)
            r_err1m2 = ratio(bz_range, Pz, mm2[0]  - mm2[1] - mm2[2], mm4[0], ) - ratio(bz_range, Pz, mm2[0], mm4[0], )
            r_err2m2 = ratio(bz_range, Pz, mm2[0], mm4[0] - mm4[1] - mm4[2], ) - ratio(bz_range, Pz, mm2[0], mm4[0], )

            r_errtm2 = np.sqrt(r_err1m2**2 + r_err2m2**2 )

            if fit:
                color = "blue"
                label = "Moments Fit"
            else:
                color = "red"
                label = "Ansatz Fit"
            init_conv = {"Z5": r"$\gamma_3\gamma_5$", "T5": r"$\gamma_0\gamma_5$"}
            plt.plot(phys_p(a,Pz)*bz_range*a,ratio(bz_range, Pz, mm2[0], mm4[0], ),label=label, color=color )
            plt.fill_between(phys_p(a,Pz)*bz_range*a,
                             ratio(bz_range, Pz, mm2[0], mm4[0], )-r_errtm1, 
                             ratio(bz_range, Pz, mm2[0], mm4[0], )+r_errtp1, color=color, alpha=0.3)
            plt.fill_between(phys_p(a,Pz)*bz_range*a,
                             ratio(bz_range, Pz, mm2[0], mm4[0], )-r_errtm1-r_errtm2, 
                             ratio(bz_range, Pz, mm2[0], mm4[0], )+r_errtm1+r_errtp2, color=color, alpha=0.2)
        plt.plot(phys_p(a,Pz)*bz_range*a,ratio(bz_range, Pz, 0.333, 0.2,),"-.", color="black", label="Flat")
        plt.plot(phys_p(a,Pz)*bz_range*a,ratio(bz_range, Pz, 0.2, 0.0857,), "--", color="black", label="Asymptotic")
        plt.legend()
        plt.xlabel(r"$\lambda=P_3z_3$")
        plt.ylabel(r"Re $\mathcal{M}$")
        plt.xlim(0,4.5)
        plt.ylim(0.4,1)
        plt.text(0.1, 0.9, f"$\\delta=${delta/100}")
        # plt.title(f"Reconstructed Matrix Elements")
        if save:
            plt.savefig(f"{save_path}/{init_char}mm_fit_comp_P0{P0}_d{delta}.pdf")
        plt.show()

            
            
        
        





from tkinter import S
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit
from functions import m_ope, phys_p, c_ope, alpha_func
import scienceplots

plt.style.use("science")

"""
Ethan Baker, ANL/Haverford College

Produces fixed-z plots comparing the different features of the Mellin moments,
e.g. truncation order, type of OPE, effect of resummation, etc. In general, this
is modified a lot, with line1, line2, line3 representing different cases to
compare. Change the features of interest and adjust plot labels to produce
what you want. 

"""

Nt = 128
a = 1/2.359 # GeV^-1
gamma_E = 0.57721 # Euler constant
P0 = 1 # Reference momentum
N_max = 2
kappa = 1


Pz_range = np.arange(2,10)
save = True

# initialize empty arrays for saving data
for smear in ["final_results_flow10"]:
    for init_char in ["T5"]:
        # create save path
        plt.figure()
        save_path = f"{smear}/2_state_matrix_results_jack/{init_char}"
        os.makedirs(save_path, exist_ok=True)
        matrix_els   = np.load(f"{save_path}/real_matrix_el_P0{P0}.npy")
        matrix_el_js = np.load(f"{save_path}/real_matrix_elj_P0{P0}.npy")
        matrix_errs  = np.load(f"{save_path}/real_matrix_err_P0{P0}.npy")

        #load in data
        for bz in range(2,8):
    
            # function for fitting calculates ratio and wraps around Mellin-OPE of 
            # the matrix elements
            def ratio_m(Pz_range, mm2, mm4,  ):
                mu = kappa*2*np.exp(-gamma_E)/(bz*a)
                alpha_s = 0.303
                num   = m_ope(bz*a, mm2, mm4, 0, 0, 0, 0, 0, 2*N_max, 0, phys_p(a,Pz_range), alpha_s, mu, init_char)
                denom = m_ope(bz*a, mm2, mm4, 0, 0, 0, 0, 0, 2*N_max, 0, phys_p(a,P0), alpha_s, mu, init_char)
                ratio_result = num/denom
                return np.real(ratio_result)
                                    
            
            def ratio_c(Pz_range, mm2, mm4, mm6 ):
                mu = kappa*2*np.exp(gamma_E)/(bz*a)
                alpha_s = 0.303
                num   = m_ope(bz*a, mm2, mm4, mm6, 0, 0, 0, 0, 6, 0, phys_p(a,Pz_range), alpha_s, mu, init_char)
                denom = m_ope(bz*a, mm2, mm4, mm6, 0, 0, 0, 0, 6, 0, phys_p(a,P0), alpha_s, mu, init_char)
                ratio_result = num/denom
                return np.real(ratio_result)
            
            def ratio_t(Pz_range, mm2, mm4,mm6, an8,):
                mu = kappa*2*np.exp(gamma_E)/(bz*a)
                alpha_s = 0.303
                num   = m_ope(bz*a, mm2, mm4, mm6, an8, 0, 0, 0, 8, 0, phys_p(a,Pz_range), alpha_s, mu, init_char)
                denom = m_ope(bz*a, mm2, mm4, mm6, an8, 0, 0, 0, 8, 0, phys_p(a,P0), alpha_s, mu, init_char)
                ratio_result = num/denom
                return np.real(ratio_result)
            
            Ns = 55
            cov = np.zeros((len(Pz_range), len(Pz_range)))
            for i, Pz in enumerate(Pz_range):
                for j, Pz1 in enumerate(Pz_range):
                    cov[i,j] = np.average((matrix_el_js[Pz, bz,:] - np.average(matrix_el_js[Pz,bz,:]))*
                                            (matrix_el_js[Pz1, bz,:] - np.average(matrix_el_js[Pz1,bz,:])))
            cov = (Ns-1)*cov
            # fit curve
            popt_m, pcov_m = curve_fit(ratio_m,
                                Pz_range,
                                matrix_els[Pz_range,bz],
                                p0=(0.3,0.2),
                                sigma=cov)
            
            popt_c, pcov_c = curve_fit(ratio_c,
                                Pz_range,
                                matrix_els[Pz_range,bz],
                                # p0=(0.34,0.2,-0.6),
                                # bounds=[(-np.inf,-np.inf,-1,), (np.inf, np.inf, 1)],
                                sigma=cov)
            popt_t, pcov_t = curve_fit(ratio_t,
                                Pz_range,
                                matrix_els[Pz_range,bz],
                                # p0 =(0.33,0.2,0.1),
                                # bounds=[(0,0, -1,), (0.5, 0.3, 1)],
                                sigma=cov)
            # calculate chi^2
            # chi2 = (np.sum(
            #     (matrix_els[Pz_range, bz]
            #         - ratio(Pz_range, *popt)
            #     ) ** 2
            #     / (matrix_errs[Pz_range,bz]) ** 2 )
            #     / (Pz_range.shape[0] - len(popt)))
            # save data
            # plotting routine
            m_err = np.abs((0.2 + (12/35)*popt_m[0]) - (0.2 + (12/35)*(popt_m[0]+np.sqrt(pcov_m[0,0]))))
            # c_err = np.abs((0.2 + (12/35)*popt_c[0]) - (0.2 + (12/35)*(popt_c[0]+np.sqrt(pcov_c[0,0]))))
            # t_err = np.abs((0.2 + (12/35)*popt_t[0]) - (0.2 + (12/35)*(popt_t[0]+np.sqrt(pcov_t[0,0]))))
            print(popt_t)
            line1 = plt.errorbar(bz, 
                    popt_m[1],
                    yerr=np.sqrt(pcov_m[1,1]),
                    fmt = "bs",
                    # color="#0082CA",
                    markerfacecolor="none",
                    label = "M-OPE",
                    capsize=3,)
            
            line2 = plt.errorbar(bz, 
                    popt_c[1],
                    yerr=np.sqrt(pcov_m[1,1]),
                    fmt = "ro",
                    # color="#CD202C",
                    markerfacecolor="none",
                    label = "C-OPE",
                    capsize=3,)

            line3 = plt.errorbar(bz, 
                    popt_t[1],
                    yerr=np.sqrt(pcov_t[1,1]),
                    fmt = "gv",
                    # color="#77B300",
                    markerfacecolor="none",
                    label = "Tree Expansion",
                    capsize=3,)



        plt.title(r"Fixed $z_3$ Fits")
        plt.xlabel(r"$z_3/a$")
        plt.ylabel(r"$\langle x^4 \rangle$")
        plt.ylim(0,0.3)
        # plt.text(6.05, 0.3, "M-OPE")
        plt.legend([line1, line2, line3], [r"$N_\text{max}=4$", r"$N_\text{max}=6$", r"$N_\text{max}=8$",])

        if save:
            save_path = f"{smear}/2_state_matrix_results_jack/{init_char}"
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(f"{save_path}/fixed_z_mm2_Nmax_resummed_kappa_"+"%.2f"%kappa+f"_P0{P0}.pdf")
        plt.show()

            
            





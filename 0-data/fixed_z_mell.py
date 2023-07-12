# Author Ethan Baker, ANL/Haverford College

from tkinter import S
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit
from functions import m_ope, phys_p, c_ope
import scienceplots

plt.style.use("science")

Nt = 128
a = 2.359 # GeV^-1
mu = 2    # GeV
alpha_s = 0.303
P0 = 3
N_max = 2
Nh = 0
Nl = 1


Pz_range = np.arange(4,10)
save = True

# initialize empty arrays for saving data

for smear in ["final_results", "final_results_eps10"]:
    for init_char in ["Z5","T5"]:
        # create save path
        plt.figure()
        save_path = f"{smear}/2_state_matrix_results_jack/{init_char}"
        os.makedirs(save_path, exist_ok=True)
        matrix_els   = np.load(f"{save_path}/real_matrix_el.npy")
        matrix_el_js = np.load(f"{save_path}/real_matrix_elj.npy")
        matrix_errs  = np.load(f"{save_path}/real_matrix_err.npy")

        #load in data
        for bz in range(2,8):
    
            # function for fitting calculates ratio and wraps around Mellin-OPE of 
            # the matrix elements
            def ratio_m(Pz_range, mm2, mm4, ):
                num   = m_ope(bz/a, mm2, mm4, 0, 0, 0, 0, 0, 2*2, Nh, phys_p(a,Pz_range), init_char)
                denom = m_ope(bz/a, mm2, mm4, 0, 0, 0, 0, 0, 2*2, Nh, phys_p(a,P0), init_char)
                ratio_result = num/denom
                return np.real(ratio_result)
            
            def ratio_c(Pz_range, an2, an4,mm6 ):
                num   = m_ope(bz/a, an2, an4, mm6, 0, 0, 0, 0, 2*3, Nh, phys_p(a,Pz_range), init_char)
                denom = m_ope(bz/a, an2, an4, mm6, 0, 0, 0, 0, 2*3, Nh, phys_p(a,P0), init_char)
                ratio_result = num/denom
                return np.real(ratio_result)
            
            def ratio_t(Pz_range, an2, an4,mm6, mm8 ):
                num   = m_ope(bz/a, an2, an4, mm6, mm8, 0, 0, 0, 2*4, Nh, phys_p(a,Pz_range), init_char)
                denom = m_ope(bz/a, an2, an4, mm6, mm8, 0, 0, 0, 2*4, Nh, phys_p(a,P0), init_char)
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
                                sigma=cov)
            
            popt_c, pcov_c = curve_fit(ratio_c,
                                Pz_range,
                                matrix_els[Pz_range,bz],
                                sigma=cov)
            
            popt_t, pcov_t = curve_fit(ratio_t,
                                Pz_range,
                                matrix_els[Pz_range,bz],
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
            c_err = np.abs((0.2 + (12/35)*popt_c[0]) - (0.2 + (12/35)*(popt_c[0]+np.sqrt(pcov_c[0,0]))))
            t_err = np.abs((0.2 + (12/35)*popt_t[0]) - (0.2 + (12/35)*(popt_t[0]+np.sqrt(pcov_t[0,0]))))
            
            line1 = plt.errorbar(bz, 
                    popt_m[1],
                    yerr=np.sqrt(pcov_m[1,1]),
                    fmt = "bs",
                    markerfacecolor="none",
                    label = "M-OPE",
                    capsize=3,)
            
            line2 = plt.errorbar(bz, 
                    popt_c[1],
                    yerr=np.sqrt(pcov_c[1,1]),
                    fmt = "rs",
                    markerfacecolor="none",
                    label = "C-OPE",
                    capsize=3,)

            # line3 = plt.errorbar(bz, 
            #         popt_t[0],
            #         yerr=np.sqrt(pcov_t[0,0]),
            #         fmt = "gs",
            #         markerfacecolor="none",
            #         label = "Tree Expansion",
            #         capsize=3,)



        plt.title(r"Fixed $z_3$ Fits")
        plt.xlabel(r"$z_3/a$")
        plt.ylabel(r"$\langle x^4 \rangle$")
        plt.ylim(0.,0.5)
        plt.legend([line1, line2,], [r"$N_\text{max}=4$", r"$N_\text{max}=6$"])

        if save:
            save_path = f"{smear}/2_state_matrix_results_jack/{init_char}/mellin_moms"
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(f"{save_path}/fixed_z_mm4.pdf")
        # plt.show()

            
            





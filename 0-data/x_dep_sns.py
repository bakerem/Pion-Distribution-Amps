# Author Ethan Baker, ANL/Haverford College

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit
from functions import x_dep_many_m_ope, phys_p

Nt = 128
a = 2.359 # GeV^-1
mu = 2    # GeV
P0 = 1
# N_max = 2
Nh = 0
Nl = 0
Ng = 2


bz_min_low = 2
bz_min_up  = 3

bz_max_low = 6
bz_max_up= 9

Pz_min_low = 2
Pz_min_up  = 4

Pz_max_low = 7
Pz_max_up= 10
save = False
plot = False

# initialize empty arrays for saving data
for smear in ["final_results", "final_results_eps10"]:
    for init_char in ["Z5","T5"]:
        for N_max in [2]:
            index = 0
            s1     = np.zeros(((bz_min_up-bz_min_low)*(bz_max_up-bz_max_low)*(Pz_min_up-Pz_min_low)*(Pz_max_up-Pz_max_low)))
            s1_err = np.zeros(((bz_min_up-bz_min_low)*(bz_max_up-bz_max_low)*(Pz_min_up-Pz_min_low)*(Pz_max_up-Pz_max_low)))
            s2     = np.zeros(((bz_min_up-bz_min_low)*(bz_max_up-bz_max_low)*(Pz_min_up-Pz_min_low)*(Pz_max_up-Pz_max_low)))
            s2_err = np.zeros(((bz_min_up-bz_min_low)*(bz_max_up-bz_max_low)*(Pz_min_up-Pz_min_low)*(Pz_max_up-Pz_max_low)))
            chi2s     = np.zeros(((bz_min_up-bz_min_low)*(bz_max_up-bz_max_low)*(Pz_min_up-Pz_min_low)*(Pz_max_up-Pz_max_low)))

            # create save path
            save_path = f"{smear}/2_state_matrix_results_jack/{init_char}"
            os.makedirs(save_path, exist_ok=True)
            matrix_els_full = np.load(f"{save_path}/real_matrix_el.npy")
            matrix_errs_full = np.load(f"{save_path}/real_matrix_err.npy")

            #load in data
            for bz_min in range(bz_min_low, bz_min_up):
                for bz_max in range(bz_max_low, bz_max_up):
                    for Pz_min in range(Pz_min_low, Pz_min_up):
                        for Pz_max in range(Pz_max_low, Pz_max_up):
                            bz_range = np.arange(bz_min, bz_max)  
                            Pz_range = np.arange(Pz_min, Pz_max)
                            bzPz_range = np.array((np.repeat(bz_range,len(Pz_range)),np.tile(Pz_range,len(bz_range))))
                            matrix_els = matrix_els_full[Pz_min:Pz_max,bz_min:bz_max].flatten("F")
                            matrix_errs = matrix_errs_full[Pz_min:Pz_max,bz_min:bz_max].flatten("F")

                            # function for fitting calculates ratio and wraps around Mellin-OPE of 
                            # the matrix elements
                            alphas = np.load(f"/home/bakerem/ANL/final_results/2_state_matrix_results_jack/Z5/mellin_moms/mellin_alpha_Nmax2_h0_l0.npy")
                            alpha = alphas[index]
                            def ratio(bzPz, s1, s2 ):
                                bz, Pz = bzPz_range
                                num   = x_dep_many_m_ope(bz/a, s1, s2, 0, 0, 0, 2*2, Nh, phys_p(a,Pz), init_char, alpha, Ng)
                                denom = x_dep_many_m_ope(bz/a, s1, s2, 0, 0, 0, 2*2, Nh, phys_p(a,P0), init_char, alpha, Ng)
                                ratio_result = num/denom
                                return np.real(ratio_result)
                            
                            # fit curve
                            popt, pcov = curve_fit(ratio,
                                                bzPz_range,
                                                matrix_els,
                                                sigma=matrix_errs)
                            # calculate chi^2
                            chi2 = (np.sum(
                                (matrix_els
                                    - ratio(bzPz_range, *popt)
                                ) ** 2
                                / (matrix_errs) ** 2 )
                                / (bzPz_range.shape[1] - len(popt)))

                            # save data
                            s1[index] = popt[0]
                            s1_err[index] = np.sqrt(pcov[0,0])
                            s2[index] = popt[1]
                            s2_err[index] = np.sqrt(pcov[1,1])
                            chi2s[index] = chi2

                            
                            # print(s1[index], chi2)
                            index += 1                                        

                            if save:
                                save_path = f"{smear}/2_state_matrix_results_jack/{init_char}/mellin_moms"
                                os.makedirs(save_path, exist_ok=True)
                                np.save(f"{save_path}/mellin_sns_chi2_Nmax{N_max}_h{Nh}_l{Nl}.npy", chi2s)
                                np.save(f"{save_path}/mellin_s1_Nmax{N_max}_h{Nh}_l{Nl}.npy",s1)
                                np.save(f"{save_path}/mellin_s1_err_Nmax{N_max}_h{Nh}_l{Nl}.npy", s1_err)
                                np.save(f"{save_path}/mellin_s2_Nmax{N_max}_h{Nh}_l{Nl}.npy",s2)
                                np.save(f"{save_path}/mellin_s2_err_Nmax{N_max}_h{Nh}_l{Nl}.npy", s2_err)
                                





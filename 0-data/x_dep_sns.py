# Author Ethan Baker, ANL/Haverford College

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit, minimize
from scipy.special import gamma
from functions import x_dep_many_m_ope, phys_p, chi2

Nt = 128
Ns = 55

a = 2.359 # GeV^-1
mu = 2    # GeV
P0 = 3
Nh = 0
Nl = 0
Ng = 3
delta = 0.10

bz_min_low = 2
bz_min_up  = 3

bz_max_low = 6
bz_max_up= 9

Pz_min_low = 4
Pz_min_up  = 5

Pz_max_low = 7
Pz_max_up= 10
save = False

# initialize empty arrays for saving data
for delta in [0.05,0.1, 0.2]:
    for smear in ["final_results", "final_results_eps10"]:
        for init_char in ["Z5","T5"]:
            for N_max in [2]:
                index = 0
                alpha  = np.zeros(((bz_min_up-bz_min_low)*(bz_max_up-bz_max_low)*(Pz_min_up-Pz_min_low)*(Pz_max_up-Pz_max_low)))
                s1     = np.zeros(((bz_min_up-bz_min_low)*(bz_max_up-bz_max_low)*(Pz_min_up-Pz_min_low)*(Pz_max_up-Pz_max_low)))
                s2     = np.zeros(((bz_min_up-bz_min_low)*(bz_max_up-bz_max_low)*(Pz_min_up-Pz_min_low)*(Pz_max_up-Pz_max_low)))
                s3     = np.zeros(((bz_min_up-bz_min_low)*(bz_max_up-bz_max_low)*(Pz_min_up-Pz_min_low)*(Pz_max_up-Pz_max_low)))
                chi2s  = np.zeros(((bz_min_up-bz_min_low)*(bz_max_up-bz_max_low)*(Pz_min_up-Pz_min_low)*(Pz_max_up-Pz_max_low)))

                # create save path
                save_path = f"{smear}/2_state_matrix_results_jack/{init_char}"
                os.makedirs(save_path, exist_ok=True)
                matrix_els_full = np.load(f"{save_path}/real_matrix_el.npy")
                matrix_el_js = np.load(f"{save_path}/real_matrix_elj.npy")

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
                                def ratio(bz, Pz, alpha, s1, s2, ):
                                    # bz, Pz = bzPz_range
                                    num   = x_dep_many_m_ope(bz/a, alpha, s1, s2, 0, 0, 0, 0, 0, 0, 2*2, Nh, phys_p(a,Pz), init_char, Ng)
                                    denom = x_dep_many_m_ope(bz/a, alpha, s1, s2, 0, 0, 0, 0, 0, 0, 2*2, Nh, phys_p(a,P0), init_char, Ng)
                                    ratio_result = num/denom
                                    return np.real(ratio_result)
                                
                                cov = np.zeros((len(bzPz_range.transpose()), len(bzPz_range.transpose())))
                                for i, bzPz in enumerate(bzPz_range.transpose()):
                                    bz, Pz = bzPz
                                    for j, bzPz1 in enumerate(bzPz_range.transpose()):
                                        bz1, Pz1 = bzPz1
                                        cov[i,j] = np.average((matrix_el_js[Pz, bz,:] - np.average(matrix_el_js[Pz,bz,:]))*
                                                                (matrix_el_js[Pz1, bz1,:] - np.average(matrix_el_js[Pz1,bz1,:])))
                                cov = (Ns-1)*cov
                                # fit curve
                                # popt, pcov = curve_fit(ratio,
                                #                     bzPz_range,
                                #                     matrix_els,
                                #                    # delta=0.05,
                                #                     maxfev=2000,
                                #                     sigma=cov)
                                def chi_wrapper(sns):
                                    return chi2(sns, bzPz_range, delta, cov,ratio, matrix_els_full)
                                res = minimize(chi_wrapper, 
                                            [0.1, 0.004, 0.004,], 
                                            method="Nelder-Mead")
                                popt = res.x
                                # calculate chi^2
                                chi2_result = (np.sum(
                                    (matrix_els
                                        - ratio(bzPz_range[0], bzPz_range[1], *popt)
                                    ) ** 2
                                    / (matrix_errs) ** 2 )
                                    / (bzPz_range.shape[1] - len(popt)))

                                # save data
                                # alpha[index] = popt[0]
                                s1[index] = popt[0]
                                s2[index] = popt[1]
                                # print(s1)
      
                                chi2s[index] = chi2_result
                                A = (2**(1+2*alpha)*gamma(1.5+alpha))/(np.sqrt(np.pi)*gamma(1+alpha))
                                mm2 = (5+2*alpha+2*s1+6*alpha*s1+4*alpha**2*s1)/(15+16*alpha+4*alpha**2)
                                mm4 = (3*(9+2*alpha)*(7+2*alpha+4*(1+a)*(1+2*alpha)*s1) + 
                                    4*(1+alpha)*(2+alpha)*(1+2*alpha)*(3+2*alpha)*s2)/((3+2*alpha)*(5+2*alpha)*(7+2*alpha)*(9+2*alpha))
                                print(mm2[index], mm4[index], chi2_result)
                                
                                # print(s1[index], s2[index], chi2_result)
                                index += 1                                        

                                if save:
                                    save_path = f"{smear}/2_state_matrix_results_jack/{init_char}/x_depend"
                                    os.makedirs(save_path, exist_ok=True)
                                    np.save(f"{save_path}/mellin_sns_chi2_Nmax{N_max}_h{Nh}_l{Nl}_Ng{Ng}_d{int(100*delta)}.npy", chi2s)
                                    np.save(f"{save_path}/mellin_s1_Nmax{N_max}_h{Nh}_l{Nl}_Ng{Ng}_d{int(100*delta)}.npy",s1)
                                    np.save(f"{save_path}/mellin_s2_Nmax{N_max}_h{Nh}_l{Nl}_Ng{Ng}_d{int(100*delta)}.npy",s2)
                                    np.save(f"{save_path}/mellin_s3_Nmax{N_max}_h{Nh}_l{Nl}_Ng{Ng}_d{int(100*delta)}.npy",s3)

                                    





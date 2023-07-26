# Author Ethan Baker, ANL/Haverford College

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit
from functions import x_dep1_m_ope, phys_p

Nt = 128
Ns = 55
a = 2.359 # GeV^-1
P0 = 1
gamma_E = 0.57721

bz_min_low = 2
bz_min_up  = 3

bz_max_low = 3
bz_max_up= 6

Pz_min_low = 2
Pz_min_up  = 6

Pz_max_low = 9
Pz_max_up= 10
save = True
plot = False

# initialize empty arrays for saving data
for smear in ["final_results","final_results_eps10"]:
    for init_char in ["T5"]:
        for Nh in [0]:
            for N_max in [2]:
                for kappa in np.arange(0.9,1.4,0.01):
                    index = 0
                    alpha     = np.zeros(((bz_min_up-bz_min_low)*(bz_max_up-bz_max_low)*(Pz_min_up-Pz_min_low)*(Pz_max_up-Pz_max_low)))
                    alpha_err = np.zeros(((bz_min_up-bz_min_low)*(bz_max_up-bz_max_low)*(Pz_min_up-Pz_min_low)*(Pz_max_up-Pz_max_low)))
                    chi2s     = np.zeros(((bz_min_up-bz_min_low)*(bz_max_up-bz_max_low)*(Pz_min_up-Pz_min_low)*(Pz_max_up-Pz_max_low)))

                    # create save path
                    save_path = f"{smear}/2_state_matrix_results_jack/{init_char}"
                    os.makedirs(save_path, exist_ok=True)
                    matrix_els_full = np.load(f"{save_path}/real_matrix_el_P0{P0}.npy")
                    matrix_errs_full = np.load(f"{save_path}/real_matrix_err_P0{P0}.npy")
                    matrix_el_js = np.load(f"{save_path}/real_matrix_elj_P0{P0}.npy")

                    #load in data
                    for bz_min in range(bz_min_low, bz_min_up):
                        for bz_max in range(bz_max_low, bz_max_up):
                            for Pz_min in range(Pz_min_low, Pz_min_up):
                                for Pz_max in range(Pz_max_low, Pz_max_up):
                                    bz_range = np.arange(bz_min, bz_max+1)  
                                    Pz_range = np.arange(Pz_min, Pz_max+1)
                                    bzPz_range = np.array((np.repeat(bz_range,len(Pz_range)),np.tile(Pz_range,len(bz_range))))
                                    matrix_els = matrix_els_full[Pz_min:Pz_max+1,bz_min:bz_max+1].flatten("F")
                                    matrix_errs = matrix_errs_full[Pz_min:Pz_max+1,bz_min:bz_max+1].flatten("F")
                                    
                                    cov = np.zeros((len(bzPz_range.transpose()), len(bzPz_range.transpose())))
                                    for i, bzPz in enumerate(bzPz_range.transpose()):
                                        bz, Pz = bzPz
                                        for j, bzPz1 in enumerate(bzPz_range.transpose()):
                                            bz1, Pz1 = bzPz1
                                            cov[i,j] = np.average((matrix_el_js[Pz, bz,:] - np.average(matrix_el_js[Pz,bz,:]))*
                                                                    (matrix_el_js[Pz1, bz1,:] - np.average(matrix_el_js[Pz1,bz1,:])))
                                    cov = (Ns-1)*cov
                                    # function for fitting calculates ratio and wraps around Mellin-OPE of 
                                    # the matrix elements
                                    if Nh == 0:
                                        def ratio(bzPz, alpha ):
                                            bz, Pz = bzPz_range
                                            mu = kappa*2*np.exp(-gamma_E)/(bz/a)
                                            alpha_s = 0.303
                                            num   = x_dep1_m_ope(bz/a, alpha, 0, 0, 0, 2*N_max, Nh, phys_p(a,Pz), alpha_s, mu, init_char)
                                            denom = x_dep1_m_ope(bz/a, alpha, 0, 0, 0, 2*N_max, Nh, phys_p(a,P0), alpha_s, mu, init_char)
                                            ratio_result = num/denom
                                            return np.real(ratio_result)
                                        
                                        popt, pcov = curve_fit(ratio,
                                                        bzPz_range,
                                                        matrix_els,
                                                        # bounds=[(0),(np.inf)],
                                                        sigma=cov)
                                    elif Nh == 1:
                                        def ratio(bzPz, alpha, h0):
                                            bz, Pz = bzPz_range
                                            mu = kappa*2*np.exp(-gamma_E)/(bz/a)
                                            alpha_s = 0.303
                                            num   = x_dep1_m_ope(bz/a, alpha, 0, h0, 0, 2*N_max, Nh, phys_p(a,Pz), alpha_s, mu, init_char)
                                            denom = x_dep1_m_ope(bz/a, alpha, 0, h0, 0, 2*N_max, Nh, phys_p(a,P0), alpha_s, mu, init_char)
                                            ratio_result = num/denom
                                            return np.real(ratio_result)
                                        
                                        popt, pcov = curve_fit(ratio,
                                                        bzPz_range,
                                                        matrix_els,
                                                        p0 = (0, -0.5),
                                                        bounds = [(-np.inf, -1),(1, 1)],
                                                        maxfev = 2000,
                                                        sigma=cov)
                                        
                                    
                                    # calculate chi^2
                                    chi2 = (np.sum(
                                        (matrix_els
                                            - ratio(bzPz_range, *popt)
                                        ) ** 2
                                        / (matrix_errs) ** 2 )
                                        / (bzPz_range.shape[1] - len(popt)))

                                    # save data
                                    alpha[index] = popt[0]
                                    alpha_err[index] = np.sqrt(pcov[0,0])
                                    chi2s[index] = chi2

                                    print(alpha[index], alpha_err[index], chi2)
                                    mm2 = 1/(3+2*alpha[index])
                                    mm4 = 3/(15+16*alpha[index]+4*alpha[index]**2)
                                    # print(mm2, mm4)
                                    index += 1        

                                    if save:
                                        save_path = f"{smear}/2_state_matrix_results_jack/{init_char}/mellin_moms"
                                        os.makedirs(save_path, exist_ok=True)
                                        np.save(f"{save_path}/mellin_alpha_chi2_Nmax{N_max}_h{Nh}_l{0}_P0{P0}_resum_k"+"%.2f"%kappa+".npy", chi2s)
                                        np.save(f"{save_path}/mellin_alpha_Nmax{N_max}_h{Nh}_l{0}_P0{P0}_resum_k"+"%.2f"%kappa+".npy",alpha)
                                        np.save(f"{save_path}/mellin_alpha_err_Nmax{N_max}_h{Nh}_l{0}_P0{P0}_resum_k"+"%.2f"%kappa+".npy", alpha_err)
                                        





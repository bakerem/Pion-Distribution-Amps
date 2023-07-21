# Author Ethan Baker, ANL/Haverford College

import numpy as np
import os
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from functions import m_ope, phys_p, alpha_func, chi2_orig

Nt = 128
a = 2.359 # GeV^-1
gamma_E = 0.57721
# alpha_s = 0.303
# mu = 2    # GeV
P0 = 1
N_max = 2
Ns = 55


bz_min_low = 2
bz_min_up  = 3

bz_max_low = 5
bz_max_up= 6

Pz_min_low = 2
Pz_min_up  = 6

Pz_max_low = 9
Pz_max_up= 10
save = True

# initialize empty arrays for saving data
for smear in ["final_results", "final_results_eps10"]:
    for init_char in ["T5"]:
        print(init_char)
        for Nh in [0]:
            for Nl in [0]:
                for kappa in np.arange(0.9,1.4,0.01):
                    # initialize empty arrays for saving data
                    moms2     = np.zeros(((bz_min_up-bz_min_low)*(bz_max_up-bz_max_low)*(Pz_min_up-Pz_min_low)*(Pz_max_up-Pz_max_low)))
                    moms2_err = np.zeros(((bz_min_up-bz_min_low)*(bz_max_up-bz_max_low)*(Pz_min_up-Pz_min_low)*(Pz_max_up-Pz_max_low)))
                    moms4     = np.zeros(((bz_min_up-bz_min_low)*(bz_max_up-bz_max_low)*(Pz_min_up-Pz_min_low)*(Pz_max_up-Pz_max_low)))
                    moms4_err = np.zeros(((bz_min_up-bz_min_low)*(bz_max_up-bz_max_low)*(Pz_min_up-Pz_min_low)*(Pz_max_up-Pz_max_low)))
                    moms6     = np.zeros(((bz_min_up-bz_min_low)*(bz_max_up-bz_max_low)*(Pz_min_up-Pz_min_low)*(Pz_max_up-Pz_max_low)))
                    moms6_err = np.zeros(((bz_min_up-bz_min_low)*(bz_max_up-bz_max_low)*(Pz_min_up-Pz_min_low)*(Pz_max_up-Pz_max_low)))
                    moms8     = np.zeros(((bz_min_up-bz_min_low)*(bz_max_up-bz_max_low)*(Pz_min_up-Pz_min_low)*(Pz_max_up-Pz_max_low)))
                    moms8_err = np.zeros(((bz_min_up-bz_min_low)*(bz_max_up-bz_max_low)*(Pz_min_up-Pz_min_low)*(Pz_max_up-Pz_max_low)))
                    chi2s     = np.zeros(((bz_min_up-bz_min_low)*(bz_max_up-bz_max_low)*(Pz_min_up-Pz_min_low)*(Pz_max_up-Pz_max_low)))
                    h0s       = np.zeros(((bz_min_up-bz_min_low)*(bz_max_up-bz_max_low)*(Pz_min_up-Pz_min_low)*(Pz_max_up-Pz_max_low)))
                    h0s_err   = np.zeros(((bz_min_up-bz_min_low)*(bz_max_up-bz_max_low)*(Pz_min_up-Pz_min_low)*(Pz_max_up-Pz_max_low)))
                    # create save path
                    save_path = f"{smear}/2_state_matrix_results_jack/{init_char}"
                    os.makedirs(save_path, exist_ok=True)
                    # load in data
                    matrix_els_full = np.load(f"{save_path}/real_matrix_el_P0{P0}.npy")
                    matrix_errs_full = np.load(f"{save_path}/real_matrix_err_P0{P0}.npy")
                    matrix_el_js = np.load(f"{save_path}/real_matrix_elj_P0{P0}.npy")
                    # iterate through a range of fitting parameters, as provided above
                    index = 0
                    for bz_min in range(bz_min_low, bz_min_up):
                        for bz_max in range(bz_max_low, bz_max_up):
                            for Pz_min in range(Pz_min_low, Pz_min_up):
                                for Pz_max in range(Pz_max_low, Pz_max_up):
                                    # create 2-D array from fitting ranges and flatten y-data to be in the right
                                    # shape for use by scipy.curvefit
                                    bz_range = np.arange(bz_min, bz_max+1)  
                                    Pz_range = np.arange(Pz_min, Pz_max+1)
                                    bzPz_range = np.array((np.repeat(bz_range,len(Pz_range)),np.tile(Pz_range,len(bz_range))))
                                    matrix_els = matrix_els_full[Pz_min:Pz_max+1,bz_min:bz_max+1].flatten("F")
                                    matrix_errs = matrix_errs_full[Pz_min:Pz_max+1,bz_min:bz_max+1].flatten("F")

                                    # create covariance matrix
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
                                    # different functions for each of the combinations of Nh and Nl
                                    if Nh == 0 and Nl == 0:
                                        def ratio(bzPz, mm2, mm4,):
                                            bz, Pz = bzPz_range
                                            mu = kappa*2*np.exp(-gamma_E)/(bz/a)
                                            # mu = 2
                                            alpha_s = 0.303
                                            num   = m_ope(bz/a, mm2, mm4, 0, 0, 0, 0, 0, 2*N_max, Nh, phys_p(a,Pz), alpha_s, mu, init_char)
                                            denom = m_ope(bz/a, mm2, mm4, 0, 0, 0, 0, 0, 2*N_max, Nh, phys_p(a,P0), alpha_s, mu, init_char)
                                            ratio_result = num/denom
                                            return np.real(ratio_result)
                                        # popts = np.zeros((Ns,2))
                                        # pcov0s = np.zeros((Ns,))
                                        # pcov1s = np.zeros((Ns,))

                                        # for s in range(Ns):
                                        #     popt_j, pcov_j = curve_fit(ratio,
                                        #                     bzPz_range,
                                        #                     matrix_el_js[Pz_min:Pz_max+1,bz_min:bz_max+1,s].flatten("F"),
                                        #                     p0=(0.33,0.15,),
                                        #                     sigma=cov,
                                        #                     )
                                        #     popts[s,:] = popt_j
                                        #     pcov0s[s] = np.sqrt(pcov_j[0,0])
                                        #     pcov1s[s] = np.sqrt(pcov_j[1,1])
                                    elif Nh == 0 and Nl == 1:
                                        def ratio(bzPz, mm2,mm4, l ):
                                            bz, Pz = bzPz_range
                                            mu = 2*np.exp(-gamma_E)/(bz/a)
                                            alpha_s = alpha_func(mu)
                                            mu = 2
                                            alpha_s = 0.303
                                            num   = m_ope(bz/a, mm2, mm4, 0, 0, l, 0, 0, 2*N_max, Nh, phys_p(a,Pz), alpha_s, mu, init_char)
                                            denom = m_ope(bz/a, mm2, mm4, 0, 0, l, 0, 0, 2*N_max, Nh, phys_p(a,P0), alpha_s, mu, init_char)
                                            ratio_result = num/denom
                                            return np.real(ratio_result)
                                        
                                        popt, pcov = curve_fit(ratio,
                                                        bzPz_range,
                                                        matrix_els,
                                                        p0=(0.3,0.15,0.0008),
                                                        bounds =[(-np.inf,-np.inf, -0.1), (1/3,np.inf, 0.1)],
                                                        sigma=cov)
                                    elif Nh == 1 and Nl == 0:
                                        def ratio(bzPz, mm2, mm4, h0, ):
                                            bz, Pz = bzPz_range
                                            mu = kappa*2*np.exp(-gamma_E)/(bz/a)
                                            alpha_s = alpha_func(mu)
                                            # mu = 2
                                            alpha_s = 0.303
                                            num   = m_ope(bz/a, mm2, mm4, 0, 0, 0, h0, 0, 2*N_max, 1, phys_p(a,Pz), alpha_s, mu, init_char)
                                            denom = m_ope(bz/a, mm2, mm4, 0, 0, 0, h0, 0, 2*N_max, 1, phys_p(a,P0), alpha_s, mu, init_char)
                                            ratio_result = num/denom
                                            return np.real(ratio_result)
                                        
                                        popt, pcov = curve_fit(ratio,
                                                        bzPz_range,
                                                        matrix_els,
                                                        p0=(0.25,0.15,0.07,),
                                                        bounds =[(-np.inf,-np.inf, -1,), (np.inf,np.inf, 1,)],
                                                        sigma=cov)
                                    elif Nh == 1 and Nl == 1:
                                        def ratio(bzPz, mm2, mm4, h0, l ):
                                            bz, Pz = bzPz_range
                                            mu = 2*np.exp(-gamma_E)/(bz/a)
                                            alpha_s = alpha_func(mu)
                                            num   = m_ope(bz/a, mm2, mm4, 0, 0, l, h0, 0, 2*N_max, Nh, phys_p(a,Pz), alpha_s, mu, init_char)
                                            denom = m_ope(bz/a, mm2, mm4, 0, 0, l, h0, 0, 2*N_max, Nh, phys_p(a,P0), alpha_s, mu, init_char)
                                            ratio_result = num/denom
                                            return np.real(ratio_result)
                                        
                                        popt, pcov = curve_fit(ratio,
                                                        bzPz_range,
                                                        matrix_els,
                                                        p0=(0.33,0.15,0.007,0.0008),
                                                        bounds =[(-np.inf,-np.inf, -1, -0.1), (np.inf,np.inf, 1, 0.1)],
                                                        sigma=cov)

                                    # popts_samples = np.zeros((Ns,2))
                                    # pcov0s_samples = np.zeros((Ns,))
                                    # pcov1s_samples = np.zeros((Ns,))
                                    # for i in range(Ns):
                                    #     popts_samples[i] = np.average(np.delete(popts, i, axis=0), axis=0)
                                    #     pcov0s_samples   = np.average(np.delete(pcov0s,i))
                                    #     pcov1s_samples   = np.average(np.delete(pcov1s,i))

                                    # popt = np.average(popts_samples, axis=0)
                                    # pcov0 = np.average(pcov0s_samples)
                                    # pcov1 = np.average(pcov1s_samples)

                                    # fit curve
                                    popt, pcov = curve_fit(ratio,
                                                        bzPz_range,
                                                        matrix_els,
                                                        p0=(0.33,0.15,),
                                                        # method="dogbox",
                                                        sigma=cov)

                                    # calculate chi^2
                                    # def ratio_chi2(bz, Pz, mm2, mm4, h0 ):
                                    #         mu = kappa*2*np.exp(-gamma_E)/(bz/a)
                                    #         alpha_s = alpha_func(mu)
                                    #         # mu = 2
                                    #         # alpha_s = 0.303
                                    #         num   = m_ope(bz/a, mm2, mm4, 0, 0, 0, h0, 0, 2*N_max, Nh, phys_p(a,Pz), alpha_s, mu, init_char)
                                    #         denom = m_ope(bz/a, mm2, mm4, 0, 0, 0, h0, 0, 2*N_max, Nh, phys_p(a,P0), alpha_s, mu, init_char)
                                    #         ratio_result = num/denom
                                    #         return np.real(ratio_result)
                                    # chi2 = chi2_orig(popt, bzPz_range, cov, ratio_chi2, matrix_els_full)/(bzPz_range.shape[1]-len(popt))
                                    chi2 = (np.sum(
                                            (matrix_els
                                                - ratio(bzPz_range, *popt)
                                            ) ** 2
                                            / (matrix_errs) ** 2 )
                                            / (bzPz_range.shape[1] - len(popt)))
                                    # save data
                                    moms2[index] = popt[0]
                                    moms2_err[index] = np.sqrt(pcov[0,0])
                                    # moms2_err[index] = pcov0
                                    moms4[index] = popt[1]
                                    moms4_err[index] = np.sqrt(pcov[1,1])
                                    # moms4_err[index] = pcov1
                                    # h0s[index] = popt[2]
                                    # h0s_err[index] = np.sqrt(pcov[2,2])
                                    # moms8[index] = popt[3]
                                    # moms8_err[index] = np.sqrt(pcov[3,3])
                                    chi2s[index] = chi2
                                    # print(popt)
                                    # print(kappa, popt[0], popt[1],  chi2)
                                    print(moms2[index], moms2_err[index], moms4[index], chi2)
                                    
                                    index += 1
                                    
                                    if save:
                                        save_path = f"{smear}/2_state_matrix_results_jack/{init_char}/mellin_moms"
                                        os.makedirs(save_path, exist_ok=True)
                                        np.save(f"{save_path}/mellin_chi2_Nmax{N_max}_h{Nh}_l{Nl}_P0{P0}_zmax{bz_max}_resum_k"+"%.2f"%kappa+".npy", chi2s)
                                        np.save(f"{save_path}/mellin_moms2_Nmax{N_max}_h{Nh}_l{Nl}_P0{P0}_zmax{bz_max}_resum_k"+"%.2f"%kappa+".npy",moms2)
                                        np.save(f"{save_path}/mellin_moms2_err_Nmax{N_max}_h{Nh}_l{Nl}_P0{P0}_zmax{bz_max}_resum_k"+"%.2f"%kappa+".npy", moms2_err)
                                        np.save(f"{save_path}/mellin_moms4_Nmax{N_max}_h{Nh}_l{Nl}_P0{P0}_zmax{bz_max}_resum_k"+"%.2f"%kappa+".npy",moms4)
                                        np.save(f"{save_path}/mellin_moms4_err_Nmax{N_max}_h{Nh}_l{Nl}_P0{P0}_zmax{bz_max}_resum_k"+"%.2f"%kappa+".npy", moms4_err)
                                        # np.save(f"{save_path}/mellin_h0_Nmax{N_max}_h{Nh}_l{Nl}_P0{P0}_zmax{bz_max}_resum_k"+"%.2f"%kappa+".npy", h0s)
                                        # np.save(f"{save_path}/mellin_h0_err_Nmax{N_max}_h{Nh}_l{Nl}_P0{P0}_zmax{bz_max}_resum_k"+"%.2f"%kappa+".npy", h0s_err)

                                        # np.save(f"{save_path}/mellin_moms6_Nmax{N_max}_h{Nh}_l{Nl}.npy",moms6)
                                        # np.save(f"{save_path}/mellin_moms6_err_Nmax{N_max}_h{Nh}_l{Nl}.npy", moms6_err)
                                        # np.save(f"{save_path}/mellin_moms8_Nmax{N_max}_h{Nh}_l{Nl}.npy",moms8)
                                        # np.save(f"{save_path}/mellin_moms8_err_Nmax{N_max}_h{Nh}_l{Nl}.npy", moms8_err)
print(np.average(moms2))
print(f"{save_path}/mellin_moms4_Nmax{N_max}_h{Nh}_l{Nl}_P0{P0}_zmax{bz_max}_resum_k")




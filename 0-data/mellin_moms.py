import numpy as np
import os
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from functions import m_ope, phys_p

"""
Ethan Baker, ANL/Haverford College

The script for calculating the Mellin moments from the lattice data. There are
several if statements used to distinguish between if higher twist or lattice
corrections are included. The main body of the script is a for statement
that loops through different fitting parameters. The ranges for bz and Pz
are denoted by _low and _max subscripts and the upper limit is exclusive. So,
if you want the lower Pz limit to range from 2 to 5, set Pz_min_low = 2 and
Pz_min_low = 6. 

The results are saved in an array that contains the results for set of Pz and bz
ranges and a new array is produced for each Nh, and kappa. In general, it is best
practice to only run over one set of [bz_min, bz_max], because it is easy to 
concatenate these results later. 

"""


Nt = 128
a = 1/2.359 # GeV^-1
gamma_E = 0.57721 # Euler constant

P0 = 1
N_max = 2
Ns = 55


bz_min_low = 2
bz_min_up  = 3

bz_max_low = 3
bz_max_up= 4

Pz_min_low = 2
Pz_min_up  = 6

Pz_max_low = 9
Pz_max_up= 10
save = False

# initialize empty arrays for saving data
for smear in ["final_results", "final_results_flow10"]:
    for init_char in ["T5"]:
        print(init_char)
        for Nh in [0]:
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
                                cov_inv = np.linalg.inv(cov)

                                # function for fitting calculates ratio and wraps 
                                # around Mellin-OPE of the matrix elements. 
                                # Different functions for each Nh
                                if Nh == 0:
                                    def ratio(bzPz, mm2, mm4,):
                                        bz, Pz = bzPz_range
                                        mu = kappa*2*np.exp(-gamma_E)/(bz*a)
                                        # mu = 2
                                        alpha_s = 0.303
                                        num   = m_ope(bz*a, mm2, mm4, 0, 0, 0, 0, 0, 2*N_max, Nh, phys_p(a,Pz), alpha_s, mu, init_char)
                                        denom = m_ope(bz*a, mm2, mm4, 0, 0, 0, 0, 0, 2*N_max, Nh, phys_p(a,P0), alpha_s, mu, init_char)
                                        ratio_result = num/denom
                                        return np.real(ratio_result)
                                
                                    popt, pcov = curve_fit(ratio,
                                                    bzPz_range,
                                                    matrix_els,
                                                    sigma=cov)
                                elif Nh == 1:
                                    def ratio(bzPz, mm2, mm4, h0,alpha_s,):
                                        bz, Pz = bzPz_range
                                        mu = kappa*2*np.exp(-gamma_E)/(bz*a)
                                        alpha_s = 0.303
                                        num   = m_ope(bz*a, mm2, mm4, 0, 0, 0, h0, 0, 2*N_max, 1, phys_p(a,Pz), alpha_s, mu, init_char)
                                        denom = m_ope(bz*a, mm2, mm4, 0, 0, 0, h0, 0, 2*N_max, 1, phys_p(a,P0), alpha_s, mu, init_char)
                                        ratio_result = num/denom
                                        return np.real(ratio_result)
                                    
                                    popt, pcov = curve_fit(ratio,
                                                    bzPz_range,
                                                    matrix_els,
                                                    p0=(0.25,0.15,0.07,0.303),
                                                    bounds =[(-np.inf,-np.inf, -1,), (np.inf,np.inf, 1,)],
                                                    sigma=cov)
                                    h0s[index] = popt[2]
                                    h0s_err[index] = np.sqrt(pcov[2,2])
                                
                                # calculate chi^2
                                chi2 = (np.sum(
                                        (matrix_els
                                            - ratio(bzPz_range, *popt)
                                        ) ** 2
                                        / (matrix_errs) ** 2 )
                                        / (bzPz_range.shape[1] - len(popt)))
                                # save data
                                moms2[index] = popt[0]
                                moms2_err[index] = np.sqrt(pcov[0,0])
                                moms4[index] = popt[1]
                                moms4_err[index] = np.sqrt(pcov[1,1])
                                
    
                                chi2s[index] = chi2
                                print(moms2[index], moms2_err[index], moms4[index], chi2)
                                
                                index += 1
                                
                                if save:
                                    save_path = f"{smear}/2_state_matrix_results_jack/{init_char}/mellin_moms"
                                    os.makedirs(save_path, exist_ok=True)
                                    np.save(f"{save_path}/mellin_chi2_Nmax{N_max}_h{Nh}_l{0}_P0{P0}_zmax{bz_max}_resum_k"+"%.2f"%kappa+".npy", chi2s)
                                    np.save(f"{save_path}/mellin_moms2_Nmax{N_max}_h{Nh}_l{0}_P0{P0}_zmax{bz_max}_resum_k"+"%.2f"%kappa+".npy",moms2)
                                    np.save(f"{save_path}/mellin_moms2_err_Nmax{N_max}_h{Nh}_l{0}_P0{P0}_zmax{bz_max}_resum_k"+"%.2f"%kappa+".npy", moms2_err)
                                    np.save(f"{save_path}/mellin_moms4_Nmax{N_max}_h{Nh}_l{0}_P0{P0}_zmax{bz_max}_resum_k"+"%.2f"%kappa+".npy",moms4)
                                    np.save(f"{save_path}/mellin_moms4_err_Nmax{N_max}_h{Nh}_l{0}_P0{P0}_zmax{bz_max}_resum_k"+"%.2f"%kappa+".npy", moms4_err)
                                    np.save(f"{save_path}/mellin_h0_Nmax{N_max}_h{Nh}_l{0}_P0{P0}_zmax{bz_max}_resum_k"+"%.2f"%kappa+".npy", h0s)
                                    np.save(f"{save_path}/mellin_h0_err_Nmax{N_max}_h{Nh}_l{0}_P0{P0}_zmax{bz_max}_resum_k"+"%.2f"%kappa+".npy", h0s_err)

                                    # np.save(f"{save_path}/mellin_moms6_Nmax{N_max}_h{Nh}_l{0}.npy",moms6)
                                    # np.save(f"{save_path}/mellin_moms6_err_Nmax{N_max}_h{Nh}_l{0}.npy", moms6_err)
                                    # np.save(f"{save_path}/mellin_moms8_Nmax{N_max}_h{Nh}_l{0}.npy",moms8)
                                    # np.save(f"{save_path}/mellin_moms8_err_Nmax{N_max}_h{Nh}_l{0}.npy", moms8_err)




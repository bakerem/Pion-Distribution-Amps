# Author Ethan Baker, ANL/Haverford College

import numpy as np
import os
from scipy.optimize import curve_fit
from functions import m_ope, phys_p

Nt = 128
a = 2.359 # GeV^-1
mu = 2    # GeV
P0 = 3
N_max = 2
Ns = 55


bz_min_low = 2
bz_min_up  = 3

bz_max_low = 6
bz_max_up= 9

Pz_min_low = 4
Pz_min_up  = 5

Pz_max_low = 7
Pz_max_up= 10
save = True

# initialize empty arrays for saving data
for smear in ["final_results", "final_results_eps10"]:
    for init_char in ["Z5","T5"]:
        print(init_char)
        for Nh in [0,1]:
            for Nl in [0,1]:
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
                h0_errs   = np.zeros(((bz_min_up-bz_min_low)*(bz_max_up-bz_max_low)*(Pz_min_up-Pz_min_low)*(Pz_max_up-Pz_max_low)))
                ls        = np.zeros(((bz_min_up-bz_min_low)*(bz_max_up-bz_max_low)*(Pz_min_up-Pz_min_low)*(Pz_max_up-Pz_max_low)))
                l_errs    = np.zeros(((bz_min_up-bz_min_low)*(bz_max_up-bz_max_low)*(Pz_min_up-Pz_min_low)*(Pz_max_up-Pz_max_low)))
                # create save path
                save_path = f"{smear}/2_state_matrix_results_jack/{init_char}"
                os.makedirs(save_path, exist_ok=True)
                matrix_els_full = np.load(f"{save_path}/real_matrix_el.npy")
                matrix_errs_full = np.load(f"{save_path}/real_matrix_err.npy")
                matrix_el_js = np.load(f"{save_path}/real_matrix_elj.npy")
                #load in data
                index = 0
                for bz_min in range(bz_min_low, bz_min_up):
                    for bz_max in range(bz_max_low, bz_max_up):
                        for Pz_min in range(Pz_min_low, Pz_min_up):
                            for Pz_max in range(Pz_max_low, Pz_max_up):
                                bz_range = np.arange(bz_min, bz_max)  
                                Pz_range = np.arange(Pz_min, Pz_max)
                                bzPz_range = np.array((np.repeat(bz_range,len(Pz_range)),np.tile(Pz_range,len(bz_range))))
                                matrix_els = matrix_els_full[Pz_min:Pz_max,bz_min:bz_max].flatten("F")
                                matrix_errs = matrix_errs_full[Pz_min:Pz_max,bz_min:bz_max].flatten("F")
                                
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
                                if Nh == 0 and Nl == 0:
                                    def ratio(bzPz, mm2, mm4,  ):
                                        bz, Pz = bzPz_range
                                        num   = m_ope(bz/a, mm2, mm4, 0, 0, 0, 0, 0, 2*N_max, Nh, phys_p(a,Pz), init_char)
                                        denom = m_ope(bz/a, mm2, mm4, 0, 0, 0, 0, 0, 2*N_max, Nh, phys_p(a,P0), init_char)
                                        ratio_result = num/denom
                                        return np.real(ratio_result)
                                elif Nh == 0 and Nl == 1:
                                    def ratio(bzPz, mm2,mm4, l ):
                                        bz, Pz = bzPz_range
                                        num   = m_ope(bz/a, mm2, mm4, 0, 0, l, 0, 0, 2*N_max, Nh, phys_p(a,Pz), init_char)
                                        denom = m_ope(bz/a, mm2, mm4, 0, 0, l, 0, 0, 2*N_max, Nh, phys_p(a,P0), init_char)
                                        ratio_result = num/denom
                                        return np.real(ratio_result)
                                elif Nh == 1 and Nl == 0:
                                    def ratio(bzPz, mm2, mm4, h0 ):
                                        bz, Pz = bzPz_range
                                        num   = m_ope(bz/a, mm2, mm4, 0, 0, 0, h0, 0, 2*N_max, Nh, phys_p(a,Pz), init_char)
                                        denom = m_ope(bz/a, mm2, mm4, 0, 0, 0, h0, 0, 2*N_max, Nh, phys_p(a,P0), init_char)
                                        ratio_result = num/denom
                                        return np.real(ratio_result)
                                elif Nh == 1 and Nl == 1:
                                    def ratio(bzPz, mm2,mm4, h0, l ):
                                        bz, Pz = bzPz_range
                                        num   = m_ope(bz/a, mm2, mm4, 0, 0, l, h0, 0, 2*N_max, Nh, phys_p(a,Pz), init_char)
                                        denom = m_ope(bz/a, mm2, mm4, 0, 0, l, h0, 0, 2*N_max, Nh, phys_p(a,P0), init_char)
                                        ratio_result = num/denom
                                        return np.real(ratio_result)

                                
                                
                                # fit curve
                                popt, pcov = curve_fit(ratio,
                                                    bzPz_range,
                                                    matrix_els,
                                                    sigma=cov)
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
                                # moms6[index] = popt[2]
                                # moms6_err[index] = np.sqrt(pcov[2,2])
                                # moms8[index] = popt[3]
                                # moms8_err[index] = np.sqrt(pcov[3,3])
                                chi2s[index] = chi2
                            
                                print(moms4[index])
                                index += 1
                            
                                if save:
                                    save_path = f"{smear}/2_state_matrix_results_jack/{init_char}/mellin_moms"
                                    os.makedirs(save_path, exist_ok=True)
                                    np.save(f"{save_path}/mellin_chi2_Nmax{N_max}_h{Nh}_l{Nl}.npy", chi2s)
                                    np.save(f"{save_path}/mellin_moms2_Nmax{N_max}_h{Nh}_l{Nl}.npy",moms2)
                                    np.save(f"{save_path}/mellin_moms2_err_Nmax{N_max}_h{Nh}_l{Nl}.npy", moms2_err)
                                    np.save(f"{save_path}/mellin_moms4_Nmax{N_max}_h{Nh}_l{Nl}.npy",moms4)
                                    np.save(f"{save_path}/mellin_moms4_err_Nmax{N_max}_h{Nh}_l{Nl}.npy", moms4_err)
                                    # np.save(f"{save_path}/mellin_moms6_Nmax{N_max}_h{Nh}_l{Nl}.npy",moms6)
                                    # np.save(f"{save_path}/mellin_moms6_err_Nmax{N_max}_h{Nh}_l{Nl}.npy", moms6_err)
                                    # np.save(f"{save_path}/mellin_moms8_Nmax{N_max}_h{Nh}_l{Nl}.npy",moms8)
                                    # np.save(f"{save_path}/mellin_moms8_err_Nmax{N_max}_h{Nh}_l{Nl}.npy", moms8_err)





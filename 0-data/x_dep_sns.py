import numpy as np
import os
from scipy.optimize import minimize
from scipy.special import gamma
from functions import x_dep_many_m_ope, phys_p, chi2

"""
Ethan Baker, ANL/Haverford College

Essentially follows the same structure as "mellin_moms.py," so for more info
look at the documentation there. There is a slightly different fitting function
that incorporates sns into the Mellin moments so that they are fitting 
parameters. Unlike there, it is better to have the final range of bz_min/bz_max 
here instead of running each case seperately. 
This uses the results from alpha from x_dep_alpha.py, so make sure
to run that first. Also, delta is the allowed deviations from the Ansatz, and 
0.05-0.2 is a typical range. 
"""

Nt = 128
Ns = 55

a = 1/2.359 # GeV^-1
gamma_E = 0.57721 # Euler constant

P0 = 1 # Reference momentum
Ng = 2 # Truncation order

bz_min_low = 2
bz_min_up  = 3

bz_max_low = 3
bz_max_up= 6

Pz_min_low = 2
Pz_min_up  = 6

Pz_max_low = 9
Pz_max_up= 10

save = False

# initialize empty arrays for saving data
for delta in [0.05, 0.10, 0.20]:
    print(f"{delta}")
    for Nh in [0]:
        for smear in ["final_results", "final_results_flow10"]:
            for init_char in ["T5"]:
                for N_max in [2]:
                    for kappa in np.arange(0.95,1.15,0.01):
                        index = 0
                        alpha  = np.zeros(((bz_min_up-bz_min_low)*(bz_max_up-bz_max_low)*(Pz_min_up-Pz_min_low)*(Pz_max_up-Pz_max_low)))
                        s1     = np.zeros(((bz_min_up-bz_min_low)*(bz_max_up-bz_max_low)*(Pz_min_up-Pz_min_low)*(Pz_max_up-Pz_max_low)))
                        s2     = np.zeros(((bz_min_up-bz_min_low)*(bz_max_up-bz_max_low)*(Pz_min_up-Pz_min_low)*(Pz_max_up-Pz_max_low)))
                        h0     = np.zeros(((bz_min_up-bz_min_low)*(bz_max_up-bz_max_low)*(Pz_min_up-Pz_min_low)*(Pz_max_up-Pz_max_low)))
                        chi2s  = np.zeros(((bz_min_up-bz_min_low)*(bz_max_up-bz_max_low)*(Pz_min_up-Pz_min_low)*(Pz_max_up-Pz_max_low)))

                        # create save path
                        save_path = f"{smear}/2_state_matrix_results_jack/{init_char}"
                        os.makedirs(save_path, exist_ok=True)
                        matrix_els_full = np.load(f"{save_path}/real_matrix_el_P0{P0}.npy")
                        matrix_el_js = np.load(f"{save_path}/real_matrix_elj_P0{P0}.npy")

                        matrix_errs_full = np.load(f"{save_path}/real_matrix_err_P0{P0}.npy")

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
                                        cov_inv = np.linalg.inv(cov)
                                        # function for fitting calculates ratio and wraps around Mellin-OPE of 
                                        # the matrix elements
                                        alphas = np.load(f"{smear}/2_state_matrix_results_jack/{init_char}/mellin_moms/mellin_alpha_Nmax2_h{Nh}_l{0}_P0{P0}_resum_k"+"%.2f"%kappa+".npy")
                                        alpha = alphas[index]
                                        if Nh == 0:
                                            def ratio(bz, Pz, s1, s2 ):
                                                    mu = kappa*2*np.exp(-gamma_E)/(bz*a)
                                                    # mu =2 
                                                    alpha_s = 0.303
                                                    num   = x_dep_many_m_ope(bz*a, alpha, s1, s2, 0, 0, 0, 2*N_max, Nh, phys_p(a,Pz), alpha_s, mu, init_char, Ng)
                                                    denom = x_dep_many_m_ope(bz*a, alpha, s1, s2, 0, 0, 0, 2*N_max, Nh, phys_p(a,P0), alpha_s, mu, init_char, Ng)
                                                    ratio_result = num/denom
                                                    return np.real(ratio_result)
                                            
                                            def chi_wrapper(params):
                                                return chi2(params, bzPz_range, delta, cov_inv, ratio, matrix_els_full)
                                            res = minimize(chi_wrapper, 
                                                    [0.004, 0.0004,], 
                                                    method="Nelder-Mead",
                                                    )
                                                
                                        elif Nh == 1:
                                            def ratio(bz, Pz, s1, s2, h0):
                                                mu = kappa*2*np.exp(-gamma_E)/(bz*a)
                                                # mu = 2
                                                alpha_s = 0.303
                                                num   = x_dep_many_m_ope(bz*a, alpha, s1, s2, 0, h0, 0, 2*N_max, Nh, phys_p(a,Pz), alpha_s, mu, init_char, Ng)
                                                denom = x_dep_many_m_ope(bz*a, alpha, s1, s2, 0, h0, 0, 2*N_max, Nh, phys_p(a,P0), alpha_s, mu, init_char, Ng)
                                                ratio_result = num/denom
                                                return np.real(ratio_result)
                                            
                                            def chi_wrapper(params):
                                                return chi2(params, bzPz_range, delta, cov_inv, ratio, matrix_els_full)
                                            res = minimize(chi_wrapper, 
                                                    [0.004, 0.001, -0.01,], 
                                                    # bounds=[(-np.inf, np.inf), (-np.inf,np.inf), (-1, 1)],
                                                    method="Nelder-Mead",
                                                    )
                                            
                                       
                                        popt = res.x
                                        chi2_result = res.fun / (bzPz_range.shape[1] - len(popt))
                                        # calculate chi^2


                                        # save data
                                        # alpha[index] = popt[0]
                                        s1[index] = popt[0]
                                        s2[index] = popt[1]
                                        # h0[index] = popt[2]
                                        # print(s1)
            
                                        chi2s[index] = chi2_result
                                        A = (2**(1+2*alpha)*gamma(1.5+alpha))/(np.sqrt(np.pi)*gamma(1+alpha))
                                        # these are the results for Ng=2, needs
                                        # modification is Ng != 2
                                        mm2 = (5+2*alpha+2*s1+6*alpha*s1+4*alpha**2*s1)/(15+16*alpha+4*alpha**2)
                                        mm4 = ((27+6*alpha)*(7+2*alpha+(4+4*alpha)*(1+2*alpha)*s1)+
                                            4*(1+alpha)*(2+alpha)+(1+2*alpha)*(3+2*alpha)*s2*gamma(1.5+alpha))/(16*gamma(5.5+alpha))
                                        print(mm2[index], mm4[index], chi2_result)
                                        
                                        # print(s1[index], s2[index], chi2_result)
                                        index += 1                                        

                                        if save:
                                            save_path = f"{smear}/2_state_matrix_results_jack/{init_char}/x_depend"
                                            os.makedirs(save_path, exist_ok=True)
                                            np.save(f"{save_path}/mellin_sns_chi2_Nmax{N_max}_h{Nh}_l{0}_Ng{Ng}_d{int(100*delta)}_P0{P0}_resum_k"+"%.2f"%kappa+".npy", chi2s)
                                            np.save(f"{save_path}/mellin_s1_Nmax{N_max}_h{Nh}_l{0}_Ng{Ng}_d{int(100*delta)}_P0{P0}_resum_k"+"%.2f"%kappa+".npy",s1)
                                            np.save(f"{save_path}/mellin_s2_Nmax{N_max}_h{Nh}_l{0}_Ng{Ng}_d{int(100*delta)}_P0{P0}_resum_k"+"%.2f"%kappa+".npy",s2)
                                            # np.save(f"{save_path}/mellin_h0_Nmax{N_max}_h{Nh}_l{0}_Ng{Ng}_d{int(100*delta)}_P0{P0}_resum_k"+"%.2f"%kappa+".npy",h0)
                                            # np.save(f"{save_path}/mellin_s3_Nmax{N_max}_h{Nh}_l{0}_Ng{Ng}_d{int(100*delta)}.npy",s3)

                                            





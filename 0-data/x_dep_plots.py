import numpy as np
import matplotlib.pyplot as plt
import scienceplots
import os
from scipy.special import gamma
from functions import phi

plt.style.use("science")

"""
Ethan Baker, ANL/Haverford College

Produces x-dependent plots and final estimate of s1, s2, and mm2, mm4 from
results in x_dep_sns.py. Loops through choices of fit parameters to get final 
estimate. 

The central value for jackknife calculations is the results at z_max=4 to 
prevent double counting. However, will need to modify this if a different 
z_max range is used. Places where changes are needed are indicated
"""

P0 = 1 # Refence momentum
Ns = 240 # number of samples for jackknife, product of indexing arrays in loops
save = True
Ng = 2
for delta in [5, 10, 20]:
    for smear in ["final_results", "final_results_flow10"]:
        for init_char in ["T5"]:
            alphas =  np.zeros((Ns,))
            s1s = np.zeros((Ns,))
            s2s = np.zeros((Ns,))
            h0s = np.zeros((Ns,))
            index = 0
            for Nh in [0]:
                for Nl in [0]:
                    for kappa in np.arange(0.95,1.15,0.01):
                        save_path = f"{smear}/2_state_matrix_results_jack/{init_char}/x_depend"
                        alpha  = np.load(f"{smear}/2_state_matrix_results_jack/{init_char}/mellin_moms/mellin_alpha_Nmax{2}_h{Nh}_l{Nl}_P0{P0}_resum_k"+"%.2f"%kappa+".npy")
                        s1     = np.load(f"{save_path}/mellin_s1_Nmax{2}_h{Nh}_l{Nl}_Ng{Ng}_d{delta}_P0{P0}_resum_k"+"%.2f"%kappa+".npy")
                        s2     = np.load(f"{save_path}/mellin_s2_Nmax{2}_h{Nh}_l{Nl}_Ng{Ng}_d{delta}_P0{P0}_resum_k"+"%.2f"%kappa+".npy")
                        if Nh == 1:
                            h0     = np.load(f"{save_path}/mellin_h0_Nmax{2}_h{Nh}_l{Nl}_Ng{Ng}_d{delta}_P0{P0}_resum_k"+"%.2f"%kappa+".npy")
                            h0s[index:index+4] = h0
                        alphas[index:index+12] = alpha
                        s1s[index:index+12] = s1
                        s2s[index:index+12] = s2
                        index += 12
            mm2s = (5+2*alphas+2*s1s+6*alphas*s1s+4*alphas**2*s1s)/(15+16*alphas+4*alphas**2)
            mm4s = ((27+6*alphas)*(7+2*alphas+(4+4*alphas)*(1+2*alphas)*s1s)+
                4*(1+alphas)*(2+alphas)+(1+2*alphas)*(3+2*alphas)*s2s*gamma(1.5+alphas))/(16*gamma(5.5+alphas))
            
            
            phis = np.zeros((len(np.arange(0,1.01,0.001)), len(s1s)))
            for i, u in enumerate(np.arange(0,1.01,0.001)):
                for j in range(len(alphas)):
                    phis[i,j] = phi(u, alphas[j], s1s[j], s2s[j])

            alpha_samples     = np.zeros((Ns,))
            s1_samples        = np.zeros((Ns,))
            s2_samples        = np.zeros((Ns,))
            h0_samples        = np.zeros((Ns,))
            
            alpha_err_samples = np.zeros((Ns,))
            s1_err_samples    = np.zeros((Ns,))
            s2_err_samples    = np.zeros((Ns,))
            h0_err_samples    = np.zeros((Ns,))

            mm2_samples        = np.zeros((Ns,))
            mm4_samples        = np.zeros((Ns,))
            mm2_err_samples    = np.zeros((Ns,))
            mm4_err_samples    = np.zeros((Ns,))

            phi_samples     = np.zeros((len(np.arange(0,1.01,0.001)),Ns))
            phi_err_samples = np.zeros((len(np.arange(0,1.01,0.001)),Ns,))

            # the indices here are meant to only take z_max=4 (central value)
            # data for calculation of mean. These indices will need to be changed
            # if z_max range changes. 
            mean2      = np.average(mm2s[80:160])
            mean4      = np.average(mm4s[80:160])
            mean_alpha = np.average(alphas[80:160])
            mean_s1     = np.average(s1s[80:160])
            mean_s2     = np.average(s2s[80:160])
            meanh0     = np.average(h0s[80:160])
            meanphi    = np.average(phis[:,80:160], axis=1)
            # perform resampling procedure
            for i in range(len(s1s)):
                alpha_samples[i] = np.average(np.delete(alphas, i))
                s1_samples[i]    = np.average(np.delete(s1s, i))
                s2_samples[i]    = np.average(np.delete(s2s, i))
                h0_samples[i]    = np.average(np.delete(h0s, i))

                mm2_samples[i]    = np.average(np.delete(mm2s, i))
                mm4_samples[i]    = np.average(np.delete(mm4s, i))
                for j, u in enumerate(np.arange(0,1.01,0.001)):
                    phi_samples[j,i]     = np.average(np.delete(phis[j,:], i))
                    phi_err_samples[j,i] = np.sqrt(np.average(np.abs(np.delete(phis[j,:], i) - meanphi[j])**2))
               
                s1_err_samples[i]   = np.sqrt(np.average(np.abs(np.delete(s1s, i) - mean_s1)**2))
                s2_err_samples[i]   = np.sqrt(np.average(np.abs(np.delete(s2s, i) - mean_s2)**2))
                h0_err_samples[i]   = np.sqrt(np.average(np.abs(np.delete(h0s, i) - meanh0)**2))

                mm2_err_samples[i]   = np.sqrt(np.average(np.abs(np.delete(mm2s, i) - mean2)**2))
                mm4_err_samples[i]   = np.sqrt(np.average(np.abs(np.delete(mm4s, i) - mean4)**2))
                alpha_err_samples[i] = np.sqrt(np.average(np.abs(np.delete(alphas, i) - mean_alpha)**2))

            # save results
            alpha = np.average(alpha_samples)
            alpha_err = np.sqrt(len(alphas)-1)*np.std(alpha_samples)

            s1 = np.average(s1_samples)
            s1_err = np.sqrt(len(s1s)-1)*np.std(s1_samples)
            s2 = np.average(s2_samples)
            s2_err = np.sqrt(len(s2s)-1)*np.std(s2_samples)


            mm2 = mean2
            mm2_err = np.sqrt(len(mm2s)-1)*np.std(mm2_samples)
            mm2_sys_err = np.average(mm2_err_samples)

            mm4 = mean4
            mm4_err = np.sqrt(len(mm4s)-1)*np.std(mm4_samples)
            mm4_sys_err = np.average(mm4_err_samples)

            h0 = meanh0
            h0_err = np.sqrt(len(h0s)-1)*np.std(h0_samples)
            h0_sys_err = np.average(h0_err_samples)

            print(mm2, mm2_err, mm2_sys_err, mm4, mm4_err, mm4_sys_err)
            if save:
                os.makedirs(f"{save_path}/{init_char}/final_mm", exist_ok=True)
                np.save(f"{smear}/2_state_matrix_results_jack/{init_char}/final_mm/ans_mm2_nocorr_d{delta}_P0{P0}.npy", 
                        np.array((mm2, mm2_err, mm2_sys_err)))
                np.save(f"{smear}/2_state_matrix_results_jack/{init_char}/final_mm/ans_mm4_nocorr_d{delta}_P0{P0}.npy", 
                        np.array((mm4, mm4_err, mm4_sys_err)))
                np.save(f"{smear}/2_state_matrix_results_jack/{init_char}/final_mm/ans_h0_nocorr_d{delta}_P0{P0}.npy", 
                        np.array((h0, h0_err, h0_sys_err)))

            init_conv = {"Z5": r"$\gamma_3\gamma_5$", "T5": r"$\gamma_0\gamma_5$"}
           
            plt.figure()
            plt.plot(np.arange(0,1.01,0.001),meanphi, label="Ansatz Fit")
            plt.fill_between(np.arange(0,1.01,0.001),
                            meanphi - np.sqrt(len(s1s)-1)*np.std(phi_samples, axis=1), 
                            meanphi + np.sqrt(len(s1s)-1)*np.std(phi_samples, axis=1),
                            color="#CD202C",
                            alpha=0.2)
            plt.fill_between(np.arange(0,1.01,0.001),
                            meanphi  - np.average(phi_err_samples, axis=1), 
                            meanphi  + np.average(phi_err_samples, axis=1),
                            color="#CD202C",
                            alpha=0.2)

            plt.xlim(0,1)
            # plt.title(f"$\\phi(u)$")
            plt.axhline(1,color="black", linestyle="dashed", label="Flat DA")
            plt.xlabel(r"$u$")
            plt.ylabel(r"$\phi(u)$")
            plt.text(0.26,0.8,f"$\\delta=${delta/100}")
            plt.axvspan(0,0.25, color="grey", alpha=0.35)
            plt.axvspan(0.75,1, color="grey", alpha=0.35)
            plt.legend()

            # plt.text(0.5, 0.05, r"$\delta =$ " + f"{delta/100}; " + init_conv[init_char])
            if save:
                save_path = f"{smear}/2_state_matrix_results_jack/{init_char}/x_depend"
                os.makedirs(save_path, exist_ok=True)
                plt.savefig(f"{save_path}/{init_char}x_dep_Ng2_d{delta}_P0{P0}.pdf")
            # plt.show()
            plt.close()

        # Author Ethan Baker, ANL/Haverford College



                













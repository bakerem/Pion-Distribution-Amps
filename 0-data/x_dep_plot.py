import numpy as np
import matplotlib.pyplot as plt
import scienceplots
import os
from scipy.special import gegenbauer, gamma
import scipy.integrate as integrate
from functions import phi

plt.style.use("science")

Ns = 9
save = False
delta = 10
Ng = 2
for delta in [5,10,20]:
    for smear in ["final_results", "final_results_eps10"]:
        for init_char in ["Z5", "T5"]:


            save_path = f"{smear}/2_state_matrix_results_jack/{init_char}/x_depend"
            alphas = np.load(f"{smear}/2_state_matrix_results_jack/{init_char}/mellin_moms/mellin_alpha_Nmax{2}_h{0}_l{0}.npy")
            s1s     = np.load(f"{save_path}/mellin_s1_Nmax{2}_h{0}_l{0}_Ng{Ng}_d{delta}.npy")
            s2s     = np.load(f"{save_path}/mellin_s2_Nmax{2}_h{0}_l{0}_Ng{Ng}_d{delta}.npy")
            # s3s     = np.load(f"{save_path}/mellin_s3_Nmax{2}_h{0}_l{0}_Ng{Ng}_d{delta}.npy")

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
            alpha_err_samples = np.zeros((Ns,))
            s1_err_samples    = np.zeros((Ns,))
            s2_err_samples    = np.zeros((Ns,))

            mm2_samples        = np.zeros((Ns,))
            mm4_samples        = np.zeros((Ns,))
            mm2_err_samples    = np.zeros((Ns,))
            mm4_err_samples    = np.zeros((Ns,))

            phi_samples     = np.zeros((len(np.arange(0,1.01,0.001)),Ns))
            phi_err_samples = np.zeros((len(np.arange(0,1.01,0.001)),Ns,))

            for i in range(len(s1s)):
                alpha_samples[i] = np.average(np.delete(alphas, i))
                s1_samples[i]    = np.average(np.delete(s1s, i))
                s2_samples[i]    = np.average(np.delete(s2s, i))
                mm2_samples[i]    = np.average(np.delete(mm2s, i))
                mm4_samples[i]    = np.average(np.delete(mm4s, i))
                for j, u in enumerate(np.arange(0,1.01,0.001)):
                    phi_samples[j,i]     = np.average(np.delete(phis[j,:], i))
                    phi_err_samples[j,i] = np.sqrt(np.average(np.delete(phis[j,:], i)**2) - (phi_samples[j,i])**2)
                s1_err_samples[i]    = np.sqrt(np.average(np.delete(s1s, i)**2) - (s1_samples[i])**2)
                s2_err_samples[i]    = np.sqrt(np.average(np.delete(s2s, i)**2) - (s2_samples[i])**2)
                mm2_err_samples[i]    = np.sqrt(np.average(np.delete(mm2s, i)**2) - (mm2_samples[i])**2)
                mm4_err_samples[i]    = np.sqrt(np.average(np.delete(mm4s, i)**2) - (mm4_samples[i])**2)
                alpha_err_samples[i] = np.sqrt(np.average(np.delete(alphas, i)**2) - (alpha_samples[i])**2)

            alpha = np.average(alpha_samples)
            alpha_err = np.sqrt(len(alphas)-1)*np.std(alpha_samples)

            s1 = np.average(s1_samples)
            s1_err = np.sqrt(len(s1s)-1)*np.std(s1_samples)
            s2 = np.average(s2_samples)
            s2_err = np.sqrt(len(s2s)-1)*np.std(s2_samples)
            # s3_err = np.sqrt(len(s3s)-1)*np.std(s3_samples)
            print(alpha, alpha_err, s1, s1_err, s2, s2_err,)


            mm2 = np.average(mm2_samples)
            mm2_err = np.sqrt(len(mm2s)-1)*np.std(mm2_samples)
            mm2_sys_err = np.average(mm2_err_samples)
            mm4 = np.average(mm4_samples)
            mm4_err = np.sqrt(len(mm4s)-1)*np.std(mm4_samples)
            mm4_sys_err = np.average(mm4_err_samples)
            print(mm2, mm2_err, mm2_sys_err, mm4, mm4_err, mm4_sys_err)
            if save:
                os.makedirs(f"{save_path}/{init_char}/final_mm", exist_ok=True)
                np.save(f"{smear}/2_state_matrix_results_jack/{init_char}/final_mm/ans_mm2_nocorr_d{delta}.npy", np.array((mm2, mm2_err, mm2_sys_err)))
                np.save(f"{smear}/2_state_matrix_results_jack/{init_char}/final_mm/ans_mm4_nocorr_d{delta}.npy", np.array((mm4, mm4_err, mm4_sys_err)))


            def mmn(u, n,phi, alpha,s1, s2):
                y = (1-2*u)**n*phi(u, alpha,s1, s2)
                return y 
            area = integrate.quad(phi,0,1,args=(alpha,s1, s2))
            mm2_check = integrate.quad(mmn,0,1,args=(2, phi, alpha,s1, s2))
            mm4_check = integrate.quad(mmn,0,1,args=(4, phi, alpha,s1, s2))
            print("Area is %.3f, mm2 is %.3f, mm4 is %.3f" %(area[0]-1, mm2_check[0], mm4_check[0]))
            

            init_conv = {"Z5": r"$\gamma_3\gamma_5$", "T5": r"$\gamma_0\gamma_5$"}

            plt.figure()
            plt.plot(np.arange(0,1.01,0.001),np.average(phi_samples, axis=1))
            plt.fill_between(np.arange(0,1.01,0.001),
                            np.average(phi_samples, axis=1) - np.sqrt(len(s1s)-1)*np.std(phi_samples, axis=1), 
                            phi(np.arange(0,1.01,0.001), alpha,s1, s2) + np.sqrt(len(s1s)-1)*np.std(phi_samples, axis=1),
                            color="blue",
                            alpha=0.3)
            plt.fill_between(np.arange(0,1.01,0.001),
                            np.average(phi_samples, axis=1) - np.sqrt(len(s1s)-1)*np.std(phi_samples, axis=1) - np.average(phi_err_samples), 
                            phi(np.arange(0,1.01,0.001), alpha,s1, s2) + np.sqrt(len(s1s)-1)*np.std(phi_samples, axis=1) + np.average(phi_err_samples),
                            color="blue",
                            alpha=0.2)
            plt.xlim(0,1)
            plt.title(r"$\phi(u)$"+ f": {init_conv[init_char]}")
            plt.xlabel(r"$u$")
            plt.ylabel(r"$\phi(u,\mu)$")
            if save:
                save_path = f"{smear}/2_state_matrix_results_jack/{init_char}/x_depend"
                os.makedirs(save_path, exist_ok=True)
                plt.savefig(f"{save_path}/{init_char}x_dep_Ng2_d{delta}.pdf")
            # plt.show()

        # Author Ethan Baker, ANL/Haverford College



                













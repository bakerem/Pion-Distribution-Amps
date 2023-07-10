import numpy as np
import matplotlib.pyplot as plt
import scienceplots
import os
from scipy.special import gegenbauer

plt.style.use("science")

Ns = 18
save = True

for smear in ["final_results", "final_results_eps10"]:
    for init_char in ["Z5", "T5"]:


        save_path = f"{smear}/2_state_matrix_results_jack/{init_char}/mellin_moms"
        alphas = np.load(f"{save_path}/mellin_alpha_Nmax{2}_h{0}_l{0}.npy")
        s1s     = np.load(f"{save_path}/mellin_s1_Nmax{2}_h{0}_l{0}.npy")
        s1s_err = np.load(f"{save_path}/mellin_s1_err_Nmax{2}_h{0}_l{0}.npy")
        s2s     = np.load(f"{save_path}/mellin_s2_Nmax{2}_h{0}_l{0}.npy")
        s2s_err = np.load(f"{save_path}/mellin_s2_err_Nmax{2}_h{0}_l{0}.npy")

        alpha_samples = np.zeros((Ns,))
        s1_samples    = np.zeros((Ns,))
        s2_samples    = np.zeros((Ns,))

        for i in range(len(s1s)):
            alpha_samples[i] = np.average(np.delete(alphas, i))
            s1_samples[i]    = np.average(np.delete(s1s, i))
            s2_samples[i]    = np.average(np.delete(s2s, i))
            
        
        alpha = np.average(alpha_samples)
        alpha_err = np.sqrt(len(alphas)-1)*np.average(s1_samples)

        s1 = np.average(s1_samples)
        s1_err = np.sqrt(len(s1s)-1)*np.average(s1_samples)
        s2 = np.average(s2_samples)
        s2_err = np.sqrt(len(s2s)-1)*np.average(s2_samples)

        def phi(u, alpha,):
            y = u**alpha*(1-u)**alpha
            return y 
        plt.figure()
        plt.plot(np.arange(0,1,0.001),phi(np.arange(0,1,0.001), alpha, ) )
        plt.title(r"$\phi(u)$"+ f": {init_char}")
        plt.xlabel(r"$u$")
        plt.ylabel(r"$\phi(u,\mu)$")
        if save:
            save_path = f"{smear}/2_state_matrix_results_jack/{init_char}/x_depend"
            os.makedirs(save_path, exist_ok=True)
            plt.savefig(f"{save_path}/x_dep_Ng2.pdf")
        plt.show()

    # Author Ethan Baker, ANL/Haverford College



            













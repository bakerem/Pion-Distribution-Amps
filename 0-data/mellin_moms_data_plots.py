import matplotlib.pyplot as plt
import numpy as np
import scienceplots
import os
from functions import phys_p, m_ope, alpha_func

"""
Ethan Baker, ANL/Haverford College

Produces plots that overlay fit results with raw data. Uses results of Mellin
moments from each momentum value with a specific z_max. 
"""

plt.style.use("science")


a = 1/2.359 # GeV^-1
gamma_E = 0.57721 #Euler Constant
P0 = 1  # Reference momentum
save = False


for init_char in ["T5"]:
    for smear in ["final_results_flow10"]:
        save_path = f"{smear}/2_state_matrix_results_jack/{init_char}"
        os.makedirs(save_path, exist_ok=True)
        formats = {"1":"s","2":"o","3":"H","4":"*","5":"D","6":"^","7":"v", "8":">", "9":"s"}
        plt.figure()
        ax = plt.gca()
        for Pz in [2,4,6,8,9]:
            kappa=1
            bz_max = 4
            real_matrix_el = np.load(f"{save_path}/real_matrix_el_P0{P0}.npy")
            real_matrix_err = np.load(f"{save_path}/real_matrix_err_P0{P0}.npy")
            mm2 = np.load(f"{save_path}/mellin_moms/mellin_moms2_Nmax2_h0_l0_P0{P0}_zmax{bz_max}_resum_k"+"%.2f"%kappa+".npy")[0]
            mm4 = np.load(f"{save_path}/mellin_moms/mellin_moms4_Nmax2_h0_l0_P0{P0}_zmax{bz_max}_resum_k"+"%.2f"%kappa+".npy")[0]
            # h0  = np.load(f"{save_path}/mellin_moms/mellin_h0_Nmax2_h0_l0_P0{P0}_zmax{bz_max}_resum_k"+"%.2f"%kappa+".npy")[0]
            # print(mm2, mm4, h0)
            def plot_ratio(bz,Pz, mm2, mm4,):
                mu = kappa*2*np.exp(-gamma_E)/(bz*a)
                # mu = 2
                alpha_s = 0.303
                num   = m_ope(bz*a, mm2, mm4, 0, 0, 0, 0, 0, 4, 1, phys_p(a,Pz), alpha_s, mu, init_char)
                denom = m_ope(bz*a, mm2, mm4, 0, 0, 0, 0, 0, 4, 1, phys_p(a,P0), alpha_s, mu, init_char)
                ratio_result = num/denom
                return np.real(ratio_result)
            color = next(ax._get_lines.prop_cycler)['color']
            bz_max += 1
            plt.plot(phys_p(a,Pz)*np.arange(2,bz_max, 0.001)*a,
                    plot_ratio(np.arange(2,bz_max, 0.001), Pz, mm2, mm4),color = color)
            plt.errorbar(phys_p(a,Pz)*np.arange(2,bz_max)*a,
                        real_matrix_el[Pz,2:bz_max],
                        yerr = real_matrix_err[Pz,2:bz_max],
                        fmt=formats[str(Pz)],
                        capsize=3,
                        markerfacecolor="none",
                        color = color,
                        label=f"$n_3$ = {Pz}")    

        init_conv = {"Z5":r"$\gamma_3\gamma_5$", "T5":r"$\gamma_0\gamma_5$"}
        plt.legend()
        plt.xlabel(r"$\lambda=P_3z_3$")
        plt.ylabel(r"Re $\mathcal{M}$")
        plt.xlim(0.35,4.5)
        plt.ylim(0.4,1.02)
        # plt.title(f"Renormalized Matrix Elements, $\\kappa=${kappa}")
        plt.text(2, 0.95, r"$P^0_3$ " + "= %.2f GeV" %phys_p(a,P0))
        plt.text(2.5, 0.9, f"$\\kappa=${kappa}")        

        # plt.text(4, 0.8, init_conv[init_char])

        if save:
            plt.savefig(f"{save_path}/{init_char}real_renorm_multi_p_full_data_zoom_P0{P0}_k{kappa}.pdf")
        plt.show()







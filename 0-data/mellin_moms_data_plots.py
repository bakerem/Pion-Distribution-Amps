import matplotlib.pyplot as plt
import numpy as np
import scienceplots
import os

from functions import phys_p, m_ope

plt.style.use("science")


a = 2.359
P0 = 3
save = True





for init_char in ["Z5", "T5"]:
    for smear in ["final_results", "final_results_eps10"]:
        save_path = f"{smear}/2_state_matrix_results_jack/{init_char}"
        os.makedirs(save_path, exist_ok=True)
        formats = {"1":"s","2":"o","3":"H","4":"*","5":"D","6":"^","7":"v", "8":">", "9":"s"}
        plt.figure()
        ax = plt.gca()
        for Pz in [4,5,6,7,8,9]:
            bz_max = 9
            real_matrix_el = np.load(f"{save_path}/real_matrix_el.npy")
            real_matrix_err = np.load(f"{save_path}/real_matrix_err.npy")
            mm2 = np.load(f"{save_path}/mellin_moms/mellin_moms2_Nmax2_h0_l0.npy")[-1]
            mm4 = np.load(f"{save_path}/mellin_moms/mellin_moms4_Nmax2_h0_l0.npy")[-1]
            def plot_ratio(bz,Pz, mm2, mm4,):
                num   = m_ope(bz/a, mm2, mm4, 0, 0, 0, 0, 0, 4, 0, phys_p(a,Pz), init_char)
                denom = m_ope(bz/a, mm2, mm4, 0, 0, 0, 0, 0, 4, 0, phys_p(a,P0), init_char)
                ratio_result = num/denom
                return np.real(ratio_result)
            color = next(ax._get_lines.prop_cycler)['color']

            plt.plot(phys_p(a,Pz)*np.arange(2,bz_max, 0.001)/a,
                    plot_ratio(np.arange(2,bz_max, 0.001), Pz, mm2, mm4,),color = color)
            plt.errorbar(phys_p(a,Pz)*np.arange(2,bz_max)/a,
                        real_matrix_el[Pz,2:bz_max],
                        yerr = real_matrix_err[Pz,2:bz_max],
                        fmt=formats[str(Pz)],
                        capsize=3,
                        markerfacecolor="none",
                        color = color,
                        label=f"Pz = {Pz}")    
            
        plt.legend()
        plt.xlabel(r"$\lambda$")
        plt.ylabel(r"Re $\mathcal{M}$")
        plt.xlim(0.35,7)
        plt.ylim(0,1.02)
        plt.title(f"Renormalized Matrix Elements: {init_char}")
        plt.text(4, 0.9, r"$P_0$ " + "= %.2f GeV" %phys_p(a,P0))
        if save:
            plt.savefig(f"{save_path}/real_renorm_multi_p_full.pdf")
        plt.show()







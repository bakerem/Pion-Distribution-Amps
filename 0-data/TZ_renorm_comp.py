import matplotlib.pyplot as plt
import numpy as np
import scienceplots
import os

from functions import phys_p, m_ope

plt.style.use("science")


a = 2.359
P0 = 3
save = True




for smear in ["final_results","final_results_eps10"]:
    plt.figure()
    for Pz in range(P0+1,10):
        for init_char in ["Z5", "T5"]:
            save_path = f"{smear}/2_state_matrix_results_jack/{init_char}"
            os.makedirs(save_path, exist_ok=True)
            formats = {"1":"s","2":"o","3":"H","4":"*","5":"D","6":"^","7":"v", "8":">", "9":"s"}
            ax = plt.gca()
            bz_max = 9
            real_matrix_el = np.load(f"{save_path}/real_matrix_el_P0{P0}.npy")
            real_matrix_err = np.load(f"{save_path}/real_matrix_err_P0{P0}.npy")
            
            color = next(ax._get_lines.prop_cycler)['color']

        
            plt.errorbar(np.arange(2,bz_max),
                        real_matrix_el[Pz,2:bz_max],
                        yerr = real_matrix_err[Pz,2:bz_max],
                        fmt="s",
                        capsize=3,
                        markerfacecolor="none",
                        color = color,
                        label=f"$n_3$ = {Pz}")    

        init_conv = {"Z5":r"$\gamma_3\gamma_5$", "T5":r"$\gamma_0\gamma_5$"}
        plt.legend()
        plt.xlabel(r"$z_3$")
        plt.ylabel(r"Re $\mathcal{M}$")
        # plt.xlim(0.35,7)
        # plt.ylim(0,1.02)
        plt.title(f"Renorm. Matrix Elements, $n^0_3=${P0}")
        # plt.text(1, 0.2, r"$P^0_3$ " + "= %.2f GeV" %phys_p(a,P0))
        # plt.text(4, 0.8, init_conv[init_char])

        if save:
            os.makedirs("final_results/2_state_matrix_results_jack/TZ_mellin_comp", exist_ok=True)
            plt.savefig(f"final_results/2_state_matrix_results_jack/TZ_mellin_comp/TZ_mellin_compPz{Pz}_P0{P0}.pdf")
        # plt.show()
        plt.close()







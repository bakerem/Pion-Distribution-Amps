# Author: Ethan Baker ANL/Haverford College

import numpy as np
import matplotlib.pyplot as plt
import os
import scienceplots
from matplotlib.ticker import MaxNLocator
from functions import phys_p


plt.style.use('science')

# init_char = "Z5"
Nt = 128
Ns = 10
a = 2.359
t_start = 4
t_end = 14
fits = {}
real = True
save = True


for smear_path in ["final_results", "final_results_eps10"]:
    if smear_path == "final_results":
        smear = "05"
    else:
        smear = "10"
    for Pz in [1,7]:
        fig = plt.figure()
        ax = fig.gca()
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        
        for init_char in ["Z5", "T5"]:
            # create directory for saving files
            parent = f"{smear_path}/2_state_matrix_results_jack"
            child = f"{init_char}"
            save_path = os.path.join(parent,child)
            os.makedirs(save_path, exist_ok=True)


            ##### Plot for Matrix elements over multiple z ######
            
            real_fit = np.load(f"{save_path}/real_raw_ratios.npy")
            real_fit_err = np.load(f"{save_path}/real_raw_ratios_err.npy")
            imag_fit = np.load(f"{save_path}/imag_raw_ratios.npy")
            imag_fit_err = np.load(f"{save_path}/imag_raw_ratios_err.npy")

            init_conv = {"Z5":r"$\gamma_3\gamma_5$", "T5":r"$\gamma_0\gamma_5$"}
            colors = {"Z5":r"red", "T5":r"blue"}
            if real:
                plt.errorbar(np.arange(0,13),
                                real_fit[Pz,:13],
                                yerr=real_fit_err[Pz,:13],
                                fmt="s",
                                label=init_conv[init_char],
                                markerfacecolor="none",
                                color=colors[init_char],
                                capsize=4)
            else:
                plt.errorbar(np.arange(0,13),
                                imag_fit[Pz,:13],
                                yerr=imag_fit_err[Pz,:13],
                                fmt="s",
                                label=init_conv[init_char],
                                markerfacecolor="none",
                                color="blue",
                                capsize=4)
        if real:
            plt.text(8,0.1, "$P_3=$ %.2f GeV" %phys_p(a, Pz))
            plt.legend()
            plt.xlabel("$z/a$")
            plt.ylabel(r"Re $h^B(z,P_3)$")
            # plt.title(f"Extrapolated Matrix Elements; {init_char}")
            if save:
                plt.savefig(f"{smear_path}/2_state_matrix_results_jack/{Pz}Z5_T5_diff.pdf")
        else:
            plt.legend()
            plt.xlabel("$z/a$")
            plt.ylabel(r"Imag $h^B(z,P_3)$")
            plt.title(f"Extrapolated Matrix Elements; {init_char}")
            if save:
                plt.savefig(f"{smear_path}/2_state_matrix_results_jack/{Pz}imag_extrapolated_R.pdf")
        # plt.show()

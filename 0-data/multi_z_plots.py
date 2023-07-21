# Author: Ethan Baker ANL/Haverford College

import numpy as np
import matplotlib.pyplot as plt
import os
import scienceplots
from matplotlib.ticker import MaxNLocator


plt.style.use('science')

# init_char = "Z5"
Nt = 128
Ns = 10
a = 2.359
fits = {}
real = True
save = True

for real in [True]:
    for smear_path in [ "final_results_eps10"]:
        if smear_path == "final_results":
            smear = "05"
        else:
            smear = "10"

        for init_char in ["T5"]:
            # create directory for saving files
            parent = f"{smear_path}/2_state_matrix_results_jack"
            child = f"{init_char}"
            save_path = os.path.join(parent,child)
            os.makedirs(save_path, exist_ok=True)


            ##### Plot for Matrix elements over multiple z ######
            formats = {"1":"s","2":"o","4":"H","6":"*","8":"D","5":"^"}
        
            for Pz in [1,2,3,4,5,6,7,8,9]:
                fig = plt.figure()
                ax = fig.gca()
                ax.xaxis.set_major_locator(MaxNLocator(integer=True))
                real_fit = np.load(f"{save_path}/real_raw_ratios.npy")
                real_fit_err = np.load(f"{save_path}/real_raw_ratios_err.npy")
                imag_fit = np.load(f"{save_path}/imag_raw_ratios.npy")
                imag_fit_err = np.load(f"{save_path}/imag_raw_ratios_err.npy")


                if real:
                    plt.errorbar(np.arange(0,13),
                                    real_fit[Pz,:13],
                                    yerr=real_fit_err[Pz,:13],
                                    fmt="bs",
                                    label=f"Pz = {Pz}",
                                    markerfacecolor="none",
                                    capsize=4)
                else:
                    plt.errorbar(np.arange(0,13),
                                    imag_fit[Pz,:13],
                                    yerr=imag_fit_err[Pz,:13],
                                    fmt=formats[f"{Pz}"],
                                    label=f"Pz = {Pz}",
                                    markerfacecolor="none",
                                    capsize=4)
                if real:
                    # plt.legend()
                    plt.xlabel("$z_3/a$")
                    plt.ylabel(r"Re $h^B(z_3,P_3)$ (GeV)")
                    bottom, top = plt.ylim()
                    if bottom> 0:
                        plt.text(0, bottom+ 0.3*bottom, r"$n_3=$" + f" {Pz}")
                    else:
                        plt.text(0, bottom- 0.3*bottom, r"$n_3=$" + f" {Pz}")


                    # plt.title(f"Extrapolated Matrix Elements; {init_char}")
                    if save:
                        plt.savefig(f"{save_path}/Pz{Pz}/real_extrapolated_R_Pz{Pz}.pdf")
                else:
                    plt.legend()
                    plt.xlabel("$z_3/a$")
                    plt.ylabel(r"Imag $h^B(z,P_3)$")
                    plt.title(f"Extrapolated Matrix Elements; {init_char}")
                    if save:
                        plt.savefig(f"{save_path}/Pz{Pz}/imag_extrapolated_R.pdf")
                # plt.show()

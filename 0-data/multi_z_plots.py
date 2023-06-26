# Author: Ethan Baker ANL/Haverford College

import numpy as np
import matplotlib.pyplot as plt
import os

init_char = "T5"
Nt = 128
Ns = 10
a = 2.359
t_start = 4
t_end = 14
fits = {}
real = False
save = True

# create directory for saving files
parent = "final_results/2_state_matrix_results_jack"
child = f"{init_char}"
save_path = os.path.join(parent,child)
os.makedirs(save_path, exist_ok=True)


##### Plot for Matrix elements over multiple z ######
formats = {"0":"bs","1":"ro","2":"gH","3":"m*","4":"cD","9":"y^"}
plt.figure()
for Pz in range(0,4):
    real_fit = np.load(f"{save_path}/real_raw_ratios.npy")
    real_fit_err = np.load(f"{save_path}/real_raw_ratios_err.npy")
    imag_fit = np.load(f"{save_path}/imag_raw_ratios.npy")
    imag_fit_err = np.load(f"{save_path}/imag_raw_ratios_err.npy")


    if real:
        plt.errorbar(np.arange(0,13),
                        real_fit[Pz,:13],
                        yerr=real_fit_err[Pz,:13],
                        fmt=formats[f"{Pz}"],
                        label=f"Pz = {Pz}",
                        capsize=4)
    else:
        plt.errorbar(np.arange(0,13),
                        imag_fit[Pz,:13],
                        yerr=imag_fit_err[Pz,:13],
                        fmt=formats[f"{Pz}"],
                        label=f"Pz = {Pz}",
                        capsize=4)
if real:
    plt.legend()
    plt.xlabel("$z/a$")
    plt.ylabel(r"Re $h^B(z,P_3)$")
    plt.title(f"Extrapolated Matrix Elements for Various Momenta; {init_char}")
    if save:
        plt.savefig(f"{save_path}/real_extrapolated_R.png")
else:
    plt.legend()
    plt.xlabel("$z/a$")
    plt.ylabel(r"Imag $h^B(z,P_3)$")
    plt.title(f"Extrapolated Matrix Elements for Various Momenta; {init_char}")
    if save:
        plt.savefig(f"{save_path}/imag_extrapolated_R.png")
plt.show()

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
real = True
save = False

# create directory for saving files
parent = "0-data/2_state_matrix_results"
child = f"{init_char}"
save_path = os.path.join(parent,child)
os.makedirs(save_path, exist_ok=True)


##### Plot for Matrix elements over multiple z ######
formats = {"4":"bs","5":"ro","6":"gH","7":"m*","8":"cD","9":"y^"}
plt.figure()
for Pz in [4,5,6,7,9]:
    real_fit = np.load(f"{save_path}/Pz{Pz}/real_data.npy")
    real_fit_err = np.load(f"{save_path}/Pz{Pz}/real_errs.npy")
    imag_fit = np.load(f"{save_path}/Pz{Pz}/imag_data.npy")
    imag_fit_err = np.load(f"{save_path}/Pz{Pz}/real_errs.npy")


    if real:
        plt.errorbar(np.arange(0,13),
                        real_fit[:13],
                        yerr=real_fit_err[:13],
                        fmt=formats[f"{Pz}"],
                        label=f"Pz = {Pz}",
                        capsize=4)
    else:
        plt.errorbar(np.arange(0,13),
                        imag_fit[:13],
                        yerr=imag_fit_err[:13],
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

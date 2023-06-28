# Author Ethan Baker, ANL/Haverford College
import numpy as np
import matplotlib.pyplot as plt
import os
 
Nt = 128
Ns = 10
a = 2.359
save = True


# Create save path
save_path = "final_results/2_state_matrix_results_jack"
os.makedirs(save_path, exist_ok=True)


##### Plot for Matrix elements over multiple z ######
plt.figure()
format_dict = {"Z5": "bo", "T5": "rs"}
for i in ["Z5", "T5"]:
    for Pz in range(0,4):
        # load in data
        real_fit = np.load(f"{save_path}/{i}/real_raw_ratios.npy")[Pz,:]
        real_fit_err = np.load(f"{save_path}/{i}/real_raw_ratios_err.npy")[Pz,:]
        imag_fit = np.load(f"{save_path}/{i}/imag_raw_ratios.npy")[Pz,:]
        imag_fit_err = np.load(f"{save_path}/{i}/imag_raw_ratios_err.npy")[Pz,:]
        if i == "Z5":
            line1 = plt.errorbar(Pz,
                    real_fit[0],
                    yerr=real_fit_err[0], 
                    fmt=format_dict[i], 
                    capsize=4)
        else: 
            line2 = plt.errorbar(Pz,
                    real_fit[0],
                    yerr=real_fit_err[0], 
                    fmt=format_dict[i], 
                    capsize=4)


plt.xlabel("$P_z$")
plt.ylabel(r"$f_\pi/Z_A$ ")
plt.title(r"Calculation of $f_{\pi}$")
plt.hlines(real_fit[1], 0,4,"gray", "--")
plt.legend([line1, line2], [r"$\gamma_3\gamma_5$", r"$\gamma_0\gamma_5$"])
if save:
    plt.savefig(f"{save_path}/real_f_pi.png")
plt.show()

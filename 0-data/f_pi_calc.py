import numpy as np
import matplotlib.pyplot as plt
import os
import scienceplots

plt.style.use('science')

"""
Ethan Baker, ANL/Haverford College

Produces plot that compares the bare matrix elements at z=0 for a range of 
momentum and operators (T5, Z5). This result should be independent of momentum
and the imaginary part should be 0. 
"""
 
Nt = 128
Ns = 10

save = True

for smear in ["final_results", "final_results_flow10"]:
    # Create save path
    save_path = f"{smear}/2_state_matrix_results_jack"
    os.makedirs(save_path, exist_ok=True)


    ##### Plot for Matrix elements over multiple z ######
    plt.figure()
    format_dict = {"Z5": "bo", "T5": "rs"}
    for i in ["Z5", "T5"]:
        for Pz in range(0,10):
            # load in data
            
            real_fit = np.load(f"{save_path}/{i}/real_raw_ratios.npy")[Pz,0]
            real_fit_err = np.load(f"{save_path}/{i}/real_raw_ratios_err.npy")[Pz,0]
            imag_fit = np.load(f"{save_path}/{i}/imag_raw_ratios.npy")[Pz,0]
            imag_fit_err = np.load(f"{save_path}/{i}/imag_raw_ratios_err.npy")[Pz,0]
            if Pz == 0:
                lineloc = real_fit
            if i == "Z5":
                line1 = plt.errorbar(Pz,
                        real_fit,
                        yerr=real_fit_err, 
                        fmt=format_dict[i], 
                        markerfacecolor="none",
                        capsize=4)
            else: 
                line2 = plt.errorbar(Pz,
                        real_fit,
                        yerr=real_fit_err, 
                        fmt=format_dict[i], 
                        markerfacecolor="none",
                        capsize=4)


    plt.xlabel("$P_z$")
    plt.ylabel(r"$f_\pi/Z_A$ ")
    # plt.title(r"Calculation of $f_{\pi}$")
    plt.axhline(lineloc,color="grey", linestyle="--")
    plt.legend([line1, line2], [r"$\gamma_3\gamma_5$", r"$\gamma_0\gamma_5$"])
    if save:
        plt.savefig(f"{save_path}/real_f_pi.pdf")
    plt.show()

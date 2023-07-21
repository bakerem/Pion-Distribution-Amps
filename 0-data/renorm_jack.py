# Author: Ethan Baker ANL/Haverford College

from operator import truediv
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
from functions import phys_p, read_data
import scienceplots

plt.style.use('science')

Nt = 128
Ns = 55
a = 2.359
bz_max = 30
P0 = 1
fits = {}
save = True
t_range = np.arange(0,128)


# Create save path
for real in [False,True]:
    for smear_path in ["final_results_eps10"]:
        if smear_path == "final_results":
            smear = "05"
        else:
            smear = "10"

        for init_char in ["T5", "Z5"]:
            save_path = f"{smear_path}/2_state_matrix_results_jack/{init_char}"
            os.makedirs(save_path, exist_ok=True)



            ##### Plot for Matrix elements over multiple z ######
            plt.figure()
            formats = {"5":"s","2":"o","3":"H","4":"*","9":"D","6":"^","7":"v", "8":">"}


            real_raw_ratios = np.load(f"{save_path}/real_raw_samples.npy")
            imag_raw_ratios = np.load(f"{save_path}/imag_raw_samples.npy")
            real_matrix_el_js = np.zeros((10,33,Ns,))
            imag_matrix_el_js = np.zeros((10,33,Ns,))

            for Pz in range(0,10):
                for s in range(Ns):
                    matrix_el_j = (((real_raw_ratios[Pz,:,s]+1j*imag_raw_ratios[Pz,:,s])/
                                    (real_raw_ratios[P0,:,s] + 1j*imag_raw_ratios[P0,:,s]))
                                  *((real_raw_ratios[P0,0,s]+1j*imag_raw_ratios[P0,0,s])/
                                    (real_raw_ratios[Pz,0,s] + 1j*imag_raw_ratios[Pz,0,s]))
                                  *(np.exp(-1j*np.arange(0,33)/a*(phys_p(a,Pz)-phys_p(a,P0))/2))
                                  )    
                    real_matrix_el_js[Pz,:,s] = np.real(matrix_el_j)
                    imag_matrix_el_js[Pz,:,s] = np.imag(matrix_el_j)

            real_matrix_el = np.average(real_matrix_el_js, axis=2)
            real_matrix_err = np.sqrt(Ns-1)*np.std(real_matrix_el_js, axis=2)
            imag_matrix_el = np.average(imag_matrix_el_js, axis=2)
            imag_matrix_err = np.sqrt(Ns-1)*np.std(imag_matrix_el_js, axis=2)


            if save:
                np.save(f"{save_path}/real_matrix_elj_P0{P0}.npy", real_matrix_el_js)
                np.save(f"{save_path}/real_matrix_el_P0{P0}.npy", real_matrix_el)
                np.save(f"{save_path}/real_matrix_err_P0{P0}.npy", real_matrix_err)
                np.save(f"{save_path}/imag_matrix_elj_P0{P0}.npy", imag_matrix_el_js)
                np.save(f"{save_path}/imag_matrix_el_P0{P0}.npy", imag_matrix_el)
                np.save(f"{save_path}/imag_matrix_err_P0{P0}.npy", imag_matrix_err)
                np.save(f"{save_path}/real_raw_ratios_P0{P0}.npy", np.average(real_raw_ratios, axis=2))
                np.save(f"{save_path}/real_raw_ratios_err_P0{P0}.npy", np.sqrt(Ns-1)*np.std(real_raw_ratios, axis=2))
                np.save(f"{save_path}/imag_raw_ratios_P0{P0}.npy", np.average(imag_raw_ratios, axis=2))
                np.save(f"{save_path}/imag_raw_ratios_err_P0{P0}.npy", np.sqrt(Ns-1)*np.std(imag_raw_ratios, axis=2))



            for Pz in np.arange(P0+1,10,2):
                bz_max = 9
                if real:
                    plt.errorbar(phys_p(a,Pz)*np.arange(0,bz_max)/a,
                                real_matrix_el[Pz,:bz_max],
                                yerr = real_matrix_err[Pz,:bz_max],
                                fmt=formats[str(Pz)],
                                capsize=3,
                                markerfacecolor="none",
                                label=f"$n_3$ = {Pz}")
                else:
                    plt.errorbar(phys_p(a,Pz)*np.arange(0,bz_max)/a,
                                imag_matrix_el[Pz,:bz_max],
                                yerr = abs(imag_matrix_err[Pz,:bz_max]),
                                fmt=formats[str(Pz)],
                                capsize=3,
                                markerfacecolor="none",
                                label=f"$n_3$ = {Pz}")
            init_conv = {"Z5":r"$\gamma_3\gamma_5$", "T5":r"$\gamma_0\gamma_5$"}

            if real:
                plt.legend()
                plt.xlabel(r"$P_3z_3$")
                plt.ylabel(r"Re $\mathcal{M}$")
                plt.xlim(0,7)
                plt.ylim(0,1.1)
                # plt.title(f"{init_conv[init_char]}")
                plt.text(0.2, 0.2, r"$P^0_3$ " + "= %.2f GeV" %phys_p(a,P0))
                if save:
                    plt.savefig(f"{save_path}/{init_char}real_renorm_multi_p_full_P0{P0}.pdf")
                plt.show()

            else:
                plt.legend()
                plt.xlabel(r"$\lambda=P_3z_3$")
                plt.ylabel(r"Imag $\mathcal{M}$")
                plt.text(0.05, 0.03, r"$P^0_3$ " + "= %.2f GeV" %phys_p(a,P0))

                # plt.title(f"{init_conv[init_char]}")
                plt.xlim(0,7)
                plt.ylim(-0.03, 0.04)
                plt.axhline(0,color="black", linestyle="--")
                if save:
                    plt.savefig(f"{save_path}/{init_char}imag_renorm_multi_p_full_P0{P0}.pdf")
                plt.show()





import numpy as np
import matplotlib.pyplot as plt
import os
from functions import read_data, phys_p
import scienceplots
plt.style.use("science")

"""
Ethan Baker ANL/Haverford College

Produces a plot of the fits for the first few z from the raw_el_fits.py. This
replicates that fitting; it does not use its results directly. 

"""
 
Nt = 128
Ns = 55
a = 1/2.359 #GeV^-1

t_start = 2
t_end = 7
dt = 1
save = False
Pz = 3

for smear_path in ["final_results", "final_results_flow10"]:
    if smear_path == "final_results":
        smear = "05"
    else:
        smear = "10"
    for init_char in ["T5"]:
    
        save_path = f"{smear_path}/2_state_matrix_results_jack/{init_char}"
        os.makedirs(save_path, exist_ok=True)
        if save == True:
            os.makedirs(save_path, exist_ok=True)



        bestE1_tmins = [2, 3, 3, 2, 2, 2, 3, 3, 3, 3] # E1 list from result of 
                                                      # 2-state fit. 

        def real_state2ratio(t,m0, m1):
            if init_char == "T5":
                num   = -(((Z0/(2*E0))*m0*(np.exp(-E0*t) - np.exp(-E0*(Nt-t))) 
                    + (Z1/(2*E1))*m1*(np.exp(-E1*t) - np.exp(-E1*(Nt-t))))
                )
            else: 
                num   = -(((Z0/(2*E0))*m0*(np.exp(-E0*t) + np.exp(-E0*(Nt-t))) 
                    + (Z1/(2*E1))*m1*(np.exp(-E1*t) + np.exp(-E1*(Nt-t))))
                )
            denom = ((Z0**2/(2*E0))*(np.exp(-E0*t) + np.exp(-E0*(Nt-t))) 
                + (Z1**2/(2*E1))*(np.exp(-E1*t) + np.exp(-E1*(Nt-t)))
            )     
            y = num/denom
            return np.real(y)

        def imag_state2ratio(t,m0, m1):
            if init_char == "T5":
                num   = -1j*(((Z0/(2*E0))*m0*(np.exp(-E0*t) - np.exp(-E0*(Nt-t))) 
                    + (Z1/(2*E1))*m1*(np.exp(-E1*t) - np.exp(-E1*(Nt-t))))
                )
            else: 
                num   = -1j*(((Z0/(2*E0))*m0*(np.exp(-E0*t) + np.exp(-E0*(Nt-t))) 
                    + (Z1/(2*E1))*m1*(np.exp(-E1*t) + np.exp(-E1*(Nt-t))))
                )
            denom = ((Z0**2/(2*E0))*(np.exp(-E0*t) + np.exp(-E0*(Nt-t))) 
                + (Z1**2/(2*E1))*(np.exp(-E1*t) + np.exp(-E1*(Nt-t)))
            )     
            y = num/denom
            return np.imag(y)

        ##### Plot for combined z over one Pz #####
        save = True
        plt.figure()
        for bz in [2,4,6,8]:
            t = np.arange(t_start,t_end)
            (real_ratio_means,
            imag_ratio_means, 
            real_ratio_stds, 
            imag_ratio_stds,
            real_samples,
            real_sample_errs,
            imag_samples,
            imag_sample_errs) = read_data(Ns, Nt, init_char, Pz, bz, smear)

            E1_data = np.load(f"final_results/two_state_fits/Pz{Pz}/E1_fits_Pz{Pz}.npy")
            E0 = np.sqrt((0.139)**2 + phys_p(a,Pz)**2)*a
            # -2 below accounts for indexing b/c 2-state fits start at t_min=2           
            E1 = E1_data[0,bestE1_tmins[Pz]-2]
            Z0 = np.sqrt(2*E0*E1_data[2,bestE1_tmins[Pz]-2])
            Z1 = np.sqrt(2*E1*E1_data[4,bestE1_tmins[Pz]-2])

            popt = np.load(f"{save_path}/real_popt.npy")
            pcov0 = np.load(f"{save_path}/real_pcov0.npy")
            pcov1 = np.load(f"{save_path}/real_pcov1.npy")
            imag_fit_err = np.load(f"{save_path}/imag_raw_ratios_err.npy")


            real_fit = real_state2ratio(np.arange(t_start,t_end,dt), popt[Pz, bz, 0], popt[Pz, bz, 1])
            real_fit_err1 = -real_fit + real_state2ratio(np.arange(t_start,t_end,dt),
                                            popt[Pz, bz, 0]+pcov0[Pz, bz], 
                                            popt[Pz, bz, 1])
            real_fit_err2 = -real_fit + real_state2ratio(np.arange(t_start,t_end,dt),
                                            popt[Pz, bz, 0], 
                                            popt[Pz, bz, 1]+(pcov1[Pz, bz]))
            real_fit_err = np.sqrt(real_fit_err1**2 + real_fit_err2**2)
            

            plt.errorbar(t,real_ratio_means[t_start:t_end],yerr=real_ratio_stds[t_start:t_end], fmt="ro", capsize=3, markerfacecolor='none')
            if init_char == "Z5":
                plt.text(2.05,real_fit[0]+ 0.05*real_fit[0], f"$z/a=${bz}")

            else:
                if real_fit[0] > 0:
                    plt.text(2.15,real_fit[0] + 0.15*real_fit[0], f"$z/a=${bz}")
                else:
                    plt.text(2.15,real_fit[0] - 0.1*real_fit[0], f"$z/a=${bz}")

# 
            plt.plot(np.arange(t_start,t_end,dt), real_fit, "b" )
            plt.fill_between(np.arange(t_start,t_end,dt),real_fit+real_fit_err, real_fit-real_fit_err, alpha=0.2, color="blue")
        bottom, top = plt.ylim()
        plt.xlabel(r"$t/a$")
        plt.ylabel(r"Re $R(t)$")
        plt.text(t_start, bottom- 0.05*bottom, r"$n_3=$" + f" {Pz}")
        if save == True:
            plt.savefig(f"{save_path}/Pz{Pz}/{init_char}multi_z_fits{Pz}_smear{smear}.pdf")
        plt.show()
        plt.close()

# Author: Ethan Baker ANL/Haverford College

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
from functions import phys_p, read_data

Nt = 128
Ns = 55
a = 2.359
bz_max = 30
init_char = "T5"
fits = {}
save = True
plot = True
t_range = np.arange(0,128)
lower_lim = 3
upper_lim = 18


# create arrays for storing jackknife blocks
real_raw_ratios = np.zeros((5,33,Ns))
imag_raw_ratios = np.zeros((5,33,Ns))

# iterate through momenta and bz values and calculate jackknife samples at each one
for Pz in range(0,4):
    save_path = f"final_results_eps10/2_state_matrix_results_jack/{init_char}/Pz{Pz}"
    os.makedirs(save_path, exist_ok=True)
    print(f"Starting Pz = {Pz}")
    for bz in range(0,33):
       
        (real_ratio_means,
        imag_ratio_means, 
        real_ratio_stds, 
        imag_ratio_stds,
        real_samples,
        real_sample_errs,
        imag_samples,
        imag_sample_errs) = read_data(Ns, Nt, init_char, Pz, bz)
        
        # load in data from previous fits 
        # E0_data is from dispersion relation and E1_data is from 2 state fit
        E1_data = np.load(f"final_results/two_state_fits/Pz{Pz}/samples.npy")
        

        """
        actual function for fitting
        spectral decomposition of DA correlator is "num" and spec. decomp.
        of 2pt correlator is denom
        For T5, the numerator in both should be a sinh function, for Z5, it
        should be  cosh function
        """
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
        
        # curve fitting and calculating chi squared
        real_popt_js = np.zeros((Ns,2))
        imag_popt_js = np.zeros((Ns,2))

        for s in range(Ns):
            E0 = np.sqrt((0.139)**2 + phys_p(a,Pz)**2)/a
            E1 = E1_data[s,2]
            Z0 = np.sqrt(2*E0*E1_data[s,0])
            Z1 = np.sqrt(2*E1*E1_data[s,1])
            real_popt_j, real_pcov_j = curve_fit(
                                            real_state2ratio, 
                                            t_range[lower_lim:upper_lim], 
                                            real_samples[s,lower_lim:upper_lim], 
                                            sigma=real_sample_errs[s,lower_lim:upper_lim],
                                            maxfev=2000,
                                            )

            imag_popt_j, imag_pcov_j = curve_fit(
                                            imag_state2ratio, 
                                            t_range[lower_lim:upper_lim], 
                                            imag_samples[s,lower_lim:upper_lim], 
                                            sigma=imag_sample_errs[s,lower_lim:upper_lim],
                                            maxfev=2000,
                                            )
    
            prefactor = {"Z5": a**2*Z0/phys_p(a,Pz), "T5": a**2*Z0/(a*E0) }

            # calculation of extrapolated value of ratio for the matrix element and
            # the errors propogated through errors in fitting parameters.      
            real_raw_ratios[Pz,bz,s] = prefactor[init_char]*real_state2ratio(100, *real_popt_j)
            imag_raw_ratios[Pz,bz,s] = prefactor[init_char]*imag_state2ratio(100, *imag_popt_j)
            real_popt_js[s,:] = real_popt_j
            imag_popt_js[s,:] = imag_popt_j

     # plotting routine

        if plot and bz%5==0:
            real_popt = np.average(real_popt_js,axis=0)
            imag_popt = np.average(imag_popt_js,axis=0)
            real_pcov0 = np.sqrt(Ns-1)*np.std(real_popt_js[:,0])
            real_pcov1 = np.sqrt(Ns-1)*np.std(real_popt_js[:,1])
            imag_pcov0 = np.sqrt(Ns-1)*np.std(imag_popt_js[:,0])
            imag_pcov1 = np.sqrt(Ns-1)*np.std(imag_popt_js[:,1])
            


            t_fit = np.arange(lower_lim,upper_lim,0.01)
            fig, axs = plt.subplots(2, 1, constrained_layout=True)
            fig.suptitle(f"R(t) at $N_z$=%.3f GeV and $z/a$={bz}" %phys_p(2.359,Pz))
            axs[0].errorbar(t_range[lower_lim:upper_lim], 
                        real_ratio_means[lower_lim:upper_lim], 
                        yerr=real_ratio_stds[lower_lim:upper_lim], 
                        capsize=4,
                        fmt="s",
                        label="Data",)
            axs[0].plot(t_fit, real_state2ratio(t_fit, *real_popt), label="Fit")
            tab_cols = ("Value", "Error")
            tab_rows = ("m0","m1")
            cells = [["%.2e" %real_popt[0], "%.2e" %np.sqrt(real_pcov0)]\
                    ,["%.2e" %real_popt[1],"%.2e" %np.sqrt(real_pcov1)]\
                    ]
            axs[0].table(cellText=cells, rowLabels=tab_rows, colLabels=tab_cols, loc="upper center", colWidths=[0.2,0.2])
            axs[0].set_xlabel(r"$t/a$")
            axs[0].set_ylabel(r"Re $R(t) $")

            axs[1].errorbar(t_range[lower_lim:upper_lim], 
                        imag_ratio_means[lower_lim:upper_lim], 
                        yerr=imag_ratio_stds[lower_lim:upper_lim], 
                        capsize=4,
                        fmt="s",
                        label="Data")
            axs[1].plot(t_fit, imag_state2ratio(t_fit, *imag_popt), label="Fit")
            tab_cols = ("Value", "Error")
            tab_rows = ("m0","m1")
            cells = [["%.2e" %imag_popt[0], "%.2e" %np.sqrt(imag_pcov0)],
                    ["%.2e" %imag_popt[1],"%.2e" %np.sqrt(imag_pcov1)]]
            axs[1].table(cellText=cells, rowLabels=tab_rows, colLabels=tab_cols, loc="lower center", colWidths=[0.2,0.2])
            axs[1].set_xlabel(r"$t/a$")
            axs[1].set_ylabel(r"Imag $(R(t))$")
            if save and bz%5==0:
                plt.savefig(f"{save_path}/Pz{Pz}_m0fit_bz{bz}.png")
            # plt.show()
            plt.close()

if save:
    save_path = f"final_results_eps10/2_state_matrix_results_jack/{init_char}"
    np.save(f"{save_path}/real_raw_ratios.npy", np.average(real_raw_ratios, axis=2))
    np.save(f"{save_path}/real_raw_ratios_err.npy", np.sqrt(Ns-1)*np.std(real_raw_ratios, axis=2))
    np.save(f"{save_path}/imag_raw_ratios.npy", np.average(imag_raw_ratios, axis=2))
    np.save(f"{save_path}/imag_raw_ratios_err.npy", np.sqrt(Ns-1)*np.std(imag_raw_ratios, axis=2))
    np.save(f"{save_path}/real_raw_samples.npy", real_raw_ratios)
    np.save(f"{save_path}/imag_raw_samples.npy", imag_raw_ratios)









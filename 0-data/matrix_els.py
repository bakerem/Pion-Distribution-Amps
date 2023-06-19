import h5py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
import os
from functions import read_data

init_char = "T5" # can be "5", "T5", or "Z5"
Nt = 128
Ns = 10
t_range = np.arange(0,128)

a = 2.359
save = True
            #Nz= 4 5 6  7 8 9
bestE0_tmins = [10,9,8,7,7,7] # with a window length of 15
bestE1_tmins = [ 4,3,4, 2,4,4]              # with a window length of 10
# set up directories for saving routines
def phys_p(a, n):
    # takes a as an energy in GeV
    return 2 * np.pi * n * a / 64

def perform_fit(lower_lim:int, upper_lim:int, bz:int, Pz:int, plot=False):
    """
    This function performs two separate fits, one for the real part of the
    fitting parameters and one for the imaginary part. Therefore, everything
    is split up into two separate routines, each labelled "real_..." and
    "imag_...". 
    """
         
    if save == True:
            parent = "0-data/2_state_matrix_results"
            child = f"{init_char}/Pz{Pz}"
            save_path = os.path.join(parent,child)
            os.makedirs(save_path, exist_ok=True)

    # read in data files
    real_ratio_means, imag_ratio_means, real_ratio_stds, imag_ratio_stds = read_data(Ns, Nt, init_char, Pz, bz)
    
    # load in data from previous fits 
    # E0_data is from 1 state fit and E1_data is from 2 state fit
    E1_data = np.load(f"stats/2state_fit_results/window_arrays/E1_fits_Pz{Pz}.npy")

    E0 = np.sqrt((0.139)**2 + phys_p(a,Pz)**2)/a
    E1 = E1_data[0,bestE1_tmins[Pz-4]-2]
    Z0 = np.sqrt(2*E0*E1_data[2,bestE1_tmins[Pz-4]-2])
    Z1 = np.sqrt(2*E1*E1_data[4,bestE1_tmins[Pz-4]-2])
    

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
    real_popt, real_pcov = curve_fit(
                            real_state2ratio, 
                            t_range[lower_lim:upper_lim], 
                            real_ratio_means[lower_lim:upper_lim], 
                            sigma=real_ratio_stds[lower_lim:upper_lim],
                            maxfev=2000,
                            )

    imag_popt, imag_pcov = curve_fit(
                            imag_state2ratio, 
                            t_range[lower_lim:upper_lim], 
                            imag_ratio_means[lower_lim:upper_lim], 
                            sigma=imag_ratio_stds[lower_lim:upper_lim],
                            maxfev=2000,
                            )

    real_chi2  = np.sum(
            (real_ratio_means[lower_lim:upper_lim]-
            real_state2ratio(t_range[lower_lim:upper_lim],*real_popt))**2/    
            (real_ratio_stds[lower_lim:upper_lim])**2) 
    imag_chi2 = np.sum(
            (imag_ratio_means[lower_lim:upper_lim]-
            imag_state2ratio(t_range[lower_lim:upper_lim],*imag_popt))**2/    
            (imag_ratio_stds[lower_lim:upper_lim])**2) 
    real_chi2  = real_chi2 / (upper_lim - lower_lim - len(real_popt))
    imag_chi2  = imag_chi2 / (upper_lim - lower_lim - len(real_popt))


    # plotting routine
    if plot == True:
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
        tab_rows = ("m0","m1", r"$\chi^2$")
        cells = [["%.2e" %real_popt[0], "%.2e" %np.sqrt(real_pcov[0,0])]\
                ,["%.2e" %real_popt[1],"%.2e" %np.sqrt(real_pcov[1,1])]\
                ,["%.3f" %real_chi2, "n/a"]]
        axs[0].table(cellText=cells, rowLabels=tab_rows, colLabels=tab_cols, loc="upper center", colWidths=[0.2,0.2])
        axs[0].set_xlabel(r"$t/a$")
        axs[0].set_ylabel(r"Re $R(t) $")
        axs[0].set_ylim(-30e-6,20e-6)

        axs[1].errorbar(t_range[lower_lim:upper_lim], 
                     imag_ratio_means[lower_lim:upper_lim], 
                     yerr=imag_ratio_stds[lower_lim:upper_lim], 
                     capsize=4,
                     fmt="s",
                     label="Data")
        axs[1].plot(t_fit, imag_state2ratio(t_fit, *imag_popt), label="Fit")
        tab_cols = ("Value", "Error")
        tab_rows = ("m0","m1", r"$\chi^2$")
        cells = [["%.2e" %imag_popt[0], "%.2e" %np.sqrt(imag_pcov[0,0])]\
                ,["%.2e" %imag_popt[1],"%.2e" %np.sqrt(imag_pcov[1,1])]\
                ,["%.3f" %imag_chi2, "n/a"]]
        axs[1].table(cellText=cells, rowLabels=tab_rows, colLabels=tab_cols, loc="lower center", colWidths=[0.2,0.2])
        axs[1].set_xlabel(r"$t/a$")
        axs[1].set_ylabel(r"Imag $(R(t))$")
        axs[1].set_ylim(-5e-6,35e-6)
        # plt.show()
        if save == True and bz%5==0:
            plt.savefig(f"{save_path}/Pz{Pz}_m0fit_bz{bz}.png")
        plt.close()
    return np.array([real_chi2, real_popt[0], real_pcov[0,0], real_popt[1], real_pcov[1,1],
                    imag_chi2, imag_popt[0], imag_pcov[0,0], imag_popt[1], imag_pcov[1,1],
                    ])


for Pz in range(4,10):
    fit_results = np.zeros((10,33))
    print(f"Starting step {Pz}")
    if save == True:
            parent = "0-data/2_state_matrix_results"
            child = f"{init_char}/Pz{Pz}"
            save_path = os.path.join(parent,child)
            os.makedirs(save_path, exist_ok=True)
    for bz in range(0,33):
        # results = perform_fit(bestE0_tmins[Pz-4],bestE0_tmins[Pz-4] + 10, bz, Pz, plot=True)
        results = perform_fit(4,14, bz, Pz, plot=True)
        if save == True:
            fit_results[:,bz] = results
    if save == True:
        np.save(f"{save_path}/Pz{Pz}_R.npy", fit_results)
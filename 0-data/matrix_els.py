import h5py
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy.optimize import curve_fit
import os

init_char = "Z5" # can be "5", "T5", or "Z5"
# Pz = 4        # ranges from 4 to 9
Nt = 128
Ns = 10
# bz = 1          # ranges from 1 t0 32
p0 = None
save = True
            #Nz= 4 5 6  7 8 9
bestE0_tmins = [10,9,8,10,8,7] # with a window length of 15
bestE1_tmins = [ 4,3,4, 2,4,4]              # with a window length of 10
# set up directories for saving routines
def phys_p(a, n):
    # takes a as an energy in GeV
    return 2 * np.pi * n * a / 64

def perform_fit(lower_lim, upper_lim, bz, Pz, plot=False):

    if save == True:
            parent = "0-data/matrix_results"
            child = f"{init_char}/Pz{Pz}"
            save_path = os.path.join(parent,child)
            os.makedirs(save_path, exist_ok=True)

    # read in DA data file
    file = h5py.File(f"0-data/qDA_cfgs/64I.qDA.ama.GSRC_W40_k6_flow05eps01.{init_char}.eta0.PX0PY0PZ{Pz}.h5")
    DA_data = file[f"b_X/bT0/bz-{bz}"]

    # read in 2pt correlation data
    columns = ["t"] + [str(i) for i in range(0,Ns)]
    c2pt = pd.read_csv(f"0-data/c2pt_cfgs/64IGSRC_W40_k6.ama.c2pt.PX0PY0PZ{Pz}.real.cfg.csv", names=columns)
    c2pt_data = np.array(c2pt.drop(axis="columns", labels=["t"])).transpose()

    # calculate means and error in ratio of DA/c2pt
    samples = np.zeros((Ns,Nt),dtype=np.cdouble)
    ratio = DA_data/c2pt_data

    
    for i in range(0,Ns):
        sample = np.mean(np.delete(-1j*ratio,i,axis=0), axis=0)
        samples[i,:] = sample

    ratio_means = np.average(samples, axis=0)
    real_ratio_stds = np.sqrt(Ns-1)*np.std(np.real(samples), axis = 0)
    imag_ratio_stds = np.sqrt(Ns-1)*np.std(np.imag(samples), axis = 0)

    # load in data from previous fits 
    # E0_data is from 1 state fit and E1_data is from 2 state fit
    E0_data = np.load(f"stats/fit_results/window_arrays/E0_fits_Pz{Pz}.npy")
    E1_data = np.load(f"stats/2state_fit_results/window_arrays/E1_fits_Pz{Pz}.npy")

    E0 = E0_data[0,bestE0_tmins[Pz-4]] 
    Z0 = np.sqrt(E0_data[2,bestE0_tmins[Pz-4]])
    E1 = E1_data[0,4]
    Z1 = np.sqrt(E1_data[4,4])

    # actual function for fitting
    def state2ratio(t,m0, m1):
        num   = -1j*((Z0/(2*E0))*m0*(np.exp(-E0*t) + np.exp(-E0*(Nt-t))) 
               + (Z1/(2*E1))*m1*(np.exp(-E1*t) + np.exp(-E1*(Nt-t))))
        
        denom = (Z0**2/(2*E0))*(np.exp(-E0*t) + np.exp(-E0*(Nt-t))) \
              + (Z1**2/(2*E1))*(np.exp(-E1*t) + np.exp(-E1*(Nt-t)))
              
        y = num/denom
        return y
    
    # split ratios into real and imaginary parts and concatenate
    ratio_real = np.real(ratio_means[lower_lim:upper_lim])
    ratio_imag = np.imag(ratio_means[lower_lim:upper_lim])
    ratio_both = np.hstack([ratio_real, ratio_imag])
    both_ratio_stds = np.hstack([real_ratio_stds[lower_lim:upper_lim],
                                  imag_ratio_stds[lower_lim:upper_lim]])

    # function used in curve-fitting  that fits real and complex data together
    def complex_combined(t, m0, m1):
        N = len(t)
        t_real = t[:N//2]
        t_imag = t[N//2:]
        y_real = np.real(state2ratio(t_real,m0, m1))
        y_imag = np.imag(state2ratio(t_imag,m0, m1))
        return np.hstack([y_real,y_imag])

    print()
    # curve fitting and calculating chi squared
    popt, pcov = curve_fit(
                    complex_combined, 
                    np.hstack([c2pt["t"].iloc[lower_lim:upper_lim], c2pt["t"].iloc[lower_lim:upper_lim]]), 
                    ratio_both, 
                    sigma=both_ratio_stds,
                    p0=p0,
                    maxfev=2000,
                    )

    chi2  = np.sum(
            (ratio_real-
            np.real(state2ratio(c2pt["t"].iloc[lower_lim:upper_lim],*popt)))**2/    
            (real_ratio_stds[lower_lim:upper_lim])**2) 
    chi2 += np.sum(
            (ratio_imag-
            np.imag(state2ratio(c2pt["t"].iloc[lower_lim:upper_lim],*popt)))**2/    
            (imag_ratio_stds[lower_lim:upper_lim])**2) 
    chi2  = chi2 / (2*(upper_lim - lower_lim - len(popt)))
    print(state2ratio(np.arange(0,10,1),*popt))
    if plot == True:
        t_fit = np.arange(lower_lim,upper_lim,0.01)
        fig, axs = plt.subplots(2, 1, constrained_layout=True)
        fig.suptitle(f"R(t) at $N_z$=%.3f GeV and $z/a$={bz}" %phys_p(2.359,Pz))
        axs[0].errorbar(c2pt["t"].iloc[lower_lim:upper_lim], 
                     np.real(ratio_means[lower_lim:upper_lim]), 
                     yerr=real_ratio_stds[lower_lim:upper_lim], 
                     capsize=4,
                     fmt="s",
                     label="Data",)
        axs[0].plot(t_fit, np.real(state2ratio(t_fit, *popt)), label="Fit")
        tab_cols = ("Value", "Error")
        tab_rows = ("m0", r"$\chi^2$")
        cells = [["%.2e" %popt[0], "%.2e" %np.sqrt(pcov[0,0])]\
                # ,["%.2e" %popt[1],"%.2e" %np.sqrt(pcov[1,1])]\
                ,["%.3f" %chi2, "n/a"]]
        axs[0].table(cellText=cells, rowLabels=tab_rows, colLabels=tab_cols, loc="upper center", colWidths=[0.2,0.2])
        axs[0].set_xlabel(r"$t/a$")
        axs[0].set_ylabel(r"Re $R(t) $")
        axs[0].set_ylim(-0.00005,0.00005)

        axs[1].errorbar(c2pt["t"].iloc[lower_lim:upper_lim], 
                     np.imag(ratio_means[lower_lim:upper_lim]), 
                     yerr=imag_ratio_stds[lower_lim:upper_lim], 
                     capsize=4,
                     fmt="s",
                     label="Data")
        axs[1].plot(t_fit, np.imag(state2ratio(t_fit, *popt)), label="Fit")
        # tab_cols = ("Value", "Error")
        # tab_rows = ("m0", r"$\chi^2$")
        # cells = [["%.2e" %popt[0], "%.2e" %np.sqrt(pcov[0,0])]\
        #         # ,["%.2e" %popt[1],"%.2e" %np.sqrt(pcov[1,1])]\
        #         ,["%.3f" %chi2, "n/a"]]
        # axs[1].table(cellText=cells, rowLabels=tab_rows, colLabels=tab_cols, loc="upper center", colWidths=[0.2,0.2])
        # axs[1].ylim()
        axs[1].set_xlabel(r"$t/a$")
        axs[1].set_ylabel(r"Imag $(R(t))$")
        axs[1].set_ylim(-0.00005,0.00005)
        plt.show()
        if save == True and bz%5==0:
            plt.savefig(f"{save_path}/Pz{Pz}_m0fit_bz{bz}.png")
        plt.close()
    return chi2,  popt, pcov[0,0]

save=False

for Pz in range(4,10):
    m0_fits = np.zeros((3,32))
    print(f"Starting step {Pz}")
    if save == True:
            parent = "0-data/2_state_matrix_results"
            child = f"{init_char}/Pz{Pz}"
            save_path = os.path.join(parent,child)
            os.makedirs(save_path, exist_ok=True)
    for bz in range(1,33):
        # results = perform_fit(bestE0_tmins[Pz-4],bestE0_tmins[Pz-4] + 15, bz, Pz, plot=True)
        results = perform_fit(4,4 + 15, bz, Pz, plot=True)
        if save == True:
            m0_fits[0,bz-1] = results[1]
            m0_fits[1,bz-1] = results[2]
            m0_fits[2,bz-1] = results[0]
    if save == True:
        np.save(f"{save_path}/Pz{Pz}_m0.npy", m0_fits)
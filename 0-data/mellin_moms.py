# Author Ethan Baker, ANL/Haverford College

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit
from functions import m_ope, phys_p

Nt = 128
a = 2.359
mu = 2/a
P0 = 4
Pz = 6
N_max = 4
bz = 5
bz_min = 2
bz_max = 10
Pz = 6
Pz_min = 5
Pz_max = 9
Pz_range = phys_p(a,np.arange(Pz_min,Pz_max))
save = False
plot = True

# initialize empty arrays for saving data
moms2 = np.zeros((bz_max-bz_min,))
moms2_err = np.zeros((bz_max-bz_min,))
moms4 = np.zeros((bz_max-bz_min,))
moms4_err = np.zeros((bz_max-bz_min,))

# create save path
save_path = "0-data/renorm_results"
os.makedirs(save_path, exist_ok=True)


# iterate over range of z values
for bz in range(bz_min, bz_max):
    matrix_els = np.zeros((5,))
    matrix_errs = np.zeros((5,))

    # load in data for each momentum value
    for Pz in range(5,10):
        matrix_els[Pz-5]  = np.real(np.load(f"{save_path}/Pz{Pz}_matrix_els.npy")[bz])
        matrix_errs[Pz-5] = np.real(np.load(f"{save_path}/renorm_results/Pz{Pz}_matrix_errs.npy")[bz])

    # function for fitting calculates ratio and wraps around Mellin-OPE of 
    # the matrix elements
    def ratio(Pz, mm2, mm4):
        num   = m_ope(bz, mm2, mm4, N_max, mu, Pz)
        denom = m_ope(bz, mm2, mm4, N_max, mu, phys_p(a,P0))
        ratio_result = num/denom
        return np.real(ratio_result)

    # fit curve
    popt, pcov = curve_fit(ratio,
                        Pz_range,
                        np.real(matrix_els[Pz_min-5:Pz_max-5]),
                        p0 = (0.3,0.1),
                        sigma=np.real(matrix_errs[Pz_min-5:Pz_max-5]))

    # calculate chi^2
    chi2 = np.sum(
        (matrix_els[Pz_min-5:Pz_max-5]
            - ratio(Pz_range, *popt)
        ) ** 2
        / (matrix_errs[Pz_min-5: Pz_max-5]) ** 2 
        / (Pz_max - Pz_min - len(popt)))
    
    # save data
    moms2[bz-bz_min] = popt[0]
    moms2_err[bz-bz_min] = np.sqrt(pcov[0,0])
    moms4[bz-bz_min] = popt[1]
    moms4_err[bz-bz_min] = np.sqrt(pcov[1,1])


    # plotting routine
    if plot:
        plt.figure()
        plt.errorbar(Pz_range, 
                    np.real(matrix_els[Pz_min-5:Pz_max-5]),
                    yerr=np.real(matrix_errs[Pz_min-5:Pz_max-5]),
                    fmt = "bs",
                    capsize=3,)

        plt.plot(phys_p(a,np.arange(Pz_min, Pz_max, 0.001)), 
                ratio(phys_p(a, np.arange(Pz_min, Pz_max, 0.001)), *popt), 
                "r",)
        tab_cols = ("Value", "Error")
        tab_rows = ("mm2","mm4", r"$\chi^2$")
        cells = [["%.2e" %popt[0], "%.2e" %np.sqrt(pcov[0,0])],
                ["%.2e" %popt[1], "%.2e" %np.sqrt(pcov[1,1])],
                ["%.3f" %chi2, "n/a"],
                ]
        plt.table(cellText=cells, rowLabels=tab_rows, colLabels=tab_cols, loc="upper center", colWidths=[0.2,0.2])
        plt.title(f"Fit to RI-ratio at $z_3$={bz}")
        plt.xlabel("P_3 (GeV)")
        plt.ylabel(r"Re $\mathcal{M}$")
        plt.show()

# save arrays
if save:
    np.save("0-data/renorm_results/moms2.npy",moms2)
    np.save("0-data/renorm_results/moms2_err.npy", moms2_err)
    np.save("0-data/renorm_results/moms4.npy",moms4)
    np.save("0-data/renorm_results/moms4_err.npy", moms4_err)





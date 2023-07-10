# Author Ethan Baker, ANL/Haverford College

import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit
from functions import c_ope, phys_p

Nt = 128
a = 2.359 # GeV^-1
alpha_s = 0.303
mu = 2    # GeV
P0 = 1
N_max = 2
Nh = 0
Nl = 0


bz_min_low = 2
bz_min_up  = 3

bz_max_low = 6
bz_max_up= 9

Pz_min_low = 2
Pz_min_up  = 4

Pz_max_low = 7
Pz_max_up= 10
save = True
plot = False

# initialize empty arrays for saving data
for smear in ["final_results", "final_results_eps10"]:
    for init_char in ["Z5","T5"]:
        moms2     = np.zeros(((bz_min_up-bz_min_low)*(bz_max_up-bz_max_low)*(Pz_min_up-Pz_min_low)*(Pz_max_up-Pz_max_low)))
        moms2_err = np.zeros(((bz_min_up-bz_min_low)*(bz_max_up-bz_max_low)*(Pz_min_up-Pz_min_low)*(Pz_max_up-Pz_max_low)))
        moms4     = np.zeros(((bz_min_up-bz_min_low)*(bz_max_up-bz_max_low)*(Pz_min_up-Pz_min_low)*(Pz_max_up-Pz_max_low)))
        moms4_err = np.zeros(((bz_min_up-bz_min_low)*(bz_max_up-bz_max_low)*(Pz_min_up-Pz_min_low)*(Pz_max_up-Pz_max_low)))
        moms6     = np.zeros(((bz_min_up-bz_min_low)*(bz_max_up-bz_max_low)*(Pz_min_up-Pz_min_low)*(Pz_max_up-Pz_max_low)))
        moms6_err = np.zeros(((bz_min_up-bz_min_low)*(bz_max_up-bz_max_low)*(Pz_min_up-Pz_min_low)*(Pz_max_up-Pz_max_low)))
        moms8     = np.zeros(((bz_min_up-bz_min_low)*(bz_max_up-bz_max_low)*(Pz_min_up-Pz_min_low)*(Pz_max_up-Pz_max_low)))
        moms8_err = np.zeros(((bz_min_up-bz_min_low)*(bz_max_up-bz_max_low)*(Pz_min_up-Pz_min_low)*(Pz_max_up-Pz_max_low)))

        # create save path
        save_path = f"{smear}/2_state_matrix_results_jack/{init_char}"
        os.makedirs(save_path, exist_ok=True)
        matrix_els_full = np.load(f"{save_path}/real_matrix_el.npy")
        matrix_errs_full = np.load(f"{save_path}/real_matrix_err.npy")

        #load in data
        index = 0
        for bz_min in range(bz_min_low, bz_min_up):
            for bz_max in range(bz_max_low, bz_max_up):
                for Pz_min in range(Pz_min_low, Pz_min_up):
                    for Pz_max in range(Pz_max_low, Pz_max_up):
                        bz_range = np.arange(bz_min, bz_max)  
                        Pz_range = np.arange(Pz_min, Pz_max)
                        bzPz_range = np.array((np.repeat(bz_range,len(Pz_range)),np.tile(Pz_range,len(bz_range))))
                        matrix_els = matrix_els_full[Pz_min:Pz_max,bz_min:bz_max].flatten("F")
                        matrix_errs = matrix_errs_full[Pz_min:Pz_max,bz_min:bz_max].flatten("F")

                        # function for fitting calculates ratio and wraps around gegen-OPE of 
                        # the matrix elements
                        def ratio(bzPz, an2, an4, ):
                            bz, Pz = bzPz_range
                            num   = c_ope(bz/a, an2, an4, 0, 0, 0, 0, 0, 2*N_max, Nh, phys_p(a,Pz), alpha_s, init_char)
                            denom = c_ope(bz/a, an2, an4, 0, 0, 0, 0, 0, 2*N_max, Nh, phys_p(a,P0), alpha_s, init_char)
                            ratio_result = num/denom
                            return np.real(ratio_result)
                        
                        # fit curve
                        popt, pcov = curve_fit(ratio,
                                            bzPz_range,
                                            matrix_els,
                                            sigma=matrix_errs)
                        # calculate chi^2
                        chi2 = (np.sum(
                            (matrix_els
                                - ratio(bzPz_range, *popt)
                            ) ** 2
                            / (matrix_errs) ** 2 )
                            / (bzPz_range.shape[1] - len(popt)))

                        # save data
                        moms2[index] = popt[0]
                        moms2_err[index] = np.sqrt(pcov[0,0])
                        moms4[index] = popt[1]
                        moms4_err[index] = np.sqrt(pcov[1,1])
                        # moms6[index] = popt[2]
                        # moms6_err[index] = np.sqrt(pcov[2,2])
                        # moms8[index] = popt[3]
                        # moms8_err[index] = np.sqrt(pcov[3,3])
                        index += 1
                        print(moms2)
                        # plotting routine
                        if plot:
                            def plot_ratio(bz,mm2, mm4, mm6, mm8 ):
                                num   = c_ope(bz/a, mm2, mm4, mm6, 0, 0,0, 0, 2*N_max, Nh, phys_p(a,Pz_min), init_char)
                                denom = c_ope(bz/a, mm2, mm4, mm6, 0, 0,0, 0, 2*N_max, Nh, phys_p(a,P0), init_char)
                                ratio_result = num/denom
                                return np.real(ratio_result)
                        
                            plt.figure()
                            plt.errorbar(bz_range/a, 
                                        matrix_els_full[Pz_min,bz_min:bz_max],
                                        yerr=np.real(matrix_errs_full[Pz_min, bz_min:bz_max]),
                                        fmt = "bs",
                                        capsize=3,)

                            plt.plot(np.arange(bz_min, bz_max, 0.001)/a, 
                                    plot_ratio(np.arange(bz_min, bz_max, 0.001), *popt), 
                                    "r",)
                            tab_cols = ("Value", "Error")
                            tab_rows = (r"$\langle x^2\rangle$",
                                        r"$\langle x^4\rangle$", 
                                        r"$\langle x^6\rangle$", 
                                        r"$\langle x^8\rangle$", 
                                        r"$\chi^2$")
                            cells = [["%.2e" %popt[0], "%.2e" %np.sqrt(pcov[0,0])],
                                        ["%.2e" %popt[1], "%.2e" %np.sqrt(pcov[1,1])],
                                        ["%.2e" %popt[2], "%.2e" %np.sqrt(pcov[2,2])],
                                        ["%.2e" %popt[3], "%.2e" %np.sqrt(pcov[3,3])],
                                    ["%.3f" %chi2, "n/a"],
                                    ]
                            plt.table(cellText=cells, rowLabels=tab_rows, colLabels=tab_cols, loc="upper right", colWidths=[0.2,0.2])
                            Pz = 3
                            plt.title(f"Fit to RI-ratio at $P_z$={Pz}")
                            plt.xlabel(r"$z_3$ (GeV$^{-1}$)")
                            plt.ylabel(r"Re $\mathcal{M}$")
                            plt.show()

                        if save:
                            save_path = f"{smear}/2_state_matrix_results_jack/{init_char}/gegen_moms"
                            os.makedirs(save_path, exist_ok=True)
                            plt.savefig(f"{save_path}/_Nmax{N_max}.png")
                            np.save(f"{save_path}/gegen_moms2_Nmax{N_max}_h{Nh}_l{Nl}.npy",moms2)
                            np.save(f"{save_path}/gegen_moms2_err_Nmax{N_max}_h{Nh}_l{Nl}.npy", moms2_err)
                            np.save(f"{save_path}/gegen_moms4_Nmax{N_max}_h{Nh}_l{Nl}.npy",moms4)
                            np.save(f"{save_path}/gegen_moms4_err_Nmax{N_max}_h{Nh}_l{Nl}.npy", moms4_err)
                            # np.save(f"{save_path}/gegen_moms6_Nmax{N_max}_h{Nh}_l{Nl}.npy",moms6)
                            # np.save(f"{save_path}/gegen_moms6_err_Nmax{N_max}_h{Nh}_l{Nl}.npy", moms6_err)
                            # np.save(f"{save_path}/gegen_moms8_Nmax{N_max}_h{Nh}_l{Nl}.npy",moms8)
                            # np.save(f"{save_path}/gegen_moms8_err_Nmax{N_max}_h{Nh}_l{Nl}.npy", moms8_err)





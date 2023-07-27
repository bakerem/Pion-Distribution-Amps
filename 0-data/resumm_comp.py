# Author Ethan Baker, ANL/Haverford College

import numpy as np
import matplotlib.pyplot as plt
import os
import scienceplots
import matplotlib as mpl
import matplotlib.font_manager as font_manager
from functions import phys_p
from scipy.special import gamma

"""
Ethan Baker, ANL/Haverford College

Produces plots of the Mellin moment results from "mellin_moms.py." 
Loops through ranges of fit parameters to plot each results seperately. Does not
perform jackknife resampling. 
"""

Ns = 4
P0 = 1


plt.style.use('science')
plt.rcParams['figure.figsize'] = [6.5,9]

save = True
for smear in ["final_results", "final_results_flow10"]:
    for init_char in ["T5"]:
        save_path = f"{smear}/2_state_matrix_results_jack"
        infos  = []
        data2 = np.zeros((Ns,2))
        data2resum = np.zeros((Ns,2))
        data4 = np.zeros((Ns,2))
        data4resum = np.zeros((Ns,2))
        index = 0
        for Nh in range(0,1):
            for Nl in range(0,1):
                for N_max in range(2,3):
                    for bz_max in range(7,8):
                        for Pz_min in range(0,4):
                            for Pz_max in range(0,1):
                                info = f" {Nh} {Pz_min+2}"
                                infos.append(info)
                                
                    moms2 = np.load(f"{save_path}/{init_char}/mellin_moms/mellin_moms2_Nmax{N_max}_h{Nh}_l{Nl}_P0{P0}_zmax{bz_max}.npy")
                    moms2_resum = np.load(f"{save_path}/{init_char}/mellin_moms/mellin_moms2_Nmax{N_max}_h{Nh}_l{Nl}_P0{P0}_zmax{bz_max}_resum_k1.00.npy")
                    moms2_err = np.load(f"{save_path}/{init_char}/mellin_moms/mellin_moms2_err_Nmax{N_max}_h{Nh}_l{Nl}_P0{P0}_zmax{bz_max}.npy")
                    moms2_resum_err = np.load(f"{save_path}/{init_char}/mellin_moms/mellin_moms2_err_Nmax{N_max}_h{Nh}_l{Nl}_P0{P0}_zmax{bz_max}_resum_k1.00.npy")
                    data2[index:index+4] = np.array([moms2, moms2_err]).transpose() 
                    data2resum[index:index+4] = np.array([moms2_resum, moms2_resum_err]).transpose() 

                    moms4 = np.load(f"{save_path}/{init_char}/mellin_moms/mellin_moms4_Nmax{N_max}_h{Nh}_l{Nl}_P0{P0}_zmax{bz_max}.npy")
                    moms4_resum = np.load(f"{save_path}/{init_char}/mellin_moms/mellin_moms4_Nmax{N_max}_h{Nh}_l{Nl}_P0{P0}_zmax{bz_max}_resum_k1.00.npy")
                    moms4_err = np.load(f"{save_path}/{init_char}/mellin_moms/mellin_moms4_err_Nmax{N_max}_h{Nh}_l{Nl}_P0{P0}_zmax{bz_max}.npy")
                    moms4_resum_err = np.load(f"{save_path}/{init_char}/mellin_moms/mellin_moms4_err_Nmax{N_max}_h{Nh}_l{Nl}_P0{P0}_zmax{bz_max}_resum_k1.00.npy")
                    data4[index:index+4] = np.array([moms4, moms4_err]).transpose() 
                    data4resum[index:index+4] = np.array([moms4_resum, moms4_resum_err]).transpose() 
                    index += 4

    
        init_conv = {"Z5": r"$\gamma_3\gamma_5$", "T5": r"$\gamma_0\gamma_5$"}
        # plt.suptitle(f"Mellin Moments for {init_conv[init_char]},"+ "$z_{max}= $"+f"{bz_max}", y=0.92, fontsize = 18)
        ###### BEGIN Z5 MM2 ######
        plt.subplot(1,2,1)
        plt.errorbar(
            data2[:,0],
            np.arange(0,Ns),
            xerr=(data2[:,1]),
            capsize=3,
            markerfacecolor="none",
            fmt="s",
            color="blue",
            label=r"Fixed $\mu$"
        )
        plt.errorbar(
            data2resum[:,0],
            np.arange(0,Ns),
            xerr=(data2resum[:,1]),
            capsize=3,
            markerfacecolor="none",
            fmt="s",
            color="red",
            label="Resummed",
        )
        plt.xlabel(r"$\langle x^2\rangle$")
        for i in range(0,Ns):
            plt.text(0.23, i, infos[i])
        plt.xlim(0.22, 0.4)
        plt.legend()
        plt.tick_params(
            axis='y',      
            which='both',    
            left="False" ,      
            labelleft=False) 

        plt.subplot(1,2,2)
        plt.errorbar(
            data4[:,0],
            np.arange(0,Ns),
            xerr=(data4[:,1]),
            capsize=3,
            markerfacecolor="none",
            fmt="s",
            color="blue",
            label=r"Fixed $\mu$"
        )
        plt.errorbar(
            data4resum[:,0],
            np.arange(0,Ns),
            xerr=(data4resum[:,1]),
            capsize=3,
            markerfacecolor="none",
            fmt="s",
            color="red",
            label="Resummed",
        )
        plt.xlabel(r"$\langle x^4\rangle$")
        for i in range(0,Ns):
            plt.text(0.02, i, infos[i])
        plt.xlim(0., 0.3)
        plt.legend()
        plt.tick_params(
            axis='y',      
            which='both',    
            left="False" ,      
            labelleft=False) 


        if save:
            plt.savefig(f"{save_path}/{init_char}mellin_resummcomp_P0{P0}_bzmax{bz_max}.pdf")
        plt.show()
        plt.close()


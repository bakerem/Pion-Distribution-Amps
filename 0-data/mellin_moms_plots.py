# Author Ethan Baker, ANL/Haverford College

import numpy as np
import matplotlib.pyplot as plt
import os
import scienceplots
from functions import phys_p
from scipy.special import gamma

Ns = 9
a = 2.359

plt.style.use('science')
plt.rcParams['figure.figsize'] = [10, 10]

save = True
for smear in ["final_results", "final_results_eps10"]:
    for init_char in ["Z5", "T5"]:
        save_path = f"{smear}/2_state_matrix_results_jack/"
        infos  = []
        data2 = np.zeros((Ns,3))
        data4 = np.zeros((Ns,3))
        index = 0
        for Nh in range(0,1):
            for Nl in range(0,1):
                for N_max in range(2,3):
                    for bz_max in range(0,3):
                        for Pz_min in range(0,1):
                            for Pz_max in range(0,3):
                                info = f"{2*N_max} {Nh} {Nl} {bz_max+6} {Pz_min+4} {Pz_max+7}"
                                infos.append(info)
                    moms2 = np.load(f"{save_path}/{init_char}/mellin_moms/mellin_moms2_Nmax{N_max}_h{Nh}_l{Nl}.npy")
                    moms4 = np.load(f"{save_path}/{init_char}/mellin_moms/mellin_moms4_Nmax{N_max}_h{Nh}_l{Nl}.npy")
                    moms2_err = np.load(f"{save_path}/{init_char}/mellin_moms/mellin_moms2_err_Nmax{N_max}_h{Nh}_l{Nl}.npy")
                    moms4_err = np.load(f"{save_path}/{init_char}/mellin_moms/mellin_moms4_err_Nmax{N_max}_h{Nh}_l{Nl}.npy")
                    chi2 =  np.load(f"{save_path}/{init_char}/mellin_moms/mellin_chi2_Nmax{N_max}_h{Nh}_l{Nl}.npy")
                    print(np.array([moms2, moms2_err, chi2]).transpose() .shape)
                    data2[index:index+9] = np.array([moms2, moms2_err, chi2]).transpose() 
                    data4[index:index+9] = np.array([moms4, moms4_err, chi2]).transpose() 
                    index +=9
                    print(np.max(chi2), np.max(chi2))

        # jackknife error calculation
        samples2     = np.zeros((Ns,))
        samples4     = np.zeros((Ns,))
        samples2_err = np.zeros((Ns,))
        samples4_err = np.zeros((Ns,))
        
        samples_s1_err   = np.zeros((Ns,))
        samples_s2_err   = np.zeros((Ns,))


        for i in range(len(data2)):
            samples2[i]     = np.average(np.delete(data2[:,0], i))
            samples4[i]     = np.average(np.delete(data4[:,0], i))
            samples2_err[i] = np.sqrt(np.average(np.delete(data2[:,0], i)**2) - (samples2[i])**2)
            samples4_err[i] = np.sqrt(np.average(np.delete(data4[:,0], i)**2) - (samples4[i])**2)

        avg2 = [np.average(samples2), np.sqrt(Ns-1)*np.std(samples2), np.average(samples2_err)]
        avg4 = [np.average(samples4), np.sqrt(Ns-1)*np.std(samples4), np.average(samples4_err)]
        print(avg2, avg4,)
        if save:
            os.makedirs(f"{save_path}/{init_char}/final_mm", exist_ok=True)
            np.save(f"{save_path}/{init_char}/final_mm/avg2_no_corr.npy", avg2)
            np.save(f"{save_path}/{init_char}/final_mm/avg4_no_corr.npy", avg4)

        init_conv = {"Z5": r"$\gamma_3\gamma_5$", "T5": r"$\gamma_0\gamma_5$"}
        plt.suptitle(f"Mellin Moments for {init_conv[init_char]}, " +f"$P_0$ = %.3f GeV" %(phys_p(a,3)), y=0.92, fontsize = 14)
        ###### BEGIN Z5 MM2 ######
        plt.subplot(1, 2, 1)
        plt.errorbar(
            data2[:,0],
            np.arange(0,Ns),
            xerr=(data2[:,1]),
            capsize=3,
            markerfacecolor="none",
            fmt="bs",
        )
        
        plt.xlabel(r"$\langle x^2\rangle$")
        for i in range(0,Ns):
            plt.text(0.23, i, infos[i])
        plt.xlim(0.22, 0.35)
        plt.tick_params(
            axis='y',      
            which='both',    
            left="False" ,      
            labelleft=False) 
        plt.axvline(avg2[0], color="black", linestyle="dashed")
        plt.axvspan(avg2[0]-avg2[1], avg2[0]+avg2[1], alpha=0.2, color='red')
        plt.axvspan(avg2[0]-avg2[1]-avg2[2], avg2[0]+avg2[1]+avg2[2], alpha=0.2, color='blue')

        ###### BEGIN Z5 MM4 ######
        plt.subplot(1,2,2)
        plt.errorbar(
            data4[:,0],
            np.arange(0,Ns),
            xerr=(data4[:,1]),
            capsize=3,
            markerfacecolor="none",
            fmt="bs",
        )
        
        plt.xlabel(r"$\langle x^4\rangle$")
        for i in range(0,Ns):
            plt.text(0.025, i, infos[i])
        plt.xlim(0., 0.3)

        plt.axvline(avg4[0], color="black", linestyle="dashed")
        plt.axvspan(avg4[0]-avg4[1], avg4[0]+avg4[1], alpha=0.2, color='red')
        plt.axvspan(avg4[0]-avg4[1]-avg4[2], avg4[0]+avg4[1]+avg4[2], alpha=0.2, color='blue')

        plt.tick_params(
            axis='y',      
            which='both',    
            left="False" ,      
            labelleft=False) 
        if save:
            plt.savefig(f"{save_path}/{init_char}mellin_nocorr.pdf")
        plt.show()


       






        # ##### Begin Z5 Chi2 ######
        # plt.figure()
        # plt.plot(
        #     data2[:,2],
        #     np.arange(0,Ns),
        #     "bs",
        # )
        # plt.title(r"$\chi^2$: $\gamma_3\gamma_5$")
        # plt.xlabel(r"$\chi^2$")
        # for i in range(0,Ns):
        #     plt.text(-0.3, i, infos[i])
        # plt.xlim(-0.35, 0.2)
        # plt.tick_params(
        #     axis='y',      
        #     which='both',    
        #     left="False" ,      
        #     labelleft=False) 
        # if save:
        #     plt.savefig(f"{save_path}/Z5mellin_chi2_Pz_range.pdf")
        # plt.show()


        # ##### Begin T5 Chi2 ######

        # plt.figure()
        # plt.plot(
        #     data2_T5[:,2],
        #     np.arange(0,Ns),
        #     "bs",
        # )
        # plt.title(r"$\chi^2$: $\gamma_0\gamma_5$")
        # plt.xlabel(r"$\chi^2$")
        # for i in range(0,Ns):
        #     plt.text(-0.3, i, infos[i])
        # plt.xlim(-0.35, 0.2)
        # plt.tick_params(
        #     axis='y',      
        #     which='both',    
        #     left="False" ,      
        #     labelleft=False) 
        # if save:
        #     plt.savefig(f"{save_path}/T5mellin_chi2_Pz_range.pdf")
        # plt.show()
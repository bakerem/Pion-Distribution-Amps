# Author Ethan Baker, ANL/Haverford College

import numpy as np
import matplotlib.pyplot as plt
import os
import scienceplots
import matplotlib as mpl
import matplotlib.font_manager as font_manager
from functions import phys_p
from scipy.special import gamma

a = 2.359
P0 = 1
# font_dirs = ["/home/bakerem/.local/lib/python3.8/site-packages/matplotlib/mpl-data/fonts/ttf"]
# font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
# for font_file in font_files:
#     font_manager.fontManager.addfont(font_file)

# mpl.rcParams['font.sans-serif'] = 'Lato'
# print(mpl.rcParams['font.sans-serif'])

plt.style.use('science')
plt.rcParams['figure.figsize'] = [6.5,9]
plt.rcParams["font.size"] = 6

save = True
for smear in ["final_results", "final_results_eps10"]:
    Ns = 80
    for init_char in ["T5"]:
        save_path = f"{smear}/2_state_matrix_results_jack"
        infos  = []
        data2 = np.zeros((Ns,3))
        data4 = np.zeros((Ns,3))
        datah0 = np.zeros((Ns,3))

        index = 0
        for Nh in range(0,1):
            for Nl in range(0,1):
                for N_max in range(2,3):
                    for kappa in np.arange(0.95,1.15,0.01):
                        for bz_max in range(5,6):
                            for Pz_min in range(0,4):
                                for Pz_max in range(0,2):
                                    info = "%0.2f  %i"%(kappa,Pz_min+6)
                                    infos.append(info)
                        moms2 = np.load(f"{save_path}/{init_char}/mellin_moms/mellin_moms2_Nmax{N_max}_h{Nh}_l{Nl}_P0{P0}_zmax{bz_max}_resum_k"+"%.2f"%kappa+".npy")
                        moms4 = np.load(f"{save_path}/{init_char}/mellin_moms/mellin_moms4_Nmax{N_max}_h{Nh}_l{Nl}_P0{P0}_zmax{bz_max}_resum_k"+"%.2f"%kappa+".npy")
                        # h0s   = np.load(f"{save_path}/{init_char}/mellin_moms/mellin_h0_Nmax{N_max}_h{Nh}_l{Nl}_P0{P0}_zmax{bz_max}.npy")

                        moms2_err = np.load(f"{save_path}/{init_char}/mellin_moms/mellin_moms2_err_Nmax{N_max}_h{Nh}_l{Nl}_P0{P0}_zmax{bz_max}_resum_k"+"%.2f"%kappa+".npy")
                        moms4_err = np.load(f"{save_path}/{init_char}/mellin_moms/mellin_moms4_err_Nmax{N_max}_h{Nh}_l{Nl}_P0{P0}_zmax{bz_max}_resum_k"+"%.2f"%kappa+".npy")
                        # h0s_err   = np.load(f"{save_path}/{init_char}/mellin_moms/mellin_h0_err_Nmax{N_max}_h{Nh}_l{Nl}_P0{P0}_zmax{bz_max}.npy")

                        chi2 =  np.load(f"{save_path}/{init_char}/mellin_moms/mellin_chi2_Nmax{N_max}_h{Nh}_l{Nl}_P0{P0}_zmax{bz_max}_resum_k"+"%.2f"%kappa+".npy")
                        
                        data2[index:index+4]  = np.array([moms2, moms2_err, chi2]).transpose() 
                        data4[index:index+4]  = np.array([moms4, moms4_err, chi2]).transpose() 
                        # datah0[index:index+4] = np.array([h0s, h0s_err, chi2]).transpose() 

                        index += 4
        print(np.std(data2[:,0]))
        # jackknife error calculation
        # data2_adj = data2[np.where(np.abs(data2[:,0]-np.average(data2[:,0]))<np.std(data2[:,0])),0]
        # data4_adj = data4[np.where(np.abs(data4[:,0]-np.average(data4[:,0]))<np.std(data4[:,0])),0]
        samples2     = np.zeros((Ns,))
        samples4     = np.zeros((Ns,))
        samplesh0    = np.zeros((Ns,))
        samples2_err = np.zeros((Ns,))
        samples4_err = np.zeros((Ns,))
        samplesh0_err= np.zeros((Ns,))


        for i in range(Ns):
            samples2[i]     = np.average(np.delete(data2[:,0], i))
            samples4[i]     = np.average(np.delete(data4[:,0], i))
            samplesh0[i]     = np.average(np.delete(datah0[:,0], i))

            samples2_err[i] = np.sqrt(np.average(np.delete(data2[:,0], i)**2) - (samples2[i])**2)
            samples4_err[i] = np.sqrt(np.average(np.delete(data4[:,0], i)**2) - (samples4[i])**2)
            samplesh0_err[i] = np.sqrt(np.average(np.delete(datah0[:,0], i)**2) - (samplesh0[i])**2)


        print(samples2[:15], data2[:15,0])

        avg2  = [np.average(samples2), np.sqrt(Ns-1)*np.std(samples2), np.average(samples2_err)]
        avg4  = [np.average(samples4), np.sqrt(Ns-1)*np.std(samples4), np.average(samples4_err)]
        avgh0 = [np.average(samplesh0), np.sqrt(Ns-1)*np.std(samplesh0), np.average(samplesh0_err)]

        print(avg2, avg4, avgh0)
        if save:
            os.makedirs(f"{save_path}/{init_char}/final_mm", exist_ok=True)
            np.save(f"{save_path}/{init_char}/final_mm/avg2_corr_P0{P0}.npy", avg2)
            np.save(f"{save_path}/{init_char}/final_mm/avg4_corr_P0{P0}.npy", avg4)
            np.save(f"{save_path}/{init_char}/final_mm/avgh0_corr_P0{P0}.npy", avgh0)


        init_conv = {"Z5": r"$\gamma_3\gamma_5$", "T5": r"$\gamma_0\gamma_5$"}
        plt.suptitle(f"Mellin Moments for {init_conv[init_char]}" +f"$P_0^3$ = %.3f GeV" %(phys_p(a,P0)), y=0.92, fontsize = 18)
        ###### BEGIN Z5 MM2 ######
        plt.subplot(1, 2, 1)
        plt.errorbar(
            data2[:,0],
            np.arange(0,Ns),
            xerr=(data2[:,1]),
            capsize=3,
            markerfacecolor="none",
            fmt="s",
            color="blue"
        )
        
        plt.xlabel(r"$\langle x^2\rangle$")
        for i in range(0,Ns):
            plt.text(0.31, i, infos[i])
        plt.xlim(0.3, 0.36)
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
            fmt="s",
            color="blue"
        )
        
        plt.xlabel(r"$\langle x^4\rangle$")
        for i in range(0,Ns):
            plt.text(0.06, i, infos[i])
        plt.xlim(0.05, 0.30)

        plt.axvline(avg4[0], color="black", linestyle="dashed")
        plt.axvspan(avg4[0]-avg4[1], avg4[0]+avg4[1], alpha=0.2, color='red')
        plt.axvspan(avg4[0]-avg4[1]-avg4[2], avg4[0]+avg4[1]+avg4[2], alpha=0.2, color='blue')

        plt.tick_params(
            axis='y',      
            which='both',    
            left="False" ,      
            labelleft=False) 
        if save:
            plt.savefig(f"{save_path}/{init_char}mellin_corr_P0{P0}_bzmax{bz_max}.pdf")
        plt.show()
        plt.close()


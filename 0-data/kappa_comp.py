# Author Ethan Baker, ANL/Haverford College

import numpy as np
import matplotlib.pyplot as plt
import os
import scienceplots
import matplotlib as mpl
import matplotlib.font_manager as font_manager
from functions import phys_p
from scipy.special import gamma
from scipy.optimize import curve_fit

a = 2.359
P0 = 1
# font_dirs = ["/home/bakerem/.local/lib/python3.8/site-packages/matplotlib/mpl-data/fonts/ttf"]
# font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
# for font_file in font_files:
#     font_manager.fontManager.addfont(font_file)

# mpl.rcParams['font.sans-serif'] = 'Lato'
# print(mpl.rcParams['font.sans-serif'])

plt.style.use('science')
plt.rcParams['figure.figsize'] = [10,10]
# plt.rcParams["font.size"] = 12
kappa_list = np.arange(0.9,1.5,0.05)
Ns = len(kappa_list)
save = True
for smear in ["final_results", ]:
    for init_char in ["T5"]:
        save_path = f"{smear}/2_state_matrix_results_jack"
        infos  = []
        data2 = np.zeros((Ns,2))
        index = 0
        for i, kappa in enumerate(kappa_list):        
            moms2 = np.load(f"{save_path}/{init_char}/mellin_moms/mellin_moms2_Nmax{2}_h{1}_l{0}_P0{P0}_zmax{3}_resum_k"+"%.2f"%kappa+".npy")
            index += 4
            # jackknife error calculation
            samples2     = np.zeros((4,))
        
            for s in range(len(moms2)):
                samples2[s]     = np.average(np.delete(moms2, s))
            data2[i] = np.array([np.average(samples2), (Ns-1)*np.std(samples2),]).transpose() 

        def func(kappa,a,b):
            y = a*np.exp(-kappa) + b
            return y
        popt, pcov = curve_fit(func,np.array(list(map(float,kappa_list))),data2[:,0], sigma=data2[:,1])


        init_conv = {"Z5": r"$\gamma_3\gamma_5$", "T5": r"$\gamma_0\gamma_5$"}
        plt.suptitle(f"2nd Moment for Low Smear", y=0.92, fontsize = 18)
        ###### BEGIN Z5 MM2 ######
        plt.errorbar(
            kappa_list,
            data2[:,0],
            xerr=(data2[:,1]),
            capsize=3,
            markerfacecolor="none",
            fmt="s",
            color="blue"
        )
        # plt.plot(np.arange(0.85,1.41,0.01),func(np.arange(0.85,1.41,0.01),*popt))
        plt.xlabel(r"$\kappa$")
        plt.ylabel(r"$\langle x^2 \rangle$")
       
        
        if save:
            plt.savefig(f"{save_path}/{init_char}mellin_kappa_low_smear.pdf")
        plt.show()
        plt.close()


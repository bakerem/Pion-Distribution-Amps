# Author Ethan Baker, ANL/Haverford College

import numpy as np
import os
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as font_manager
from functions import m_ope, phys_p
import scienceplots

plt.style.use("science")

# font_dirs = ["/home/bakerem/.local/lib/python3.8/site-packages/matplotlib/mpl-data/fonts/ttf"]
# font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
# for font_file in font_files:
#     font_manager.fontManager.addfont(font_file)

# mpl.rcParams['font.sans-serif'] = 'Lato'
# print(mpl.rcParams['font.sans-serif'])
# plt.rcParams["font.size"]= 14
Nt = 128
a = 2.359 # GeV^-1
P0 = 3
N_max = 2

save = True
info = []
init_conv = {"Z5": r"$\gamma_3\gamma_5$", "T5": r"$\gamma_0\gamma_5$"}

# initialize empty arrays for saving data
index = 0
plt.figure()
for smear in ["final_results"]:
    for init_char in ["Z5","T5"]:
        for corr in ["no_corr","corr"]:
            for P0 in [1,3]:
                # create save path
                info.append(init_conv[init_char]+ f"{corr} {P0}")
                save_path = f"{smear}/2_state_matrix_results_jack/{init_char}/final_mm"
                os.makedirs(save_path, exist_ok=True)
                mm2 = np.load(f"{save_path}/avg2_{corr}_P0{P0}.npy")
                mm4 = np.load(f"{save_path}/avg4_{corr}_P0{P0}.npy")
                print(mm2)
                print(mm4)
                #load in data

                # function for fitting calculates ratio and wraps around Mellin-OPE of 
                                
                plt.errorbar(mm4[0], index, xerr=(mm4[1]+mm4[2]), color="black", fmt="s", capsize=3, markerfacecolor="none")
                plt.text(0.01, index, info[index])
                index += 1
plt.xlabel(r"$\angle x^2 \rangle$")
# plt.ylabel(r"Re $\mathcal{M}$")
plt.xlim(0.,0.4)
# plt.title(f"Comparison of Mellin Moments, {init_conv[init_char]}")
if save:
    plt.savefig(f"{smear}/2_state_matrix_results_jack/P0mm_fit_comp_mm4.pdf")
plt.show()

            
            
        
        





# Author Ethan Baker, ANL/Haverford College

import numpy as np
import matplotlib.pyplot as plt
import os


Nt = 128
a = 2.359
mu = 2/a
P0 = 4
Pz = 6
N_max = 2
bz = 4
bz_min = 2
bz_max = 10
Pz = 6
Pz_min = 5
Pz_max = 8
save = False
save_path = "final_results/2_state_matrix_results_jack/"
infos  = []
data2 = np.zeros((12,3))
data4 = np.zeros((12,3))
data6 = np.zeros((12,3))
data8 = np.zeros((12,3))
index = 0
for Nh in range(0,2):
    for Nl in range(0,2):
        for N_max in range(2,5):
            info = f"{N_max} {Nh} {Nl}"
            moms2_Z5 = np.load(f"final_results/2_state_matrix_results_jack/Z5/gegen_moms/gegen_moms2_Nmax{N_max}_h{Nh}_l{Nl}.npy")
            moms2_T5 = np.load(f"final_results/2_state_matrix_results_jack/T5/gegen_moms/gegen_moms2_Nmax{N_max}_h{Nh}_l{Nl}.npy")
            moms4_Z5 = np.load(f"final_results/2_state_matrix_results_jack/Z5/gegen_moms/gegen_moms4_Nmax{N_max}_h{Nh}_l{Nl}.npy")
            moms4_T5 = np.load(f"final_results/2_state_matrix_results_jack/T5/gegen_moms/gegen_moms4_Nmax{N_max}_h{Nh}_l{Nl}.npy")
            moms2_Z5_err = np.load(f"final_results/2_state_matrix_results_jack/Z5/gegen_moms/gegen_moms2_err_Nmax{N_max}_h{Nh}_l{Nl}.npy")
            moms2_T5_err = np.load(f"final_results/2_state_matrix_results_jack/T5/gegen_moms/gegen_moms2_err_Nmax{N_max}_h{Nh}_l{Nl}.npy")
            moms4_Z5_err = np.load(f"final_results/2_state_matrix_results_jack/Z5/gegen_moms/gegen_moms4_err_Nmax{N_max}_h{Nh}_l{Nl}.npy")
            moms4_T5_err = np.load(f"final_results/2_state_matrix_results_jack/T5/gegen_moms/gegen_moms4_err_Nmax{N_max}_h{Nh}_l{Nl}.npy")
            # moms6_Z5 = np.load(f"final_results/2_state_matrix_results_jack/Z5/gegen_moms/gegen_moms6_Nmax{N_max}_h{Nh}_l{Nl}.npy")
            # moms6_T5 = np.load(f"final_results/2_state_matrix_results_jack/T5/gegen_moms/gegen_moms6_Nmax{N_max}_h{Nh}_l{Nl}.npy")
            # moms8_Z5 = np.load(f"final_results/2_state_matrix_results_jack/Z5/gegen_moms/gegen_moms8_Nmax{N_max}_h{Nh}_l{Nl}.npy")
            # moms8_T5 = np.load(f"final_results/2_state_matrix_results_jack/T5/gegen_moms/gegen_moms8_Nmax{N_max}_h{Nh}_l{Nl}.npy")
            moms2 = np.concatenate((moms2_Z5.flatten(), moms2_T5.flatten()))
            moms4 = np.concatenate((moms4_Z5.flatten(), moms4_T5.flatten()))
            moms2_err = np.concatenate((moms2_Z5_err.flatten(), moms2_T5_err.flatten()))
            moms4_err = np.concatenate((moms4_Z5_err.flatten(), moms4_T5_err.flatten()))
            # moms6 = np.concatenate((moms6_Z5.flatten(), moms6_T5.flatten()))
            # moms8 = np.concatenate((moms8_Z5.flatten(), moms8_T5.flatten()))
            samples2 = np.zeros((len(moms2),))
            samples4 = np.zeros((len(moms4),))
            sample_errs2 = np.zeros((len(moms2),))
            sample_errs4 = np.zeros((len(moms4),)) 
            # samples6 = np.zeros((len(moms6),))
            # samples8 = np.zeros((len(moms8),))
            for i in range(len(moms2)):
                samples2[i] = np.average(np.delete(moms2.flatten(), i))
                samples4[i] = np.average(np.delete(moms4.flatten(), i))
                sample_errs2[i] = np.average(np.delete(moms2_err.flatten(), i))
                sample_errs4[i] = np.average(np.delete(moms4_err.flatten(), i))
                # samples6[i] = np.average(np.delete(moms6.flatten(), i))
                # samples8[i] = np.average(np.delete(moms8.flatten(), i))
            
            infos.append(info)
            data2[index] = np.array((np.average(0.2+(12/35)*samples2), np.sqrt(len(moms2)-1)*np.std(0.2+(12/35)*samples2), np.average(0.2+(12/35)*sample_errs2)))
            data4[index] = np.array((np.average(samples4), np.sqrt(len(moms4)-1)*np.std(samples4), np.average(sample_errs4)))
            # data6[index] = np.array((np.average(samples6), np.sqrt(len(moms6)-1)*np.std(samples6)))
            # data8[index] = np.array((np.average(samples8), np.sqrt(len(moms8)-1)*np.std(samples8)))
            index +=1
            

# print(data2[0,:])
# print(data4)
# print("The second moment is %.4f" %(np.average(samples2)))
# print("The error is %.5f" %(np.sqrt(len(moms2)-1)*np.std(samples2)))

# print("The fourth moment is %.4f" %(np.average(samples4)))
# print("The error is %.5f" %(np.sqrt(len(moms2)-1)*np.std(samples4)))



plt.figure()
plt.errorbar(
    data2[:,0],
    np.arange(0,12),
    xerr=(data2[:,1]),
    capsize=3,
    fmt="bs",
)
# plt.ylim(-0.5, 0.3)
plt.title(r"$\langle x^2\rangle$")
plt.xlabel(r"$\langle x^2\rangle$")
for i in range(0,12):
    plt.text(0.25, i, infos[i])
# plt.xlim(0.24, 0.45)
plt.tick_params(
    axis='y',      
    which='both',    
    left="False" ,      
    labelleft=False) 
if save:
    plt.savefig(f"{save_path}/gegen2.png")
plt.show()

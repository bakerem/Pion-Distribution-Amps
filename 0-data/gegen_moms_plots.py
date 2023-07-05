# Author Ethan Baker, ANL/Haverford College

import numpy as np
import matplotlib.pyplot as plt
import os



save = True
save_path = "final_results_eps10/2_state_matrix_results_jack"
infos  = []
data2_Z5 = np.zeros((4,3))
data4_Z5 = np.zeros((4,3))
data2_T5 = np.zeros((4,3))
data4_T5 = np.zeros((4,3))
index = 0
for Nh in range(0,2):
    for Nl in range(0,2):
        for N_max in range(2,3):
            info = f"{2*N_max} {Nh} {Nl}"
            moms2_Z5 = np.load(f"{save_path}/Z5/gegen_moms/gegen_moms2_Nmax{N_max}_h{Nh}_l{Nl}.npy")
            moms2_T5 = np.load(f"{save_path}/T5/gegen_moms/gegen_moms2_Nmax{N_max}_h{Nh}_l{Nl}.npy")
            moms4_Z5 = np.load(f"{save_path}/Z5/gegen_moms/gegen_moms4_Nmax{N_max}_h{Nh}_l{Nl}.npy")
            moms4_T5 = np.load(f"{save_path}/T5/gegen_moms/gegen_moms4_Nmax{N_max}_h{Nh}_l{Nl}.npy")
            
            moms2_Z5_err = np.load(f"{save_path}/Z5/gegen_moms/gegen_moms2_err_Nmax{N_max}_h{Nh}_l{Nl}.npy")
            moms2_T5_err = np.load(f"{save_path}/T5/gegen_moms/gegen_moms2_err_Nmax{N_max}_h{Nh}_l{Nl}.npy")
            moms4_Z5_err = np.load(f"{save_path}/Z5/gegen_moms/gegen_moms4_err_Nmax{N_max}_h{Nh}_l{Nl}.npy")
            moms4_T5_err = np.load(f"{save_path}/T5/gegen_moms/gegen_moms4_err_Nmax{N_max}_h{Nh}_l{Nl}.npy")
            
            samples2_Z5 = np.zeros((len(moms2_Z5),))
            samples4_Z5 = np.zeros((len(moms4_Z5),))
            sample_errs2_Z5 = np.zeros((len(moms2_Z5),))
            sample_errs4_Z5 = np.zeros((len(moms4_Z5),)) 
            samples2_T5 = np.zeros((len(moms2_T5),))
            samples4_T5 = np.zeros((len(moms4_T5),))
            sample_errs2_T5 = np.zeros((len(moms2_T5),))
            sample_errs4_T5 = np.zeros((len(moms4_T5),)) 
            for i in range(len(moms2_Z5)):
                samples2_Z5[i] = np.average(np.delete(moms2_Z5, i))
                samples4_Z5[i] = np.average(np.delete(moms4_Z5, i))
                sample_errs2_Z5[i] = np.average(np.delete(moms2_Z5_err, i))
                sample_errs4_Z5[i] = np.average(np.delete(moms4_Z5_err, i))
                samples2_T5[i] = np.average(np.delete(moms2_T5, i))
                samples4_T5[i] = np.average(np.delete(moms4_T5, i))
                sample_errs2_T5[i] = np.average(np.delete(moms2_T5_err, i))
                sample_errs4_T5[i] = np.average(np.delete(moms4_T5_err, i))
            
            infos.append(info)
            data2_Z5[index] = np.array((0.2+(12/35)*np.average(samples2_Z5), np.sqrt(len(moms2_Z5)-1)*np.std(0.2+(12/35)*samples2_Z5), np.average(sample_errs2_Z5)))
            data4_Z5[index] = np.array((0.2+(12/35)*np.average(samples4_Z5), np.sqrt(len(moms4_Z5)-1)*np.std(0.2+(12/35)*samples4_Z5), np.average(sample_errs4_Z5)))
            data2_T5[index] = np.array((0.2+(12/35)*np.average(samples2_T5), np.sqrt(len(moms2_T5)-1)*np.std(0.2+(12/35)*samples2_T5), np.average(sample_errs2_T5)))
            data4_T5[index] = np.array((0.2+(12/35)*np.average(samples4_T5), np.sqrt(len(moms4_T5)-1)*np.std(0.2+(12/35)*samples4_T5), np.average(sample_errs4_T5)))
            index +=1
            





plt.figure()
plt.errorbar(
    data2_Z5[:,0],
    np.arange(0,4),
    xerr=(data2_Z5[:,1]+data2_Z5[:,2]),
    capsize=3,
    fmt="bs",
)
# plt.ylim(-0.5, 0.3)
plt.title(r"$\langle x^2\rangle$: $\gamma_3\gamma_5$")
plt.xlabel(r"$\langle x^2\rangle$")
for i in range(0,4):
    plt.text(0.25, i, infos[i])
plt.xlim(0.24, 0.45)
plt.tick_params(
    axis='y',      
    which='both',    
    left="False" ,      
    labelleft=False) 
if save:
    plt.savefig(f"{save_path}/Z5gegen2.png")
plt.show()

plt.figure()
plt.errorbar(
    data2_T5[:,0],
    np.arange(0,4),
    xerr=(data2_T5[:,1]+data2_T5[:,2]),
    capsize=3,
    fmt="bs",
)
# plt.ylim(-0.5, 0.3)
plt.title(r"$\langle x^2\rangle$: $\gamma_0\gamma_5$")
plt.xlabel(r"$\langle x^2\rangle$")
for i in range(0,4):
    plt.text(0.25, i, infos[i])
plt.xlim(0.24, 0.45)
plt.tick_params(
    axis='y',      
    which='both',    
    left="False" ,      
    labelleft=False) 
if save:
    plt.savefig(f"{save_path}/T5gegen2.png")
plt.show()

# plt.figure()
# plt.errorbar(
#     data4_Z5[:,0],
#     np.arange(0,4),
#     xerr=(data4_Z5[:,1]+data4_Z5[:,2]),
#     capsize=3,
#     fmt="bs",
# )
# plt.title(r"$\langle x^4\rangle$: $\gamma_3\gamma_5$")
# plt.xlabel(r"$\langle x^4\rangle$")
# for i in range(0,4):
#     plt.text(0.1, i, infos[i])
# plt.xlim(0.05, 2.25)
# plt.tick_params(
#     axis='y',      
#     which='both',    
#     left="False" ,      
#     labelleft=False) 
# if save:
#     plt.savefig(f"{save_path}/Z5gegen4.png")
# plt.show()


# plt.figure()
# plt.errorbar(
#     data4_T5[:,0],
#     np.arange(0,4),
#     xerr=(data4_T5[:,1]+data4_T5[:,2]),
#     capsize=3,
#     fmt="bs",
# )
# plt.title(r"$\langle x^4\rangle$: $\gamma_0\gamma_5$")
# plt.xlabel(r"$\langle x^4\rangle$")
# for i in range(0,4):
#     plt.text(0.1, i, infos[i])
# plt.xlim(0.05, 2.25)
# plt.tick_params(
#     axis='y',      
#     which='both',    
#     left="False" ,      
#     labelleft=False) 
# if save:
#     plt.savefig(f"{save_path}/T5gegen4.png")
# plt.show()

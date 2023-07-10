# Author Ethan Baker, ANL/Haverford College

import numpy as np
import matplotlib.pyplot as plt
import os
import scienceplots

Ns = 18

plt.style.use('science')
plt.rcParams['figure.figsize'] = [6.75, 9]

save = True
save_path = "final_results_eps10/2_state_matrix_results_jack"
infos  = []
data2_Z5 = np.zeros((Ns,2))
data4_Z5 = np.zeros((Ns,2))
data2_T5 = np.zeros((Ns,2))
data4_T5 = np.zeros((Ns,2))
index = 0
for Nh in range(0,1):
    for Nl in range(0,1):
        for N_max in range(2,3):
            for bz_max in range(0,3):
                for Pz_min in range(0,2):
                    for Pz_max in range(0,3):
                        info = f"{2*N_max} {bz_max+6} {Pz_min+2} {Pz_max+7}"
                        infos.append(info)
            moms2_Z5 = np.load(f"{save_path}/Z5/tree_moms/tree_moms2_Nmax{N_max}_h{Nh}_l{Nl}.npy")
            moms2_T5 = np.load(f"{save_path}/T5/tree_moms/tree_moms2_Nmax{N_max}_h{Nh}_l{Nl}.npy")
            moms4_Z5 = np.load(f"{save_path}/Z5/tree_moms/tree_moms4_Nmax{N_max}_h{Nh}_l{Nl}.npy")
            moms4_T5 = np.load(f"{save_path}/T5/tree_moms/tree_moms4_Nmax{N_max}_h{Nh}_l{Nl}.npy")
            moms2_Z5_err = np.load(f"{save_path}/Z5/tree_moms/tree_moms2_err_Nmax{N_max}_h{Nh}_l{Nl}.npy")
            moms2_T5_err = np.load(f"{save_path}/T5/tree_moms/tree_moms2_err_Nmax{N_max}_h{Nh}_l{Nl}.npy")
            moms4_Z5_err = np.load(f"{save_path}/Z5/tree_moms/tree_moms4_err_Nmax{N_max}_h{Nh}_l{Nl}.npy")
            moms4_T5_err = np.load(f"{save_path}/T5/tree_moms/tree_moms4_err_Nmax{N_max}_h{Nh}_l{Nl}.npy")
            data2_Z5[index:index+18] = np.array([moms2_Z5, moms2_Z5_err]).transpose() 
            data4_Z5[index:index+18] = np.array([moms4_Z5, moms4_Z5_err]).transpose() 
            data2_T5[index:index+18] = np.array([moms2_T5, moms2_T5_err]).transpose() 
            data4_T5[index:index+18] = np.array([moms4_T5, moms4_T5_err]).transpose() 
            index +=18
            

# jackknife error calculation
samples2_Z5     = np.zeros((Ns,))
samples4_Z5     = np.zeros((Ns,))
samples2_T5     = np.zeros((Ns,))
samples4_T5     = np.zeros((Ns,))
samples2_Z5_err = np.zeros((Ns,))
samples4_Z5_err = np.zeros((Ns,))
samples2_T5_err = np.zeros((Ns,))
samples4_T5_err = np.zeros((Ns,))


for i in range(len(data2_Z5)):
    samples2_Z5[i]     = 0.2 + (12/35)*np.average(np.delete(data2_Z5[:,0], i))
    samples4_Z5[i]     = np.average(np.delete(data4_Z5[:,0], i))
    samples2_T5[i]     = 0.2 + (12/35)*np.average(np.delete(data2_T5[:,0], i))
    samples4_T5[i]     = np.average(np.delete(data4_T5[:,0], i))
    samples2_Z5_err[i] = np.sqrt(np.average(np.delete(0.2 + (12/35)*data2_Z5[:,0], i)**2) - (samples2_Z5[i])**2)
    samples4_Z5_err[i] = np.sqrt(np.average(np.delete(data4_Z5[:,0], i)**2) - (samples4_Z5[i])**2)
    samples2_T5_err[i] = np.sqrt(np.average(np.delete(0.2 + (12/35)*data2_T5[:,0], i)**2) - (samples2_T5[i])**2)

avg2_Z5 = [np.average(samples2_Z5), np.sqrt(Ns-1)*np.std(0.2 + (12/35)*samples2_Z5), np.average(samples2_Z5_err)]
avg2_T5 = [np.average(samples2_T5), np.sqrt(Ns-1)*np.std(0.2 + (12/35)*samples2_T5), np.average(samples2_T5_err)]
print(avg2_Z5, avg2_T5)


###### BEGIN Z5 MM2 ######
plt.figure()
plt.title(r"Tree Level for $\gamma_3\gamma_5$")
err = np.abs((0.2 + (12/35)*data2_Z5[:,0]) - (0.2 + (12/35)*(data2_Z5[:,0]+data2_Z5[:,1])))
plt.errorbar(
    0.2 + (12/35)*data2_Z5[:,0],
    np.arange(0,Ns),
    xerr=(data2_Z5[:,1]),
    capsize=3,
    markerfacecolor="none",
    fmt="bs",
)
plt.xlabel(r"$\langle x^2 \rangle$")
for i in range(0,Ns):
    plt.text(0.23, i, infos[i])
plt.xlim(0.22, 0.40)
plt.tick_params(
    axis='y',      
    which='both',    
    left="False" ,      
    labelleft=False) 
plt.axvline(avg2_Z5[0], color="black", linestyle="dashed")
plt.axvspan(avg2_Z5[0]-avg2_Z5[1], avg2_Z5[0]+avg2_Z5[1], alpha=0.2, color='red')
plt.axvspan(avg2_Z5[0]-avg2_Z5[1]-avg2_Z5[2], avg2_Z5[0]+avg2_Z5[1]+avg2_Z5[2], alpha=0.2, color='blue')
if save:
    plt.savefig(f"{save_path}/Z5tree_Pz_range.pdf")
plt.show()


plt.figure()
plt.title(r"Tree Level for $\gamma_0\gamma_5$")
###### BEGIN T5 MM2 ######
err = np.abs((0.2 + (12/35)*data2_T5[:,0]) - (0.2 + (12/35)*(data2_T5[:,0]+data2_T5[:,1])))
plt.errorbar(
    0.2 + (12/35)*data2_T5[:,0],
    np.arange(0,Ns),
    xerr=err,
    capsize=3,
    markerfacecolor="none",
    fmt="bs",
)
plt.xlabel(r"$\langle x^2 \rangle$")
for i in range(0,Ns):
    plt.text(0.21, i, infos[i])
plt.xlim(0.20, 0.35)
plt.axvline(avg2_T5[0], color="black", linestyle="dashed")
plt.axvspan(avg2_T5[0]-avg2_T5[1], avg2_T5[0]+avg2_T5[1], alpha=0.2, color='red')
plt.axvspan(avg2_T5[0]-avg2_T5[1]-avg2_T5[2], avg2_T5[0]+avg2_T5[1]+avg2_T5[2], alpha=0.2, color='blue')
plt.tick_params(
    axis='y',      
    which='both',    
    left="False" ,      
    labelleft=False) 
if save:
    plt.savefig(f"{save_path}/T5tree_Pz_range.pdf")
plt.show()

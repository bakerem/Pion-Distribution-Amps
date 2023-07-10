# Author Ethan Baker, ANL/Haverford College

import numpy as np
import matplotlib.pyplot as plt
import os
import scienceplots

Ns = 72

plt.style.use('science')
plt.rcParams['figure.figsize'] = [6.75, 9]

save = True
save_path = "final_results/2_state_matrix_results_jack"
infos  = []
data2_Z5 = np.zeros((Ns,2))
data4_Z5 = np.zeros((Ns,2))
data2_T5 = np.zeros((Ns,2))
data4_T5 = np.zeros((Ns,2))
index = 0
for Nh in range(0,2):
    for Nl in range(0,2):
        for N_max in range(2,3):
            for bz_max in range(0,3):
                for Pz_min in range(0,2):
                    for Pz_max in range(0,3):
                        info = f"{Nh} {Nl} {2*N_max} {bz_max+6} {Pz_min+2} {Pz_max+7}"
                        infos.append(info)
            moms2_Z5 = np.load(f"{save_path}/Z5/gegen_moms/gegen_moms2_Nmax{N_max}_h{Nh}_l{Nl}.npy")
            moms2_T5 = np.load(f"{save_path}/T5/gegen_moms/gegen_moms2_Nmax{N_max}_h{Nh}_l{Nl}.npy")
            moms4_Z5 = np.load(f"{save_path}/Z5/gegen_moms/gegen_moms4_Nmax{N_max}_h{Nh}_l{Nl}.npy")
            moms4_T5 = np.load(f"{save_path}/T5/gegen_moms/gegen_moms4_Nmax{N_max}_h{Nh}_l{Nl}.npy")
            moms2_Z5_err = np.load(f"{save_path}/Z5/gegen_moms/gegen_moms2_err_Nmax{N_max}_h{Nh}_l{Nl}.npy")
            moms2_T5_err = np.load(f"{save_path}/T5/gegen_moms/gegen_moms2_err_Nmax{N_max}_h{Nh}_l{Nl}.npy")
            moms4_Z5_err = np.load(f"{save_path}/Z5/gegen_moms/gegen_moms4_err_Nmax{N_max}_h{Nh}_l{Nl}.npy")
            moms4_T5_err = np.load(f"{save_path}/T5/gegen_moms/gegen_moms4_err_Nmax{N_max}_h{Nh}_l{Nl}.npy")
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
    samples2_Z5[i]     = np.average(np.delete(data2_Z5[:,0], i))
    samples4_Z5[i]     = np.average(np.delete(data4_Z5[:,0], i))
    samples2_T5[i]     = np.average(np.delete(data2_T5[:,0], i))
    samples4_T5[i]     = np.average(np.delete(data4_T5[:,0], i))
    samples2_Z5_err[i] = np.sqrt(np.average(np.delete(data2_Z5[:,0], i)**2) - (samples2_Z5[i])**2)
    samples4_Z5_err[i] = np.sqrt(np.average(np.delete(data4_Z5[:,0], i)**2) - (samples4_Z5[i])**2)
    samples2_T5_err[i] = np.sqrt(np.average(np.delete(data2_T5[:,0], i)**2) - (samples2_T5[i])**2)
    samples4_T5_err[i] = np.sqrt(np.average(np.delete(data4_T5[:,0], i)**2) - (samples4_T5[i])**2)

avg2_Z5 = [np.average(samples2_Z5), np.sqrt(Ns-1)*np.std(samples2_Z5), np.average(samples2_Z5_err)]
avg4_Z5 = [np.average(samples4_Z5), np.sqrt(Ns-1)*np.std(samples4_Z5), np.average(samples4_Z5_err)]
avg2_T5 = [np.average(samples2_T5), np.sqrt(Ns-1)*np.std(samples2_T5), np.average(samples2_T5_err)]
avg4_T5 = [np.average(samples4_T5), np.sqrt(Ns-1)*np.std(samples4_T5), np.average(samples4_T5_err)]
print(avg2_Z5, avg4_Z5, avg2_T5, avg4_T5)


# plt.suptitle(r"Gegen Moments for $\gamma_3\gamma_5$")
###### BEGIN Z5 MM2 ######
plt.subplot(1, 2, 1)
plt.errorbar(
    data2_Z5[:,0],
    np.arange(0,Ns),
    xerr=(data2_Z5[:,1]),
    capsize=3,
    markerfacecolor="none",
    fmt="bs",
)
plt.xlabel(r"$a_2$")
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

###### BEGIN Z5 MM4 ######
plt.subplot(1,2,2)
plt.errorbar(
    data4_Z5[:,0],
    np.arange(0,Ns),
    xerr=(data4_Z5[:,1]),
    capsize=3,
    markerfacecolor="none",
    fmt="bs",
)
plt.xlabel(r"$a_4$")
for i in range(0,Ns):
    plt.text(0.05, i, infos[i])
plt.xlim(0., 0.8)

plt.axvline(avg4_Z5[0], color="black", linestyle="dashed")
plt.axvspan(avg4_Z5[0]-avg4_Z5[1], avg4_Z5[0]+avg4_Z5[1], alpha=0.2, color='red')
plt.axvspan(avg4_Z5[0]-avg4_Z5[1]-avg4_Z5[2], avg4_Z5[0]+avg4_Z5[1]+avg4_Z5[2], alpha=0.2, color='blue')

plt.tick_params(
    axis='y',      
    which='both',    
    left="False" ,      
    labelleft=False) 
if save:
    plt.savefig(f"{save_path}/Z5gegen_Pz_range.pdf")
plt.show()


plt.figure()
# plt.suptitle(r"Gegen Moments for $\gamma_0\gamma_5$")
###### BEGIN T5 MM2 ######
plt.subplot(1,2,1)
plt.errorbar(
    data2_T5[:,0],
    np.arange(0,Ns),
    xerr=(data2_T5[:,1]),
    capsize=3,
    markerfacecolor="none",
    fmt="bs",
)
plt.xlabel(r"$a_2$")
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


###### BEGIN T5 MM4 ######

plt.subplot(1,2,2)
plt.errorbar(
    data4_T5[:,0],
    np.arange(0,Ns),
    xerr=(data4_T5[:,1]),
    capsize=3,
    markerfacecolor="none",
    fmt="bs",
)
# plt.title(r"$\langle x^4\rangle$: $\gamma_0\gamma_5$")
plt.xlabel(r"$a_4$")
for i in range(0,Ns):
    plt.text(-0.3, i, infos[i])
plt.xlim(-0.35, 0.2)
plt.axvline(avg4_T5[0], color="black", linestyle="dashed")
plt.axvspan(avg4_T5[0]-avg4_T5[1], avg4_T5[0]+avg4_T5[1], alpha=0.2, color='red')
plt.axvspan(avg4_T5[0]-avg4_T5[1]-avg4_T5[2], avg4_T5[0]+avg4_T5[1]+avg4_T5[2], alpha=0.2, color='blue')
plt.tick_params(
    axis='y',      
    which='both',    
    left="False" ,      
    labelleft=False) 
if save:
    plt.savefig(f"{save_path}/T5gegen_Pz_range.pdf")
plt.show()

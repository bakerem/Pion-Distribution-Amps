# Author Ethan Baker, ANL/Haverford College

import numpy as np
import matplotlib.pyplot as plt
import os
import scienceplots
from functions import phys_p

Ns = 24
a = 2.359

plt.style.use('science')
plt.rcParams['figure.figsize'] = [10, 10]

save = True
save_path = "final_results_eps10/2_state_matrix_results_jack"
infos  = []
data2_Z5 = np.zeros((Ns,3))
data4_Z5 = np.zeros((Ns,3))
data2_T5 = np.zeros((Ns,3))
data4_T5 = np.zeros((Ns,3))
index = 0
for Nh in range(0,2):
    for Nl in range(0,2):
        for N_max in range(2,3):
            for bz_max in range(0,3):
                for Pz_min in range(0,2):
                    for Pz_max in range(0,1):
                        info = f"{2*N_max} {bz_max+6} {Pz_min+5} {Pz_max+7}"
                        infos.append(info)
            moms2_Z5 = np.load(f"{save_path}/Z5/mellin_moms/mellin_moms2_Nmax{N_max}_h{Nh}_l{Nl}.npy")
            moms2_T5 = np.load(f"{save_path}/T5/mellin_moms/mellin_moms2_Nmax{N_max}_h{Nh}_l{Nl}.npy")
            moms4_Z5 = np.load(f"{save_path}/Z5/mellin_moms/mellin_moms4_Nmax{N_max}_h{Nh}_l{Nl}.npy")
            moms4_T5 = np.load(f"{save_path}/T5/mellin_moms/mellin_moms4_Nmax{N_max}_h{Nh}_l{Nl}.npy")
            moms2_Z5_err = np.load(f"{save_path}/Z5/mellin_moms/mellin_moms2_err_Nmax{N_max}_h{Nh}_l{Nl}.npy")
            moms2_T5_err = np.load(f"{save_path}/T5/mellin_moms/mellin_moms2_err_Nmax{N_max}_h{Nh}_l{Nl}.npy")
            moms4_Z5_err = np.load(f"{save_path}/Z5/mellin_moms/mellin_moms4_err_Nmax{N_max}_h{Nh}_l{Nl}.npy")
            moms4_T5_err = np.load(f"{save_path}/T5/mellin_moms/mellin_moms4_err_Nmax{N_max}_h{Nh}_l{Nl}.npy")
            chi2_Z5 =  np.load(f"{save_path}/Z5/mellin_moms/mellin_chi2_Nmax{N_max}_h{Nh}_l{Nl}.npy")
            chi2_T5  = np.load(f"{save_path}/T5/mellin_moms/mellin_chi2_Nmax{N_max}_h{Nh}_l{Nl}.npy")
            print(np.array([moms2_Z5, moms2_Z5_err, chi2_Z5]).transpose() .shape)
            data2_Z5[index:index+6] = np.array([moms2_Z5, moms2_Z5_err, chi2_Z5]).transpose() 
            data4_Z5[index:index+6] = np.array([moms4_Z5, moms4_Z5_err, chi2_Z5]).transpose() 
            data2_T5[index:index+6] = np.array([moms2_T5, moms2_T5_err, chi2_T5]).transpose() 
            data4_T5[index:index+6] = np.array([moms4_T5, moms4_T5_err, chi2_T5]).transpose() 
            index +=6


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


plt.suptitle(r"Mellin Moments for $\gamma_3\gamma_5, \; P_0=$"+f" %.3f" %(phys_p(a,4)))
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
plt.xlabel(r"$\langle x^2\rangle$")
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
plt.xlabel(r"$\langle x^4\rangle$")
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
    plt.savefig(f"{save_path}/Z5mellin_P04.pdf")
plt.show()


plt.figure()
plt.suptitle(r"Mellin Moments for $\gamma_0\gamma_5$")
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
plt.xlabel(r"$\langle x^2\rangle$")
for i in range(0,Ns):
    plt.text(0.23, i, infos[i])
plt.xlim(0.22, 0.40)
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
plt.xlabel(r"$\langle x^4\rangle$")
for i in range(0,Ns):
    plt.text(0.05, i, infos[i])
plt.xlim(0., 0.8)
plt.axvline(avg4_T5[0], color="black", linestyle="dashed")
plt.axvspan(avg4_T5[0]-avg4_T5[1], avg4_T5[0]+avg4_T5[1], alpha=0.2, color='red')
plt.axvspan(avg4_T5[0]-avg4_T5[1]-avg4_T5[2], avg4_T5[0]+avg4_T5[1]+avg4_T5[2], alpha=0.2, color='blue')
plt.tick_params(
    axis='y',      
    which='both',    
    left="False" ,      
    labelleft=False) 
if save:
    plt.savefig(f"{save_path}/T5mellin_P04.pdf")
plt.show()






# ##### Begin Z5 Chi2 ######
# plt.figure()
# plt.plot(
#     data2_Z5[:,2],
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
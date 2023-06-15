import numpy as np
import matplotlib.pyplot as plt
import os
from functions import read_data
 
init_char = "Z5"
Nt = 128
Ns = 10
a = 2.359
t_start = 4
t_end = 11
fits = {}
save = True


if save == True:
    parent = "0-data/2_state_matrix_results"
    child = f"{init_char}"
    save_path = os.path.join(parent,child)
    os.makedirs(save_path, exist_ok=True)

for i in range(4,10):
    m0_fits = np.load(f"0-data/2_state_matrix_results/{init_char}/Pz{i}/Pz{i}_R.npy")
    fits[f"Pz = {i}"] = m0_fits
bestE0_tmins = [10,9,8,10,8,7] # with a window length of 15
bestE1_tmins = [ 4,3,4, 2,4,4] # with a window length of 10

def phys_p(a, n):
    # takes a as an energy in GeV
    return 2 * np.pi * n * a / 64

def real_state2ratio(t,m0, m1):
    num   = -(((Z0/(2*E0))*m0*(np.exp(-E0*t) + np.exp(-E0*(Nt-t))) 
        + (Z1/(2*E1))*m1*(np.exp(-E1*t) + np.exp(-E1*(Nt-t))))
    )
    denom = ((Z0**2/(2*E0))*(np.exp(-E0*t) + np.exp(-E0*(Nt-t))) 
        + (Z1**2/(2*E1))*(np.exp(-E1*t) + np.exp(-E1*(Nt-t)))
    )     
    y = num/denom
    return np.real(y)

def imag_state2ratio(t,m0, m1):
    num   = (-1j*((Z0/(2*E0))*m0*(np.exp(-E0*t) + np.exp(-E0*(Nt-t))) 
        + (Z1/(2*E1))*m1*(np.exp(-E1*t) + np.exp(-E1*(Nt-t))))
    )
    denom = ((Z0**2/(2*E0))*(np.exp(-E0*t) + np.exp(-E0*(Nt-t))) 
        + (Z1**2/(2*E1))*(np.exp(-E1*t) + np.exp(-E1*(Nt-t)))
    )     
    y = num/denom
    return np.imag(y)


##### Plot for combined z over one Pz #####
save = False
Pz = 4
plt.figure()
for bz in [0,3,4,6,8]:
    t = np.arange(t_start,t_end)
    real_means, imag_means, real_stds, imag_stds = read_data(Ns, Nt, init_char, Pz, bz)
    E0_data = np.load(f"stats/fit_results/window_arrays/E0_fits_Pz{Pz}.npy")
    E1_data = np.load(f"stats/2state_fit_results/window_arrays/E1_fits_Pz{Pz}.npy")
    E0 = E0_data[0,bestE0_tmins[Pz-4] -2 ] 
    Z0 = np.sqrt(2*E0*E0_data[2,bestE0_tmins[Pz-4]-2])
    E1 = E1_data[0,bestE1_tmins[Pz-4]-2]
    Z1 = np.sqrt(2*E1*E1_data[4,bestE1_tmins[Pz-4]-2])
    real_fit = real_state2ratio(np.arange(t_start,t_end,0.01),fits[f"Pz = {Pz}"][1,bz], fits[f"Pz = {Pz}"][3,bz])
    real_fit_err1 = -real_fit + real_state2ratio(np.arange(t_start,t_end,0.01),
                                    fits[f"Pz = {Pz}"][1,bz]+np.sqrt(fits[f"Pz = {Pz}"][2,bz]), 
                                    fits[f"Pz = {Pz}"][3,bz])
    real_fit_err2 = -real_fit + real_state2ratio(np.arange(t_start,t_end,0.01),
                                    fits[f"Pz = {Pz}"][1,bz], 
                                    fits[f"Pz = {Pz}"][3,bz]+np.sqrt(fits[f"Pz = {Pz}"][4,bz]))
    real_fit_err = np.sqrt(real_fit_err1**2 + real_fit_err2**2)
    
    plt.errorbar(t,real_means[t_start:t_end],yerr=real_stds[t_start:t_end], fmt="ro", capsize=3)
    plt.text(4,real_fit[0]+ 0.05*real_fit[0], f"z={bz}")
    plt.plot(np.arange(t_start,t_end,0.01), real_fit, "b" )
    plt.fill_between(np.arange(t_start,t_end,0.01),real_fit+real_fit_err, real_fit-real_fit_err, alpha=0.2, color="blue")
plt.xlabel(r"$t/a$")
plt.ylabel(r"Re $R(t)$")
# plt.ylim(-0.75e-5,1.5e-5)
plt.title(r"$N_z=$" + f" {Pz}")
if save == True:
    plt.savefig(f"{save_path}/Pz{Pz}/multi_z_fits.png")

plt.show()

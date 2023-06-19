import numpy as np
import matplotlib.pyplot as plt
import os
from functions import read_data, phys_p
 
init_char = "T5"
Nt = 128
Ns = 10
a = 2.359
t_start = 4
t_end = 11
dt = 1
save = False

parent = "0-data/2_state_matrix_results"
child = f"{init_char}"
save_path = os.path.join(parent,child)

if save == True:
    os.makedirs(save_path, exist_ok=True)

fits = {}
for i in range(4,10):
    fit = np.load(f"{save_path}/Pz{i}/Pz{i}_R.npy")
    fits[f"Pz = {i}"] = fit


         # Nz = 4,5,6,7,8,9
bestE1_tmins = [4,3,4,2,4,4] # with a window length of 10


def real_state2ratio(t,init_char, m0, m1):
    if init_char == "T5":
        num   = -(((Z0/(2*E0))*m0*(np.exp(-E0*t) - np.exp(-E0*(Nt-t))) 
                + (Z1/(2*E1))*m1*(np.exp(-E1*t) - np.exp(-E1*(Nt-t))))
            )
    else: 
        num   = -(((Z0/(2*E0))*m0*(np.exp(-E0*t) + np.exp(-E0*(Nt-t))) 
              + (Z1/(2*E1))*m1*(np.exp(-E1*t) + np.exp(-E1*(Nt-t))))
            )
    denom = ((Z0**2/(2*E0))*(np.exp(-E0*t) + np.exp(-E0*(Nt-t))) 
            + (Z1**2/(2*E1))*(np.exp(-E1*t) + np.exp(-E1*(Nt-t)))
    )     
    y = num/denom
    return np.real(y)

def imag_state2ratio(t,init_char, m0, m1):
    if init_char == "T5":
        num   = -1j*(((Z0/(2*E0))*m0*(np.exp(-E0*t) - np.exp(-E0*(Nt-t))) 
                + (Z1/(2*E1))*m1*(np.exp(-E1*t) - np.exp(-E1*(Nt-t))))
            )
    else: 
            num   = -1j*(((Z0/(2*E0))*m0*(np.exp(-E0*t) + np.exp(-E0*(Nt-t))) 
                + (Z1/(2*E1))*m1*(np.exp(-E1*t) + np.exp(-E1*(Nt-t))))
            )
    denom = ((Z0**2/(2*E0))*(np.exp(-E0*t) + np.exp(-E0*(Nt-t))) 
            + (Z1**2/(2*E1))*(np.exp(-E1*t) + np.exp(-E1*(Nt-t)))
    )     
    y = num/denom
    return np.imag(y)


##### Plot for combined z over one Pz #####
save = True
Pz = 4
plt.figure()
for bz in [0,2,4, 6,8,]:
    t = np.arange(t_start,t_end)
    real_means, imag_means, real_stds, imag_stds = read_data(Ns, Nt, init_char, Pz, bz)

    E1_data = np.load(f"stats/2state_fit_results/window_arrays/E1_fits_Pz{Pz}.npy")
    E0 = np.sqrt((0.139)**2 + phys_p(a,Pz)**2)/a
    E1 = E1_data[0,bestE1_tmins[Pz-4]-2]
    Z0 = np.sqrt(2*E0*E1_data[2,bestE1_tmins[Pz-4]-2])
    Z1 = np.sqrt(2*E1*E1_data[4,bestE1_tmins[Pz-4]-2])

    real_fit = real_state2ratio(np.arange(t_start,t_end,dt), init_char, fits[f"Pz = {Pz}"][1,bz], fits[f"Pz = {Pz}"][3,bz])
    real_fit_err1 = -real_fit + real_state2ratio(np.arange(t_start,t_end,dt), init_char,
                                    fits[f"Pz = {Pz}"][1,bz]+np.sqrt(fits[f"Pz = {Pz}"][2,bz]), 
                                    fits[f"Pz = {Pz}"][3,bz])
    real_fit_err2 = -real_fit + real_state2ratio(np.arange(t_start,t_end,dt), init_char,
                                    fits[f"Pz = {Pz}"][1,bz], 
                                    fits[f"Pz = {Pz}"][3,bz]+np.sqrt(fits[f"Pz = {Pz}"][4,bz]))
    real_fit_err = np.sqrt(real_fit_err1**2 + real_fit_err2**2)


    imag_fit = imag_state2ratio(np.arange(t_start,t_end,dt), init_char, fits[f"Pz = {Pz}"][1,bz], fits[f"Pz = {Pz}"][3,bz])
    imag_fit_err1 = -imag_fit + imag_state2ratio(np.arange(t_start,t_end,dt), init_char,
                                    fits[f"Pz = {Pz}"][1,bz]+np.sqrt(fits[f"Pz = {Pz}"][2,bz]), 
                                    fits[f"Pz = {Pz}"][3,bz])
    imag_fit_err2 = -imag_fit + imag_state2ratio(np.arange(t_start,t_end,dt), init_char,
                                    fits[f"Pz = {Pz}"][1,bz], 
                                    fits[f"Pz = {Pz}"][3,bz]+np.sqrt(fits[f"Pz = {Pz}"][4,bz]))
    imag_fit_err = np.sqrt(imag_fit_err1**2 + imag_fit_err2**2)
    

    plt.errorbar(t,real_means[t_start:t_end],yerr=real_stds[t_start:t_end], fmt="ro", capsize=3)
    plt.text(4,real_fit[0]+ 0.05*real_fit[0], f"z={bz}")
    plt.plot(np.arange(t_start,t_end,dt), real_fit, "b" )
    plt.fill_between(np.arange(t_start,t_end,dt),real_fit+real_fit_err, real_fit-real_fit_err, alpha=0.2, color="blue")
plt.xlabel(r"$t/a$")
plt.ylabel(r"Re $R(t)$")
# plt.ylim(-0.75e-5,1.5e-5)
plt.title(r"$N_z=$" + f" {Pz}")
if save == True:
    plt.savefig(f"{save_path}/Pz{Pz}/multi_z_fits.png")

plt.show()

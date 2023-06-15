import numpy as np
import matplotlib.pyplot as plt
import os
from functions import read_data, phys_p
 
init_char = "T5"
Nt = 128
Ns = 10
a = 2.359
t_start = 4
t_end = 14
fits = {}
save = True

# For T5, the factor if -i has already been accounted for, so to plot the 
# imaginary part, you actually need to plot things labeled as real and multiply
# by a factor of -1. However, for Z5, this has not been accounted for and 
# you should instead things labeled real for real parts, etc. 

# create directory for saving files
if save == True:
    parent = "0-data/2_state_matrix_results"
    child = f"{init_char}"
    save_path = os.path.join(parent,child)
    os.makedirs(save_path, exist_ok=True)

# load data into a dictionary with keys given by lattice momentum values
for i in range(4,10):
    fit_result = np.load(f"0-data/2_state_matrix_results/{init_char}/Pz{i}/Pz{i}_R.npy")
    fits[f"Pz = {i}"] = fit_result

# best values for E0 and E1 from the plateaus of earlier c2pt analysiss
bestE0_tmins = [10,9,8,10,8,7] # with a window length of 15
bestE1_tmins = [ 4,3,4, 2,4,4] # with a window length of 10

# functions for calculating extrapolated values. Again, for Z5, use things 
# as labelled, but for T5, use things labeled real for imaginary and vice versa
def real_state2ratio(t,m0, m1):
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
    
def imag_state2ratio(t,m0, m1):
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



##### Plot for Matrix elements over multiple z ######
formats = {"4":"bs","5":"ro","6":"gH","7":"m*","8":"kD","9":"y^"}
plt.figure()
for Pz in range(4,8):
    E0_data = np.load(f"stats/fit_results/window_arrays/E0_fits_Pz{Pz}.npy")
    E1_data = np.load(f"stats/2state_fit_results/window_arrays/E1_fits_Pz{Pz}.npy")
    E0 = E0_data[0,bestE0_tmins[Pz-4]] 
    Z0 = np.sqrt(2*E0*E0_data[2,bestE0_tmins[Pz-4]])
    E1 = E1_data[0,4]
    Z1 = np.sqrt(E1_data[4,4])
    prefactor = {"Z5": phys_p(a,Pz), "T5": a*E0 }

    real_fit = real_state2ratio(100,fits[f"Pz = {i}"][1,:], fits[f"Pz = {i}"][3,:])
    imag_fit = imag_state2ratio(100,fits[f"Pz = {i}"][6,:], fits[f"Pz = {i}"][8,:])

    real_fit_err =np.abs(real_state2ratio(100,
                                    fits[f"Pz = {i}"][1,:]+np.sqrt(fits[f"Pz = {i}"][2,:]), 
                                    fits[f"Pz = {i}"][3,:]+np.sqrt(fits[f"Pz = {i}"][4,:])) \
                 - real_fit)
    imag_fit_err =np.abs(imag_state2ratio(100,
                                    fits[f"Pz = {i}"][6,:]+np.sqrt(fits[f"Pz = {i}"][7,:]), 
                                    fits[f"Pz = {i}"][8,:]+np.sqrt(fits[f"Pz = {i}"][9,:])) \
                 - imag_fit)
    # plt.errorbar(np.arange(0,13),
    #              Z0*real_fit[:13]/prefactor[init_char],
    #              yerr=Z0*real_fit_err[:13]/prefactor[init_char], 
    #              fmt=formats[f"{Pz}"], 
    #              label=f"Pz = {Pz}",
    #              capsize=4)
    plt.errorbar(np.arange(0,13),
                 -Z0*imag_fit[:13]/prefactor[init_char],
                 yerr=Z0*imag_fit_err[:13]/prefactor[init_char], 
                 fmt=formats[f"{Pz}"], 
                 label=f"Pz = {Pz}",
                 capsize=4)
plt.legend()
plt.xlabel("$z/a$")
plt.ylabel(r"Re $h^B(z,P_3)$")
plt.title(f"Extrapolated Matrix Elements for Various Momenta; {init_char}")
if save == True:
    plt.savefig(f"{save_path}/real_extrapolated_R.png")
plt.show()

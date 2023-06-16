import numpy as np
import matplotlib.pyplot as plt
import os
from functions import read_data, phys_p

init_char = "Z5"
Nt = 128
Ns = 25
a = 2.359
fits = {}
save = True

# For T5, the factor if -i has already been accounted for, so to plot the
# imaginary part, you actually need to plot things labeled as real and multiply
# by a factor of -1. However, for Z5, this has not been accounted for and
# you should instead things labeled real for real parts, etc.

# create directory for saving files
if save == True:
    parent = "k0/2_state_matrix_results"
    child = f"{init_char}"
    save_path = os.path.join(parent,child)
    os.makedirs(save_path, exist_ok=True)

# load data into a dictionary with keys given by lattice momentum values
for i in range(0,4):
    fit_result = np.load(f"k0/2_state_matrix_results/{init_char}/Pz{i}/Pz{i}_R.npy")
    fits[f"Pz = {i}"] = fit_result

# best values for E0 and E1 from the plateaus of earlier c2pt analysiss
bestE1_tmins = [4,4,4,2] # with a window length of 10

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
formats = {"0":"bs","1":"ro","2":"gH","3":"m*","4":"kD","5":"y^"}
plt.figure()
for Pz in range(0,4):
    E1_data = np.load(f"k0/2state_fit_results/window_arrays/E1_fits_Pz{Pz}.npy")
    E0 = np.sqrt((0.139)**2 + phys_p(a,Pz)**2)/a
    E1 = E1_data[0,bestE1_tmins[Pz]-2]
    Z0 = np.sqrt(2*E0*E1_data[2,bestE1_tmins[Pz]-2])
    Z1 = np.sqrt(2*E1*E1_data[4,bestE1_tmins[Pz]-2])


    prefactor = {"Z5": phys_p(a,Pz), "T5": a*E0 }
    real_fit = real_state2ratio(20,fits[f"Pz = {Pz}"][1,:], fits[f"Pz = {Pz}"][3,:])
    imag_fit = imag_state2ratio(20,fits[f"Pz = {Pz}"][6,:], fits[f"Pz = {Pz}"][8,:])
    real_fit_err =np.abs(real_state2ratio(20,
                                    fits[f"Pz = {Pz}"][1,:]+np.sqrt(fits[f"Pz = {Pz}"][2,:]),
                                    fits[f"Pz = {Pz}"][3,:]+np.sqrt(fits[f"Pz = {Pz}"][4,:])) \
                 - real_fit)
    imag_fit_err =np.abs(imag_state2ratio(20,
                                    fits[f"Pz = {Pz}"][6,:]+np.sqrt(fits[f"Pz = {Pz}"][7,:]),
                                    fits[f"Pz = {Pz}"][8,:]+np.sqrt(fits[f"Pz = {Pz}"][9,:])) \
                 - imag_fit)
    plt.errorbar(np.arange(0,13),
                 Z0*real_fit[:13]/prefactor[init_char],
                 yerr=np.sqrt(a)*Z0*real_fit_err[:13]/prefactor[init_char],
                 fmt=formats[f"{Pz}"],
                 label=f"Pz = {Pz}",
                 capsize=4)
    # plt.errorbar(np.arange(0,13),
    #              a**2*Z0*imag_fit[:13]/prefactor[init_char],
    #              yerr=a**2*Z0*imag_fit_err[:13]/prefactor[init_char],
    #              fmt=formats[f"{Pz}"],
    #              label=f"Pz = {Pz}",
    #              capsize=4)
plt.legend()
plt.xlabel("$z/a$")
plt.ylabel(r"Re $h^B(z,P_3)$")
plt.title(f"Extrapolated Matrix Elements for Various Momenta; {init_char}")
if save == True:
    plt.savefig(f"{save_path}/real_extrapolated_R.png")
plt.show()

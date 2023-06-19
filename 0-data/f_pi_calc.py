import numpy as np
import matplotlib.pyplot as plt
import os
from functions import phys_p

 
Nt = 128
Ns = 10
a = 2.359
Z5_fits = {}
T5_fits = {}
save = True

# For T5, the factor if -i has already been accounted for, so to plot the 
# imaginary part, you actually need to plot things labeled as real and multiply
# by a factor of -1. However, for Z5, this has not been accounted for and 
# you should instead things labeled real for real parts, etc. 

# Create save path
if save == True:
    save_path = "0-data/2_state_matrix_results"
    os.makedirs(save_path, exist_ok=True)

# load data into a dictionary with keys given by lattice momentum values
for i in range(4,10):
    Z5_fit = np.load(f"0-data/2_state_matrix_results/Z5/Pz{i}/Pz{i}_R.npy")
    T5_fit = np.load(f"0-data/2_state_matrix_results/T5/Pz{i}/Pz{i}_R.npy")
    Z5_fits[f"Pz = {i}"] = Z5_fit
    T5_fits[f"Pz = {i}"] = T5_fit

# best values for E0 and E1 from the plateaus of earlier c2pt analysis
bestE0_tmins = [10,9,8,7,7,7] # with a window length of 15
bestE1_tmins = [ 4,3,4,2,4,4] # with a window length of 10


# functions for calculating extrapolated values. Again, for Z5, use things 
# as labelled, but for T5, use things labeled real for imaginary and vice versa
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


##### Plot for Matrix elements over multiple z ######
plt.figure()
for i in ["Z5", "T5"]:
    for Pz in range(4,10):
        E1_data = np.load(f"stats/2state_fit_results/window_arrays/E1_fits_Pz{Pz}.npy")
        E0 = np.sqrt(phys_p(a, Pz)**2 + (0.139)**2)/a
        E1 = E1_data[0,bestE1_tmins[Pz-4]-2]
        Z0 = np.sqrt(2*E0*E1_data[2,bestE1_tmins[Pz-4]-2])
        Z1 = np.sqrt(2*E1*E1_data[4,bestE1_tmins[Pz-4]-2])

        prefactor = {"Z5": phys_p(a,Pz), "T5": np.sqrt(phys_p(a,Pz)**2 + (0.139)**2)}

        if i == "Z5":
            real_fit = real_state2ratio(100,i,Z5_fits[f"Pz = {Pz}"][1,0], Z5_fits[f"Pz = {Pz}"][3,0])
            imag_fit = imag_state2ratio(100,i,Z5_fits[f"Pz = {Pz}"][6,0], Z5_fits[f"Pz = {Pz}"][8,0])
            real_fit_err1 =np.abs(real_state2ratio(100,i,
                                Z5_fits[f"Pz = {Pz}"][1,0]+np.sqrt(Z5_fits[f"Pz = {Pz}"][2,0]), 
                                Z5_fits[f"Pz = {Pz}"][3,0]) 
                                 - real_fit)
            real_fit_err2 =np.abs(real_state2ratio(100,i,
                                Z5_fits[f"Pz = {Pz}"][1,0], 
                                Z5_fits[f"Pz = {Pz}"][3,0]+np.sqrt(Z5_fits[f"Pz = {Pz}"][4,0])) 
                                 - real_fit)
            real_fit_err = np.sqrt(real_fit_err1**2 + real_fit_err2**2)

            imag_fit_err1 =np.abs(imag_state2ratio(100,i,
                                Z5_fits[f"Pz = {Pz}"][6,0], 
                                Z5_fits[f"Pz = {Pz}"][8,0]+np.sqrt(Z5_fits[f"Pz = {Pz}"][9,0])) 
                                 - imag_fit)
            imag_fit_err2 =np.abs(imag_state2ratio(100,i,
                                Z5_fits[f"Pz = {Pz}"][6,0]+np.sqrt(Z5_fits[f"Pz = {Pz}"][7,0]), 
                                Z5_fits[f"Pz = {Pz}"][8,0]) 
                                 - imag_fit)
            imag_fit_err = np.sqrt(imag_fit_err1**2 + imag_fit_err2**2)
            line1 = plt.errorbar(Pz,
                        a*Z0*real_fit/prefactor[i],
                        yerr=a*Z0*real_fit_err/prefactor[i], 
                        fmt="bo", 
                        label=r"$\gamma_3\gamma_5$",
                        capsize=4)

        else:
            real_fit = real_state2ratio(100,i,T5_fits[f"Pz = {Pz}"][1,0], T5_fits[f"Pz = {Pz}"][3,0])
            imag_fit = imag_state2ratio(100,i,T5_fits[f"Pz = {Pz}"][6,0], T5_fits[f"Pz = {Pz}"][8,0])

            real_fit_err1 =np.abs(real_state2ratio(100,i,
                                T5_fits[f"Pz = {Pz}"][1,0], 
                                T5_fits[f"Pz = {Pz}"][3,0]+np.sqrt(T5_fits[f"Pz = {Pz}"][4,0])) 
                                 - real_fit)
            real_fit_err2 =np.abs(real_state2ratio(100,i,
                                T5_fits[f"Pz = {Pz}"][1,0]+np.sqrt(T5_fits[f"Pz = {Pz}"][2,0]), 
                                T5_fits[f"Pz = {Pz}"][3,0]) 
                                 - real_fit)
            real_fit_err = np.sqrt(real_fit_err1**2 + real_fit_err2**2)

            imag_fit_err1 =np.abs(imag_state2ratio(100,i,
                                T5_fits[f"Pz = {Pz}"][6,0], 
                                T5_fits[f"Pz = {Pz}"][8,0]+np.sqrt(T5_fits[f"Pz = {Pz}"][9,0])) 
                                 - imag_fit)
            imag_fit_err2 =np.abs(imag_state2ratio(100,i,
                                T5_fits[f"Pz = {Pz}"][6,0]+np.sqrt(T5_fits[f"Pz = {Pz}"][7,0]), 
                                T5_fits[f"Pz = {Pz}"][8,0]) 
                                 - imag_fit)
            imag_fit_err = np.sqrt(imag_fit_err1**2 + imag_fit_err2**2)
            
            line2 = plt.errorbar(Pz,
                        -a*Z0*imag_fit/prefactor[i],
                        yerr=a*Z0*imag_fit_err/prefactor[i], 
                        fmt="rs", 
                        label=f"Pz = {Pz}",
                        capsize=4)

plt.xlabel("$P_z$")
plt.ylabel(r"$f_\pi/Z_A$ ")
plt.title(r"Calculation of $f_{\pi}$")
plt.legend((line1, line2), (r"$\gamma_3\gamma_5$", r"$\gamma_0\gamma_5$"))
if save == True:
    plt.savefig(f"{save_path}/real_f_pi.png")
plt.show()

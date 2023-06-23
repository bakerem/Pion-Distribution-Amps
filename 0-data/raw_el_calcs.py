# Author Ethan Baker, ANL/Haverford College
import numpy as np
import matplotlib.pyplot as plt
import os
from functions import phys_p

init_char = "Z5"
Nt = 128
Ns = 10
a = 2.359
t_start = 4
t_end = 14
fits = {}
save = True

# create directory for saving files
if save == True:
    parent = "final_results/2_state_matrix_results"
    child = f"{init_char}"
    save_path = os.path.join(parent,child)
    os.makedirs(save_path, exist_ok=True)

# load data into a dictionary with keys given by lattice momentum values
for Pz in range(0,4):
    fit = np.load(f"final_results/2_state_matrix_results/{init_char}/Pz{Pz}/Pz{Pz}_R.npy")
    fits[f"Pz = {Pz}"] = fit

# best values for E0 and E1 from the plateaus of earlier c2pt analysiss
bestE1_tmins = [3,3,3,3]    # with a window length of 10


def real_state2ratio(t, m0, m1):
    """
    functions for calculating real part of the ratio formed from
    DA/C2pt. In the case of T5, there is anti-symmetry, so a minus sign is 
    needed between terms in the numerator
    """
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

def imag_state2ratio(t, m0, m1):
    """
    functions for calculating imaginary part of the ratio formed from
    DA/C2pt. In the case of T5, there is anti-symmetry, so a minus sign is 
    needed between terms in the numerator
    """
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




for Pz in range(0,4):
    # load in data for calculations with function
    E1_data = np.load(f"final_results/two_state_fits/Pz{Pz}/E1_fits_Pz{Pz}.npy")

    E0 = np.sqrt((0.139)**2 + phys_p(a,Pz)**2)/a
    E1 = E1_data[0,bestE1_tmins[Pz]-2]
    Z0 = np.sqrt(2*E0*E1_data[2,bestE1_tmins[Pz]-2])
    Z1 = np.sqrt(2*E1*E1_data[4,bestE1_tmins[Pz]-2])

    # definition of prefactor used to convert from ratio to raw physical 
    # quantities
    prefactor = {"Z5": a**2*Z0/phys_p(a,Pz), "T5": a**2*Z0/(a*E0) }

    # calculation of extrapolated value of ratio for the matrix element and
    # the errors propogated through errors in fitting parameters. 
    real_fit = prefactor[init_char]*real_state2ratio(100,fits[f"Pz = {Pz}"][1,:], fits[f"Pz = {Pz}"][3,:])
    imag_fit = prefactor[init_char]*imag_state2ratio(100, fits[f"Pz = {Pz}"][6,:], fits[f"Pz = {Pz}"][8,:])
    real_fit_err1 = np.abs(prefactor[init_char]*real_state2ratio(100,
                        fits[f"Pz = {Pz}"][1,:]+np.sqrt(fits[f"Pz = {Pz}"][2,:]), 
                        fits[f"Pz = {Pz}"][3,:])
                            - real_fit)
    real_fit_err2 =np.abs(prefactor[init_char]*real_state2ratio(100,
                        fits[f"Pz = {Pz}"][1,:], 
                        fits[f"Pz = {Pz}"][3,:]+np.sqrt(fits[f"Pz = {Pz}"][4,:]))
                            - real_fit) 
    real_fit_err = np.sqrt(real_fit_err1**2 + real_fit_err2**2)
    print(real_fit[0])
    imag_fit_err1 =np.abs(prefactor[init_char]*imag_state2ratio(100,
                        fits[f"Pz = {Pz}"][6,:], 
                        fits[f"Pz = {Pz}"][8,:]+np.sqrt(fits[f"Pz = {Pz}"][9,:]))
                            - imag_fit)
    imag_fit_err2 =np.abs(prefactor[init_char]*imag_state2ratio(100,
                        fits[f"Pz = {Pz}"][6,:]+np.sqrt(fits[f"Pz = {Pz}"][7,:]), 
                        fits[f"Pz = {Pz}"][8,:])
                            - imag_fit)
    imag_fit_err = np.sqrt(imag_fit_err1**2 + imag_fit_err2**2)

    # saving routine
    if save:
        np.save(f"{save_path}/Pz{Pz}/real_data.npy", real_fit)
        np.save(f"{save_path}/Pz{Pz}/real_errs.npy", real_fit_err)
        np.save(f"{save_path}/Pz{Pz}/imag_data.npy", imag_fit)
        np.save(f"{save_path}/Pz{Pz}/imag_errs.npy", imag_fit_err)
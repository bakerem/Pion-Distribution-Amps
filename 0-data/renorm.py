import numpy as np
import matplotlib.pyplot as plt
import os
from functions import phys_p

import this
Nt = 128
Ns = 10
a = 2.359
bz_max = 13
P0 = 4
init_char = "Z5"
fits = {}
save = False
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
    fit = np.load(f"0-data/2_state_matrix_results/Z5/Pz{i}/Pz{i}_R.npy")
    T5_fit = np.load(f"0-data/2_state_matrix_results/T5/Pz{i}/Pz{i}_R.npy")
    fits[f"Pz = {i}"] = fit

# best values for E0 and E1 from the plateaus of earlier c2pt analysis
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

# Calculating stuff for P0
P0prefactor = {"Z5": phys_p(a,P0), "T5": np.sqrt(phys_p(a,P0)**2 + (0.139)**2)}
E1_data = np.load(f"stats/2state_fit_results/window_arrays/E1_fits_Pz{P0}.npy")
E0 = np.sqrt(phys_p(a, P0)**2 + (0.139)**2)/a
E1 = E1_data[0,bestE1_tmins[P0-4]-2]
Z0 = np.sqrt(2*E0*E1_data[2,bestE1_tmins[P0-4]-2])
Z1 = np.sqrt(2*E1*E1_data[4,bestE1_tmins[P0-4]-2])

P0_reals = a**2*Z0*real_state2ratio(100,
                                    init_char,
                                    fits[f"Pz = {P0}"][1,:], 
                                    fits[f"Pz = {P0}"][3,:])/P0prefactor[init_char]
P0_imags = a**2*Z0*imag_state2ratio(100,
                                    init_char,
                                    fits[f"Pz = {P0}"][6,:], 
                                    fits[f"Pz = {P0}"][8,:])/P0prefactor[init_char]

P0_rerr1 =np.abs(a**2*Z0*real_state2ratio(100,init_char,
                        fits[f"Pz = {P0}"][1,:]+np.sqrt(fits[f"Pz = {P0}"][2,:]), 
                        fits[f"Pz = {P0}"][3,:])/P0prefactor[init_char]
                            - P0_reals)
P0_rerr2 =np.abs(a**2*Z0*real_state2ratio(100,init_char,
                    fits[f"Pz = {P0}"][1,:], 
                    fits[f"Pz = {P0}"][3,:]+np.sqrt(fits[f"Pz = {P0}"][4,:]))/P0prefactor[init_char]
                        - P0_reals)
P0_rerr   = np.sqrt(P0_rerr1**2 + P0_rerr2**2)

P0_ierr1 =np.abs(a**2*Z0*imag_state2ratio(100,init_char,
                    fits[f"Pz = {P0}"][6,:], 
                    fits[f"Pz = {P0}"][8,:]+np.sqrt(fits[f"Pz = {P0}"][9,:]))/P0prefactor[init_char]
                        - P0_imags)
P0_ierr2 = np.abs(a**2*Z0*imag_state2ratio(100,init_char,
                    fits[f"Pz = {P0}"][6,:]+np.sqrt(fits[f"Pz = {P0}"][7,:]), 
                    fits[f"Pz = {P0}"][8,:])/P0prefactor[init_char]
                        - P0_imags)
P0_ierr  = np.sqrt(P0_ierr1*2 + P0_ierr2**2)

##### Plot for Matrix elements over multiple z ######
plt.figure()
formats = {"4":"bs","5":"ro","6":"gH","7":"m*","8":"cD","9":"y^"}
for Pz in range(5,8):
    E1_data = np.load(f"stats/2state_fit_results/window_arrays/E1_fits_Pz{Pz}.npy")
    E0 = np.sqrt(phys_p(a, Pz)**2 + (0.139)**2)/a
    E1 = E1_data[0,bestE1_tmins[Pz-4]-2]
    Z0 = np.sqrt(2*E0*E1_data[2,bestE1_tmins[Pz-4]-2])
    Z1 = np.sqrt(2*E1*E1_data[4,bestE1_tmins[Pz-4]-2])
    prefactor = {"Z5": phys_p(a,Pz), "T5": np.sqrt(phys_p(a,Pz)**2 + (0.139)**2)}
    


    real_fit = a**2*Z0*real_state2ratio(100,
                                     init_char,
                                     fits[f"Pz = {Pz}"][1,:], 
                                     fits[f"Pz = {Pz}"][3,:])/prefactor[init_char]
    imag_fit = a**2*Z0*imag_state2ratio(100,
                                     init_char,
                                     fits[f"Pz = {Pz}"][6,:], 
                                     fits[f"Pz = {Pz}"][8,:])/prefactor[init_char]

    real_fit_err1 =np.abs(a**2*Z0*real_state2ratio(100,init_char,
                        fits[f"Pz = {Pz}"][1,:]+np.sqrt(fits[f"Pz = {Pz}"][2,:]), 
                        fits[f"Pz = {Pz}"][3,:])/prefactor[init_char]
                            - real_fit)
    real_fit_err2 =np.abs(a**2*Z0*real_state2ratio(100,init_char,
                        fits[f"Pz = {Pz}"][1,:], 
                        fits[f"Pz = {Pz}"][3,:]+np.sqrt(fits[f"Pz = {Pz}"][4,:]))/prefactor[init_char]
                            - real_fit)
    real_fit_err = np.sqrt(real_fit_err1**2 + real_fit_err2**2)

    imag_fit_err1 =np.abs(a**2*Z0*imag_state2ratio(100,init_char,
                        fits[f"Pz = {Pz}"][6,:], 
                        fits[f"Pz = {Pz}"][8,:]+np.sqrt(fits[f"Pz = {Pz}"][9,:]))/prefactor[init_char]
                            - imag_fit)
    imag_fit_err2 =np.abs(a**2*Z0*imag_state2ratio(100,init_char,
                        fits[f"Pz = {Pz}"][6,:]+np.sqrt(fits[f"Pz = {Pz}"][7,:]), 
                        fits[f"Pz = {Pz}"][8,:])/prefactor[init_char]
                            - imag_fit)
    imag_fit_err    = np.sqrt(imag_fit_err1**2 + imag_fit_err2**2)

    matrix_els = (((real_fit+1j*imag_fit)/(P0_reals + 1j*P0_imags))
                    *((P0_reals[0]+1j*P0_imags[0])/(real_fit[0] + 1j*imag_fit[0]))
                    *(np.exp(-1j*np.arange(0,33)*(phys_p(a,Pz)-phys_p(a,P0))/2))
                 )
    m_err1 = (((real_fit+real_fit_err+1j*imag_fit)/(P0_reals + 1j*P0_imags))
                    *((P0_reals[0]+1j*P0_imags[0])/(real_fit[0] + real_fit_err[0] + 1j*imag_fit[0]))
                    *(np.exp(-1j*np.arange(0,33)*(phys_p(a,Pz)-phys_p(a,P0))/2))
                 ) - matrix_els
    m_err2 = (((real_fit+1j*(imag_fit+imag_fit_err))/(P0_reals + 1j*P0_imags))
                    *((P0_reals[0]+1j*P0_imags[0])/(real_fit[0] + 1j*(imag_fit[0]+imag_fit_err[0])))
                    *(np.exp(-1j*np.arange(0,33)*(phys_p(a,Pz)-phys_p(a,P0))/2))
                 ) - matrix_els
    m_err3 = (((real_fit+1j*imag_fit)/(P0_reals + P0_rerr + 1j*P0_imags))
                    *((P0_reals[0]+P0_rerr[0]+1j*P0_imags[0])/(real_fit[0] + 1j*imag_fit[0]))
                    *(np.exp(-1j*np.arange(0,33)*(phys_p(a,Pz)-phys_p(a,P0))/2))
                 ) - matrix_els
    m_err4 = (((real_fit+1j*imag_fit)/(P0_reals + 1j*(P0_imags+P0_ierr)))
                    *((P0_reals[0]+1j*(P0_imags[0]+P0_ierr[0]))/(real_fit[0] + 1j*imag_fit[0]))
                    *(np.exp(-1j*np.arange(0,33)*(phys_p(a,Pz)-phys_p(a,P0))/2))
                 ) - matrix_els
    m_err = np.sqrt(m_err1**2 + m_err2**2 + m_err3**2 + m_err4**2)
    plt.errorbar(np.arange(0,bz_max),
                 np.real(matrix_els[:bz_max]),
                #  yerr = m_err[:bz_max],
                 fmt=formats[str(Pz)],
                 capsize=3,
                 label=f"Pz = {Pz}")
    plt.text(bz_max-0.8, np.real(matrix_els[bz_max-1]),f"Pz = {Pz}")

    
plt.legend()
plt.xlabel(r"$z_3/a$")
plt.ylabel(r"Re $\mathcal{M}$")
plt.xlim(-1,15)
plt.title("Renormalized Matrix Elements")
if save == True:
    plt.savefig(f"{save_path}/renorm_multi_p.png")
plt.show()


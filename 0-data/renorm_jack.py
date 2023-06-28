# Author: Ethan Baker ANL/Haverford College

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import os
from functions import phys_p, read_data

Nt = 128
Ns = 55
a = 2.359
bz_max = 30
P0 = 1
init_char = "Z5"
fits = {}
save = True
real = False
t_range = np.arange(0,128)
lower_lim = 3
upper_lim = 18

# Create save path
save_path = f"final_results_eps10/2_state_matrix_results_jack/{init_char}"
os.makedirs(save_path, exist_ok=True)



##### Plot for Matrix elements over multiple z ######
plt.figure()
formats = {"1":"bs","2":"ro","3":"gH","4":"m*","4":"cD","9":"y^"}
bestE1_tmins = [0,2,1,0]    # with a window length of 10

# calculation of P0 stuff
P0_real_fit_js = np.zeros((33,Ns))
P0_imag_fit_js = np.zeros((33,Ns))
for bz in range(0,33):
    (real_ratio_means,
        imag_ratio_means, 
        real_ratio_stds, 
        imag_ratio_stds,
        real_samples,
        real_sample_errs,
        imag_samples,
        imag_sample_errs) = read_data(Ns, Nt, init_char, P0, bz)
    
    # load in data from previous fits 
    # E0_data is from dispersion relation and E1_data is from 2 state fit
    E1_data = np.load(f"final_results/two_state_fits/Pz{P0}/samples.npy")
    


    """
    actual function for fitting
    spectral decomposition of DA correlator is "num" and spec. decomp.
    of 2pt correlator is denom
    For T5, the numerator in both should be a sinh function, for Z5, it
    should be  cosh function
    """
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
    
    # curve fitting and calculating chi squared

    for s in range(Ns):
        E0 = np.sqrt((0.139)**2 + phys_p(a,P0)**2)/a
        E1 = E1_data[s,2]
        Z0 = np.sqrt(2*E0*E1_data[s,0])
        Z1 = np.sqrt(2*E1*E1_data[s,1])
        real_popt_j, real_pcov_j = curve_fit(
                                        real_state2ratio, 
                                        t_range[lower_lim:upper_lim], 
                                        real_samples[s,lower_lim:upper_lim], 
                                        sigma=real_sample_errs[s,lower_lim:upper_lim],
                                        maxfev=2000,
                                        )

        imag_popt_j, imag_pcov_j = curve_fit(
                                            imag_state2ratio, 
                                            t_range[lower_lim:upper_lim], 
                                            imag_samples[s,lower_lim:upper_lim], 
                                            sigma=imag_sample_errs[s,lower_lim:upper_lim],
                                            maxfev=2000,
                                            )

        prefactor = {"Z5": a**2*Z0/phys_p(a,P0), "T5": a**2*Z0/(a*E0) }

        # calculation of extrapolated value of ratio for the matrix element and
        # the errors propogated through errors in fitting parameters. 
        P0_real_fit_js[bz,s] = prefactor[init_char]*real_state2ratio(100, *real_popt_j)
        P0_imag_fit_js[bz,s] = prefactor[init_char]*imag_state2ratio(100, *imag_popt_j)



real_raw_ratios = np.load(f"{save_path}/real_raw_samples.npy")
print(real_raw_ratios.shape)
imag_raw_ratios = np.load(f"{save_path}/imag_raw_samples.npy")
real_matrix_el_js = np.zeros((5,33,Ns,))
imag_matrix_el_js = np.zeros((5,33,Ns,))

for Pz in range(0,4):
    for s in range(Ns):
        matrix_el_j = (((real_raw_ratios[Pz,:,s]+1j*imag_raw_ratios[Pz,:,s])/(P0_real_fit_js[:,s] + 1j*P0_imag_fit_js[:,s]))
                        *((P0_real_fit_js[0,s]+1j*P0_imag_fit_js[0,s])/(real_raw_ratios[Pz,0,s] + 1j*imag_raw_ratios[Pz,0,s]))
                        *(np.exp(-1j*np.arange(0,33)/a*(phys_p(a,Pz)-phys_p(a,P0))/2))
                    )
        real_matrix_el_js[Pz,:,s] = np.real(matrix_el_j)
        imag_matrix_el_js[Pz,:,s] = np.imag(matrix_el_j)

real_matrix_el = np.average(real_matrix_el_js, axis=2)
real_matrix_err = np.sqrt(Ns-1)*np.std(real_matrix_el_js, axis=2)
imag_matrix_el = np.average(imag_matrix_el_js, axis=2)
imag_matrix_err = np.sqrt(Ns-1)*np.std(imag_matrix_el_js, axis=2)


if save:
    np.save(f"{save_path}/real_matrix_el.npy", real_matrix_el)
    np.save(f"{save_path}/real_matrix_err.npy", real_matrix_err)
    np.save(f"{save_path}/imag_matrix_el.npy", imag_matrix_el)
    np.save(f"{save_path}/imag_matrix_err.npy", imag_matrix_err)

    np.save(f"{save_path}/real_raw_ratios.npy", np.average(real_raw_ratios, axis=2))
    np.save(f"{save_path}/real_raw_ratios_err.npy", np.sqrt(Ns-1)*np.std(real_raw_ratios, axis=2))
    np.save(f"{save_path}/imag_raw_ratios.npy", np.average(imag_raw_ratios, axis=2))
    np.save(f"{save_path}/imag_raw_ratios_err.npy", np.sqrt(Ns-1)*np.std(imag_raw_ratios, axis=2))



for Pz in range(2,4):
    if real:
        plt.errorbar(phys_p(a,Pz)*np.arange(0,bz_max)/a,
                    real_matrix_el[Pz,:bz_max],
                    yerr = real_matrix_err[Pz,:bz_max],
                    fmt=formats[str(Pz)],
                    capsize=3,
                    label=f"Pz = {Pz}")
        # plt.text(bz_max-0.8, np.real(matrix_els[bz_max-1]),f"Pz = {Pz}")
    else:
        plt.errorbar(phys_p(a,Pz)*np.arange(0,bz_max)/a,
                    imag_matrix_el[Pz,:bz_max],
                    yerr = abs(imag_matrix_err[Pz,:bz_max]),
                    fmt=formats[str(Pz)],
                    capsize=3,
                    label=f"Pz = {Pz}")
        # plt.text(bz_max-0.8, np.imag(matrix_els[bz_max-1]),f"Pz = {Pz}")

if real:
    plt.legend()
    plt.xlabel(r"$P_zz_3$")
    plt.ylabel(r"Re $\mathcal{M}$")
    plt.xlim(0,3)
    plt.ylim(0.4,1.1)
    plt.title(f"Renormalized Matrix Elements: {init_char}")
    plt.text(0.25, 0.75, r"$P_0$ " + "= %.2f GeV" %phys_p(a,P0))
    if save:
        plt.savefig(f"{save_path}/real_renorm_multi_p.png")
    plt.show()

else:
    plt.legend()
    plt.xlabel(r"$P_zz_3$")
    plt.ylabel(r"Imag $\mathcal{M}$")
    plt.text(0.25, 0.08, r"$P_0$ " + "= %.2f GeV" %phys_p(a,P0))
    plt.title(f"Renormalized Matrix Elements: {init_char}")
    plt.xlim(0,3)
    plt.ylim(-0.03, 0.16)
    plt.hlines(0,0,3,"gray", "--")
    if save:
        plt.savefig(f"{save_path}/imag_renorm_multi_p.png")
    plt.show()





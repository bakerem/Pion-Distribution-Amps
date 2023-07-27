import numpy as np
import matplotlib.pyplot as plt
import scienceplots

plt.style.use('science')


"""
Ethan Baker, Haverford College/ANL
script for producing plots from fitting over multiple momenta in order
to test that the fitted results conform to the dispersion relation. BestE0s is 
the best t_min for each momentum value and should be inputted directly based on
inspection of the results of "one_state_fits.py"
"""
trials = 10
a = 1/2.359 # lattice spacing in GeV^-1

E0s     = np.zeros((trials,))
E0_errs = np.zeros((trials,))
A0s     = np.zeros((trials,))
A0_errs = np.zeros((trials,))
chi2s   = np.zeros((trials,))
bestE0s = [19,12,9,13,4,3,4,6,2,5] 


for i in range(0,trials):
    E0s[i]     = np.load(f"final_results/one_state_fits/Pz{i}/E0_fits_Pz{i}.npy")[0,bestE0s[i]]
    E0_errs[i] = np.load(f"final_results/one_state_fits/Pz{i}/E0_fits_Pz{i}.npy")[1, bestE0s[i]]

def phys_p(a,n):
    # takes a as an energy in GeV^-1
    # 64 is the lattice spatial dimension and may need modification
    return 2*np.pi*n/(64*a)

phys_ps = phys_p(a,np.arange(0,trials))


# Produces a plot of the E0 values and the dispersion relation in physical units
plt.figure()
plt.errorbar(phys_ps,E0s/a, 
             yerr=E0_errs/a,
             capsize=5, 
             fmt="bs", 
             markerfacecolor="none",
             label="From Fit")
plt.plot(phys_p(a,np.arange(0,trials, 0.05)),
         np.sqrt((E0s[0]/a)**2+phys_p(a,np.arange(0,trials, 0.05))**2), 
         "r",
         label="Predicted")
plt.ylabel(r"$E_0$ (GeV)")
plt.xlabel(r"$P_3$ (GeV)")
plt.legend()
plt.savefig("final_results/one_state_fits/1_state_multi_p.pdf")
plt.show()


import numpy as np
import matplotlib.pyplot as plt

"""
Ethan Baker, Haverford College/ANL
script for producing plots from fitting over multiple momenta in order
to test that the fitted results conform to the dispersion relation
"""
trials = 4
E0s     = np.zeros((trials,))
E0_errs = np.zeros((trials,))
A0s     = np.zeros((trials,))
A0_errs = np.zeros((trials,))
chi2s   = np.zeros((trials,))
bestE0s = [19,12,9,13]

Pz = 3
chi2s_single_Pz = np.load(f"final_results/one_state_fits/Pz{Pz}/E0_fits_Pz{Pz}.npy")[6]
for i in range(0,len(chi2s_single_Pz)):
    print(i, chi2s_single_Pz[i])
for i in range(0,trials):
    E0s[i]     = np.load(f"final_results/one_state_fits/Pz{i}/E0_fits_Pz{i}.npy")[0,bestE0s[i]]
    E0_errs[i] = np.load(f"final_results/one_state_fits/Pz{i}/E0_fits_Pz{i}.npy")[1, bestE0s[i]]

def phys_p(a,n):
    # takes a as an energy in GeV
    return 2*np.pi*n*a/64

phys_ps = phys_p(2.359,np.arange(0,trials))
print(phys_p(2.359, 1))

plt.figure()
plt.errorbar(phys_ps,(2.359*E0s), 
             yerr=2.359*E0_errs,
             capsize=5, 
             fmt="bs", 
             label="From Fit")
plt.plot(phys_p(2.359,np.arange(0,trials, 0.05)),
         np.sqrt((E0s[0]*2.359)**2+phys_p(2.359,np.arange(0,trials, 0.05))**2), 
         "r",
         label="Predicted")
plt.ylabel("Energy (GeV)")
plt.xlabel("$P_z$ (GeV)")
plt.legend()
plt.savefig("final_results/one_state_fits/1_state_multi_p.pdf")
plt.show()

# plt.figure(np.arange(0,trials), np.sqrt(2*E0s*chi2s), ".")
# plt.plot()

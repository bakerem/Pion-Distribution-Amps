import numpy as np
import matplotlib.pyplot as plt

"""
Ethan Baker, Haverford College/ANL
script for producing plots from fitting over multiple momenta in order
to test that the fitted results conform to the dispersion relation
"""
trials = 10
E0s     = np.zeros((trials,))
E0_errs = np.zeros((trials,))
A0s     = np.zeros((trials,))
A0_errs = np.zeros((trials,))
chi2s   = np.zeros((trials,))

for i in range(0,trials):
    E0s[i]     = np.load(f"stats/fit_results/E0data_1state_Pz{i}.npy")[0]
    E0_errs[i] = np.sqrt(np.load(f"stats/fit_results/E0data_1state_Pz{i}.npy")[1])
    A0s[i]     = np.load(f"stats/fit_results/E0data_1state_Pz{i}.npy")[2]
    A0_errs[i] = np.sqrt(np.load(f"stats/fit_results/E0data_1state_Pz{i}.npy")[3])
    chi2s[i]   = np.load(f"stats/fit_results/E0data_1state_Pz{i}.npy")[4]

print(np.load(f"stats/fit_results/E0data_1state_Pz{8}.npy")[3])
def phys_p(a,n):
    # takes a as an energy in GeV
    return 2*np.pi*n*a/64

phys_ps = phys_p(2.359,np.arange(0,trials))
print((np.sqrt(E0s[0]*2.359)**2+phys_p(2.359,np.arange(0,10))**2)/2.359)


plt.figure()
plt.errorbar(phys_ps,(2.359*E0s), 
             yerr=2.359*E0_errs,
             capsize=5, 
             fmt=".", 
             label="From Fit")
plt.plot(phys_p(2.359,np.arange(0,trials, 0.05)),
         np.sqrt((E0s[0]*2.359)**2+phys_p(2.359,np.arange(0,trials, 0.05))**2), 
         label="Predicted")
plt.ylabel("Energy (GeV)")
plt.xlabel("$P_z$ (GeV)")
plt.legend()
plt.savefig("stats/fit_results/1_state_multi_p.pdf")
plt.show()

plt.figure()
plt.errorbar(phys_ps, chi2s, "s")
plt.title("Chi-Square Values for Various Pz")
plt.xlabel("Pz (GeV)")
plt.ylabel("$\chi^2$")
plt.savefig("stats/fit_results/chi2_over_p.png")
plt.show()

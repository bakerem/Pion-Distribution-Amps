import numpy as np
import matplotlib.pyplot as plt

trials = 10
E0s = np.zeros((trials,))
errs = np.zeros((trials,))
chi2s = np.zeros((trials,))

for i in range(0,trials):
    E0s[i] = np.load(f"stats/fit_results/E0data_1state_Pz{i}.npy")[0]
    errs[i] = np.sqrt(np.load(f"stats/fit_results/E0data_1state_Pz{i}.npy")[1])
    chi2s[i] = np.load(f"stats/fit_results/E0data_1state_Pz{i}.npy")[2]

print(np.load(f"stats/fit_results/E0data_1state_Pz{8}.npy")[3])
def phys_p(a,n):
    # takes a as an energy in GeV
    return 2*np.pi*n*a/64

phys_ps = phys_p(2.359,np.arange(0,trials))
print((np.sqrt(E0s[0]*2.359)**2+phys_p(2.359,np.arange(0,10))**2)/2.359)


plt.figure()
plt.errorbar(phys_ps,(2.359*E0s), 
             yerr=errs,
             capsize=5, 
             fmt=".", 
             label="From Fit")
plt.plot(phys_p(2.359,np.arange(0,trials, 0.05)),
         np.sqrt((E0s[0]*2.359)**2+phys_p(2.359,np.arange(0,trials, 0.05))**2), 
         label="Predicted")
plt.ylabel("Energy (GeV)")
plt.xlabel("$P_z$ (GeV)")
plt.legend()
plt.savefig("stats/fit_results/1_state_multi_p.png")
plt.show()

plt.figure()
plt.plot(phys_ps, chi2s, "s")
plt.title("Chi-Square Values for Various Pz")
plt.xlabel("Pz (GeV)")
plt.ylabel("$\chi^2$")
plt.savefig("stats/fit_results/chi2_over_p.png")
plt.show()

import numpy as np
import matplotlib.pyplot as plt

trials = 10
E0s = np.zeros((trials,))
E0_errs = np.zeros((trials,))
E1s = np.zeros((trials,))
E1_errs = np.zeros((trials,))
chi2s = np.zeros((trials,))

for i in range(0,trials):
    E1s[i] = np.load(f"stats/2state_fit_results/E0data_1state_Pz{i}.npy")[0]
    E1_errs[i] = np.sqrt(np.load(f"stats/2state_fit_results/E0data_1state_Pz{i}.npy")[1])
    chi2s[i] = np.load(f"stats/2state_fit_results/E0data_1state_Pz{i}.npy")[2]


def phys_p(a,n):
    # takes a as an energy in GeV
    return 2*np.pi*n*a/64

phys_ps = phys_p(2.359,np.arange(0,trials))
print(phys_p(2.359, 6))
print(np.sqrt((E0s[0]*2.359)**2+phys_p(2.359,np.arange(0,10))**2)/2.359)


fig, ax = plt.subplots()
plt.errorbar(phys_ps,(2.359*E1s), 
             yerr=E1_errs*2.359,
             capsize=5, 
             fmt=".", 
             label="From Fit")
t1 = phys_p(2.359,np.arange(0,trials, 0.05))
y1 = np.sqrt(((E1s[0] - E1_errs[0])*2.359)**2+phys_p(2.359,np.arange(0,trials, 0.05))**2)
y2 = np.sqrt(((E1s[0] + E1_errs[0])*2.359)**2+phys_p(2.359,np.arange(0,trials, 0.05))**2)
plt.plot(phys_p(2.359,np.arange(0,trials, 0.05)),
         np.sqrt((E1s[0]*2.359)**2+phys_p(2.359,np.arange(0,trials, 0.05))**2), 
         label="Predicted")
plt.fill_between(t1, y1, y2, alpha=0.2, color="orange")
plt.ylabel("Energy (GeV)")
plt.xlabel("$P_z$ (GeV)")
plt.legend()
# plt.savefig("stats/2state_fit_results/2_state_multi_p.png")
plt.show()

plt.figure()
plt.plot(phys_ps, chi2s, "s")
plt.title("Chi-Square Values for Various Pz")
plt.xlabel("Pz (GeV)")
plt.ylabel("$\chi^2$")
# plt.savefig("stats/2state_fit_results/chi2_over_p.png")
plt.show()
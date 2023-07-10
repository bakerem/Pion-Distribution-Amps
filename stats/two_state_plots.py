import numpy as np
import matplotlib.pyplot as plt
import scienceplots

plt.style.use('science')
# plt.rcParams['figure.figsize'] = [3.3, 4.8]


"""
Ethan Baker, Haverford College/ANL
script for producing plots from fitting over multiple momenta in order
to test that the fitted results conform to the dispersion relation
"""

trials = 10

save_path = f"final_results/two_state_fits"

E0s = np.zeros((trials,))
E0_errs = np.zeros((trials,))
E1s = np.zeros((trials,))
E1_errs = np.zeros((trials,))
chi2s = np.zeros((trials,))
best_E1_tmins = [0,1,1,0,0,0,1,1,1,1]


for i in range(0,trials):
    E1s[i] = np.load(f"final_results/two_state_fits/Pz{i}/E1_fits_Pz{i}.npy")[0, best_E1_tmins[i]]
    E1_errs[i] = np.load(f"final_results/two_state_fits/Pz{i}/E1_fits_Pz{i}.npy")[1, best_E1_tmins[i]]
    chi2s[i] = np.load(f"final_results/two_state_fits/Pz{i}/E1_fits_Pz{i}.npy")[6, best_E1_tmins[i]]


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
             fmt="bs", 
             markerfacecolor="None",
             label="From Fit")
t1 = phys_p(2.359,np.arange(0,trials, 0.05))
y1 = np.sqrt(((E1s[0] - E1_errs[0])*2.359)**2+phys_p(2.359,np.arange(0,trials, 0.05))**2)
y2 = np.sqrt(((E1s[0] + E1_errs[0])*2.359)**2+phys_p(2.359,np.arange(0,trials, 0.05))**2)
plt.plot(phys_p(2.359,np.arange(0,trials, 0.05)),
         np.sqrt((E1s[0]*2.359)**2+phys_p(2.359,np.arange(0,trials, 0.05))**2), 
         "r",
         label="Predicted")
plt.fill_between(t1, y1, y2, alpha=0.2, color="red")
plt.ylabel(r"$E_1$ (GeV)")
plt.xlabel(r"$P_3$ (GeV)")
plt.legend()
plt.savefig(f"{save_path}/2_state_multi_p.pdf")
plt.show()
E0s = np.sqrt(phys_ps**2 + 0.139**2)
plt.figure()
plt.plot(phys_ps, np.sqrt(2*E0s*chi2s), "s")
plt.title("Chi-Square Values for Various Pz")
fig_width, fig_height = plt.gcf().get_size_inches()
print(fig_width, fig_height)
plt.xlabel("Pz (GeV)")
plt.ylabel("$\chi^2$")
plt.savefig(f"{save_path}/chi2_over_p.png")
plt.show()
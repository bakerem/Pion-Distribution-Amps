import numpy as np
import matplotlib.pyplot as plt
import scienceplots

plt.style.use('science')
# plt.rcParams['figure.figsize'] = [3.3, 4.8]


"""
Ethan Baker, Haverford College/ANL
script for producing plots from fitting over multiple momenta in order
to test that the fitted results conform to the dispersion relation. BestE1s is 
the best t_min for each momentum value and should be inputted directly based on
inspection of the results of "two_state_fits.py"
"""

trials = 10
a = 1/2.359 # GeV^-1


save_path = f"final_results/two_state_fits"

E0s = np.zeros((trials,))
E0_errs = np.zeros((trials,))
E1s = np.zeros((trials,))
E1_errs = np.zeros((trials,))
chi2s = np.zeros((trials,))
best_E1s = [2,3,3,2,2,2,3,3,3,3]


for i in range(0,trials):
    # the minus 2 accounts for the starting t_min value of 2 in the previous fits
    E1s[i] = np.load(f"final_results/two_state_fits/Pz{i}/E1_fits_Pz{i}.npy")[0, best_E1s[i]-2]
    E1_errs[i] = np.load(f"final_results/two_state_fits/Pz{i}/E1_fits_Pz{i}.npy")[1, best_E1s[i]-2]
    chi2s[i] = np.load(f"final_results/two_state_fits/Pz{i}/E1_fits_Pz{i}.npy")[6, best_E1s[i]-2]


def phys_p(a,n):
    # takes a as an energy in GeV^-1
    # 64 is the lattice spatial dimension and may need modification

    return 2*np.pi*n/(64*a)

phys_ps = phys_p(a,np.arange(0,trials))

fig, ax = plt.subplots()
plt.errorbar(phys_ps,E1s/a, 
             yerr=E1_errs/a,
             capsize=5, 
             fmt="bs", 
             markerfacecolor="None",
             label="From Fit")
t1 = phys_p(a,np.arange(0,trials, 0.05))
y1 = np.sqrt(((E1s[0] - E1_errs[0])/a)**2+phys_p(a,np.arange(0,trials, 0.05))**2)
y2 = np.sqrt(((E1s[0] + E1_errs[0])/a)**2+phys_p(a,np.arange(0,trials, 0.05))**2)
plt.plot(phys_p(a,np.arange(0,trials, 0.05)),
         np.sqrt((E1s[0]/a)**2+phys_p(a,np.arange(0,trials, 0.05))**2), 
         "r",
         label="Predicted")
plt.fill_between(t1, y1, y2, alpha=0.2, color="red")
plt.ylabel(r"$E_1$ (GeV)")
plt.xlabel(r"$P_3$ (GeV)")
plt.legend()
# plt.savefig(f"{save_path}/2_state_multi_p.pdf")
plt.show()

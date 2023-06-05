import numpy as np
import matplotlib.pyplot as plt


one_state = np.load("stats/E0data_1state.npy")
E0_1state = one_state[::2]
err_1state = np.sqrt(one_state[1::2])
print(one_state)
two_state = np.load("stats/E0data_2state.npy")
E0_2state = two_state[::2]
err_2state = np.sqrt(two_state[1::2])

def phys_p(a,n):
    # takes a as an energy in GeV
    return 2*np.pi*n*a/64
phys_ps = phys_p(2.359,np.array([0,1,2,3,4,5,6]))
# print(E0_1state.shape)

print(2.359*E0_1state)
plt.figure()
plt.errorbar(phys_ps,2.359*E0_1state, yerr=err_1state, capsize=5, fmt=".")
plt.plot(phys_ps,np.sqrt(0.14**2+phys_ps**2), ".")
plt.ylabel("Energy (GeV)")
plt.xlabel("$n_z$")
plt.savefig("1_state_multi_p.pdf")
plt.show()

# plt.figure()
# plt.errorbar(phys_ps[:-1],2.359*E0_2state[:-1], yerr=err_2state[:-1], capsize=5, fmt=".")
# plt.plot(phys_ps[:-1],np.sqrt(0.14**2+phys_ps[:-1]**2))
# plt.show()

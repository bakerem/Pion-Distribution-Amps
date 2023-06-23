# Author Ethan Baker, ANL/Haverford College

import numpy as np
import matplotlib.pyplot as plt
import os


Nt = 128
a = 2.359
mu = 2/a
P0 = 4
Pz = 6
N_max = 2
bz = 4
bz_min = 2
bz_max = 10
Pz = 6
Pz_min = 5
Pz_max = 8

save_path = "0-data/renorm_results"
os.makedirs(save_path, exist_ok=True)

moms2 = np.load(f"{save_path}/gegen_moms2.npy")
errs = np.load(f"{save_path}/renorm_results/gegen_moms2_err.npy")

plt.errorbar(
    np.arange(bz_min, bz_max),
    moms2,
    errs,
    capsize=3,
    fmt="bs",
)
plt.ylim(-0.5, 0.3)
plt.title(r"$z$ Dependence of $\langle x^2\rangle$")
plt.xlabel(r"$z/a$")
plt.ylabel(r"$\langle x^2\rangle$")
plt.savefig(f"{save_path}/m_ope_multi_z.png")
plt.show()
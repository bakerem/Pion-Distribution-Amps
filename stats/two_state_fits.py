import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, fsolve
import os

"""
Ethan Baker, Haverford College/ANL
Uses a two-state three parameter fit for fitting C2pt data. E0 is fixed
from the dispersion relation. Pz gives the momentum of the data in 
simulation units, p0 is an initial guess for fitting and bounds are bounds
for the fit. 
"""

Nt = 128
Pz = 0
save = False
Ns = 55
p0 = (2e6, 1e7, 0.49)
bounds = ([1e2,1e2,0], [1e12, 1e12, 5])

lower_t = 2
upper_t = 18
window = 10


save_path = f"final_results/two_state_fits/Pz{Pz}"
os.makedirs(save_path, exist_ok=True)

# read in data as pandas dataframe
columns = ["t"] + [str(i) for i in range(0, Ns)]
if Pz < 4:
    df = pd.read_csv(
        f"0-data/c2pt_cfgs/64IGSRC_W40_k0.ama.c2pt.PX0PY0PZ{Pz}.real.cfg.csv",
        names=columns,
        dtype=np.float64)
else:
    df = pd.read_csv(
        f"stats/c2pt-data/64IGSRC_W40_k6.ama.c2pt.PX0PY0PZ{Pz}.real.cfg.csv",
        names=columns,
        dtype=np.float64)

df_data = df.drop(axis="columns", labels=["t"])

# jackknife error for use in fitting and error analysis
samples = np.zeros((Ns, Nt))
for i in range(0, Ns):
    sample = df_data.drop(axis="columns", labels=[str(i)]).mean(axis=1)
    samples[i, :] = sample
means = np.average(samples, axis=0)
std_devs = np.sqrt(Ns - 1) * np.std(samples, axis=0)
df["mean"] = means
df["std dev"] = std_devs

# calculation of covariance matrix 

cov = np.zeros((Nt, Nt))
for i in range(0,Nt):
    for j in range(Nt):
        cov[i,j] = np.average((df_data.iloc[i] - means[i])*
                              (df_data.iloc[j] - means[j]))
cov = cov/(Ns-1)


# function to convert nz to physical Pz
def phys_p(a, n):
    # takes a as an energy in GeV
    return 2 * np.pi * n * a / 64



def perform_fit(lower_lim, upper_lim, plot=False):
    """
    Performs a 3 parameter, 2 state fit with fixed values of E0 from the dispersion relation.
    lower_lim, upper_lim define the fitting window
    kwarg plot defines whether or not to produce plots. 
    """

    # dispersion relation to fix E0
    E0s = np.sqrt(0.13957**2 + phys_p(2.359, np.arange(0, 10)) ** 2) / 2.359

    # objective function for fitting
    def obj_func(t, A0, A1, E1):
        y = (
            A0 * np.exp(-t * E0s[Pz])
            + A0 * np.exp(-(Nt - t) * E0s[Pz])
            + A1 * np.exp(-t * E1)
            + A1 * np.exp(-(Nt - t) * E1)
        )
        return y

    # curve fitting and calculating chi squared
    popt, pcov = curve_fit(
        obj_func,
        df["t"].iloc[lower_lim:upper_lim],
        df["mean"].iloc[lower_lim:upper_lim],
        # sigma=df["std dev"].iloc[lower_lim:upper_lim],
        sigma=cov[lower_lim:upper_lim,lower_lim:upper_lim],
        p0=p0,
        bounds=bounds,
        maxfev=2000,
    )

    chi2 = np.sum(
        (df["mean"].iloc[lower_lim:upper_lim]
            - obj_func(df["t"].iloc[lower_lim:upper_lim], *popt)
        ) ** 2
        / (df["std dev"].iloc[lower_lim:upper_lim]) ** 2 
        / (upper_lim - lower_lim - len(popt))
    )



    # append lists with data for plotting over multiple window sizes


    #######PLOTS#######
    if plot:
        # raw data plots w/ error bars
        plt.errorbar(
            df["t"].iloc[lower_lim:upper_lim],
            df["mean"].iloc[lower_lim:upper_lim],
            yerr=df["std dev"].iloc[lower_lim:upper_lim],
            capsize=3,
            fmt=".",
        )
        plt.plot(
            df["t"].iloc[lower_lim:upper_lim],
            obj_func(df["t"].iloc[lower_lim:upper_lim], *popt),
        )
        # plt.yscale("log")
        tab_cols = ("Value", "Error")
        tab_rows = ("A0", "A1", "E1", "$\chi^2$")
        cells = [
            ["%.2e" %popt[0], "%.2e" % np.sqrt(pcov[0, 0])],
            ["%.2e" % popt[1], "%.2e" % np.sqrt(pcov[1, 1])],
            ["%.2e" %(2.359*popt[2]), "%.2e" %(2.359*np.sqrt(pcov[2, 2]))],
            ["%.3f" % chi2, "n/a"],
        ]
        plt.table(
            cellText=cells,
            rowLabels=tab_rows,
            colLabels=tab_cols,
            loc="upper center",
            colWidths=[0.2, 0.2],
        )
        plt.xlabel("$n_t$")
        plt.ylabel("$C(n_t)$")
        plt.title(f"Nz = {Pz}")
        if save:
            plt.savefig(f"{save_path}/Pz{Pz}_corrfunc.png")
        plt.show()

    return chi2, popt[2], np.sqrt(pcov[2, 2]), popt[0], np.sqrt(pcov[0,0]), popt[1], np.sqrt(pcov[1,1])



fit_results = np.zeros((7,upper_t-lower_t-window))
for i in range(lower_t, upper_t-window):
    results = perform_fit(i,upper_t, plot=False)
    fit_results[0,i-lower_t] = results[1] # save E1
    fit_results[1,i-lower_t] = results[2] # save E1 err
    fit_results[2,i-lower_t] = results[3] # save A0
    fit_results[3,i-lower_t] = results[4] # save A0 err
    fit_results[4,i-lower_t] = results[5] # save A1
    fit_results[5,i-lower_t] = results[6] # save A1 err
    fit_results[6,i-lower_t] = results[0] # save chi2

    print(i, results[0], 2.359*results[1], 2.359*results[2])

plt.figure()
plt.errorbar(np.arange(lower_t, upper_t-window), 2.359*np.array(fit_results[0]), yerr=(2.359*np.array(fit_results[1])), fmt="rs", capsize=4)
plt.xlabel(r"$t_{min}/a$")
plt.ylabel(r"$E_1(P_z)$ (Gev)")
plt.ylim(0,4)
plt.text(2,3.5, "Pz = %.2fGeV" %phys_p(2.359, Pz), fontfamily="sans-serif", fontsize="large", fontstyle="normal")
plt.title(r"Fitted $E_1$ from [$t_{min}/a$, " + f"{upper_t}]")
if save:
    plt.savefig(f"{save_path}/Pz{Pz}window_length{window}.png")
    np.save(f"{save_path}/E1_fits_Pz{Pz}.npy", fit_results)
plt.show()



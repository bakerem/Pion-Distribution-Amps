import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, fsolve


"""
Ethan Baker, Haverford College/ANL
Uses a two-state three parameter fit for fitting C2pt data. E0 is fixed
from the dispersion relation. Pz gives the momentum of the data in 
simulation units, p0 is an initial guess for fitting and bounds are bounds
for the fit. 
"""

Nt = 128
Pz = 1
save = False
p0 = (1e8, 1e8, 0.57)
# p0 = None
bounds = ([1e5,1e5,0], [1e9, 1e9, 2])

# create list of columns for df
if Pz < 4:
    Ns = 29
else:
    Ns = 30

# read in data as pandas dataframe
columns = [str(i) for i in range(0, Ns)]
columns = ["t"] + columns
if Pz < 4:
    df = pd.read_csv(
        f"stats/c2pt-data/64IGSRC_W40_k0.ama.c2pt.PX0PY0PZ{Pz}.real.cfg.csv",
        names=columns,
        dtype=np.float64,
    )
else:
    df = pd.read_csv(
        f"stats/c2pt-data/64IGSRC_W40_k6.ama.c2pt.PX0PY0PZ{Pz}.real.cfg.csv",
        names=columns,
        dtype=np.float64,
    )

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

# empty lists for plotting over different window sizes
E1_fits = []
E1_errs = []
chi2_list = []


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
        sigma=df["std dev"].iloc[lower_lim:upper_lim],
        absolute_sigma=True,
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

    # data saving routine
    if save == True:
        new_data = np.array((popt[2], pcov[2, 2], chi2))
        np.save(f"stats/2state_fit_results/E0data_1state_Pz{Pz}.npy", new_data)

    # append lists with data for plotting over multiple window sizes
    E1_fits.append(popt[2])
    E1_errs.append(np.sqrt(pcov[2, 2]))
    chi2_list.append(chi2)


    #######PLOTS#######
    if plot == True:
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
        if save == True:
            plt.savefig(f"stats/2state_fit_results/Pz{Pz}_corrfunc.png")
        plt.show()

    return chi2, popt[2], np.sqrt(pcov[2, 2])



lower_t = 2
upper_t = 7

plt.figure()
for i in range(lower_t, upper_t):
    chi2, E1_fit, E1_err = perform_fit(i,i+10, plot=False)
    print(chi2,2.359*E1_fit, 2.359*E1_err)
    plt.errorbar(i, 2.359*E1_fit, 2.359*E1_err, fmt="rs", capsize=4)
    plt.xlabel(r"$a t_{min}$")
    plt.ylabel(r"$E_1(P_z)$ (Gev)")
    plt.ylim(0,4)
    plt.text(3, 
             3.5, 
             "Pz = %.2fGeV" %phys_p(2.359, Pz), 
             fontfamily="sans-serif", 
             fontsize="large", 
             fontstyle="normal")
    plt.title(r"Fitted $E_1$ from [$t_{min}a$, $t_{min}a + 10$]")
if save == True:
    plt.savefig(f"stats/2state_fit_results/window_length_Pz{Pz}.png")
plt.show()
perform_fit(2, 12, plot=True)

# plt.figure()
# plt.plot(np.arange(lower_t, upper_t+1), E1_fits)
# plt.xlabel(r"$t_{min}$")
# plt.ylabel(r"$E_1$")
# plt.title(r"$E_1$ for various $t_{min}")
# plt.show()

# plt.figure()
# plt.plot(np.arange(lower_t, upper_t+1), chi2_list)
# plt.xlabel(r"$t_{min}$")
# plt.ylabel(r"$\chi^2$")
# plt.title(r"$\chi^2$ for various $t_{min}")
# plt.show()

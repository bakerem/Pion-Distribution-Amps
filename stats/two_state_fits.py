from operator import truediv
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
for the fit. p0 especially may need modification for different momenta
How to use:
1) Run loop at the bottom of the plot to determine fit for range of t_mins
   for one momentum
2) Once best t_min has been determined, run perform_fit function with kwarg 
   savebest=True. Note the best t_min used as this is used when producing a plot
   in two_state_plots.py. To speed this up, comment out the loop that ranges
   over t_min.
3) Repeat for different momenta.

The results of this fit are used in "two_state_plots.py" in this directory
to produce a plot of the dispersion relation. That relies on a list called 
bestE1s, which is put together manually by finding the t_min value that
produces the fit with the lowest chi-squared during the fit. These are printed
when the script is run to allow for easy computation. 
"""

Nt = 128
a = 1/2.359 # GeV^-1
Pz = 4

save = False
Ns = 55
p0 = (2e7, 8e6, 1.) # initial gueses for A0, A1, E1
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
        f"0-data/c2pt_cfgs/64IGSRC_W40_k6.ama.c2pt.PX0PY0PZ{Pz}.real.cfg.csv",
        names=columns,
        dtype=np.float64)

df_data = df.drop(axis="columns", labels=["t"])

# jackknife error for use in fitting and error analysis
samples = np.zeros((Ns, Nt))
samples_err = np.zeros((Ns,Nt))
for i in range(0, Ns):
    sample = df_data.drop(axis="columns", labels=[str(i)]).mean(axis=1)
    samples[i, :] = sample
    samples_err[i,:] = df_data.drop(axis="columns", labels=[str(i)]).std(axis=1)
means = np.average(samples, axis=0)
std_devs = np.sqrt(Ns - 1) * np.std(samples, axis=0)
df["mean"] = means
df["std dev"] = std_devs


# function to convert nz to physical Pz
def phys_p(a, n):
    # takes a as an energy in GeV
    # 64 is the spatial lattice length and may need modification
    return 2 * np.pi * n / (a * 64)



def perform_fit(lower_lim, upper_lim, plot=False, savebest=False):
    """
    Performs a 3 parameter, 2 state fit with fixed values of E0 from the dispersion relation.
    lower_lim, upper_lim define the fitting window
    kwarg plot defines whether or not to produce plots. 
    """

    # dispersion relation to fix E0, in lattice units
    E0s = np.sqrt(0.13957**2 + phys_p(a, np.arange(0, 10)) ** 2) * a

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
    popt_js = np.zeros((Ns,3))
    popt_errs = np.zeros((Ns,3))
    pcov = np.zeros((3,3))
    for s in range(Ns):
        data = df_data.drop(axis="columns", labels=[str(s)])
        cov = np.zeros((upper_lim-lower_lim, upper_lim-lower_lim))
        for i in range(lower_lim,upper_lim):
            for j in range(lower_lim,upper_lim):
                cov[i-lower_lim,j-lower_lim] = np.average((data.iloc[i-lower_lim] - samples[s,i-lower_lim])*
                                    (data.iloc[j-lower_lim] - samples[s,j-lower_lim]))
        cov = cov/(Ns-1)
        popt_j, pcov_j = curve_fit(
                            obj_func,
                            df["t"].iloc[lower_lim:upper_lim],
                            samples[s,lower_lim:upper_lim],
                            sigma=cov,
                            p0=p0,
                            bounds=bounds,
                            maxfev=2000,)
        popt_js[s,:] = popt_j
        popt_errs[s,0] = pcov_j[0,0]
        popt_errs[s,1] = pcov_j[1,1]
        popt_errs[s,2] = pcov_j[2,2]
    

    popt = np.average(popt_js,axis=0)
    pcov[0,0] = (Ns-1)*(np.std(popt_js[:,0]))**2
    pcov[1,1] = (Ns-1)*(np.std(popt_js[:,1]))**2
    pcov[2,2] = (Ns-1)*(np.std(popt_js[:,2]))**2


    chi2 = np.sum(
        (df["mean"].iloc[lower_lim:upper_lim]
            - obj_func(df["t"].iloc[lower_lim:upper_lim], *popt)
        ) ** 2
        / (df["std dev"].iloc[lower_lim:upper_lim]) ** 2 
        / (upper_lim - lower_lim - len(popt))
    )




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
            ["%.2e" %(popt[2]/a), "%.2e" %(np.sqrt(pcov[2, 2])/a)],
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


    if savebest:
        np.save(f"{save_path}/samples.npy", popt_js)
        np.save(f"{save_path}/sample_errs.npy", popt_errs)

    return chi2, popt[2], np.sqrt(pcov[2, 2]), popt[0], np.sqrt(pcov[0,0]), popt[1], np.sqrt(pcov[1,1])



fit_results = np.zeros((7,upper_t-lower_t-window))
for i in range(lower_t, upper_t-window):
    results = perform_fit(i,upper_t, plot=True)
    fit_results[0,i-lower_t] = results[1] # save E1
    fit_results[1,i-lower_t] = results[2] # save E1 err
    fit_results[2,i-lower_t] = results[3] # save A0
    fit_results[3,i-lower_t] = results[4] # save A0 err
    fit_results[4,i-lower_t] = results[5] # save A1
    fit_results[5,i-lower_t] = results[6] # save A1 err
    fit_results[6,i-lower_t] = results[0] # save chi2

    print(i, results[0], results[1]/a, results[2]/a)

plt.figure()
plt.errorbar(np.arange(lower_t, upper_t-window), np.array(fit_results[0])/a, yerr=(np.array(fit_results[1])/a), fmt="rs", capsize=4)
plt.xlabel(r"$t_{min}/a$")
plt.ylabel(r"$E_1(P_z)$ (Gev)")
plt.ylim(0,4)
plt.text(2,3.5, "Pz = %.2fGeV" %phys_p(a, Pz), fontfamily="sans-serif", fontsize="large", fontstyle="normal")
plt.title(r"Fitted $E_1$ from [$t_{min}/a$, " + f"{upper_t}]")
if save:
    plt.savefig(f"{save_path}/Pz{Pz}window_length{window}.pdf")
    np.save(f"{save_path}/E1_fits_Pz{Pz}.npy", fit_results)
plt.show()


## Uncomment the below line to save the best fit, change best_t_min and 
#  best_t_max to be the appropriate values determined from ranging t_min. 
# perform_fit(best_t_min, best_t_max,savebest=True) 


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, fsolve


Nt = 128
Pz = 0
save = True

# create list of columns for df
if Pz < 4:
    Ns = 29
else:
    Ns = 30

columns = ["t"]
for i in range(0,Ns):
    columns.append(f"{i}")

# read in data as pandas dataframe 
if Pz < 4:
    df = pd.read_csv(f"/home/bakerem/ANL/stats/c2pt-data/64IGSRC_W40_k0.ama.c2pt.PX0PY0PZ{Pz}.real.cfg.csv",names=columns, dtype=np.float64)
else:
    df = pd.read_csv(f"/home/bakerem/ANL/stats/c2pt-data/64IGSRC_W40_k6.ama.c2pt.PX0PY0PZ{Pz}.real.cfg.csv",names=columns, dtype=np.float64)


df_data = df.drop(axis="columns", labels=["t"])

# jackknife error
samples = np.zeros((Ns,Nt))
for i in range(0,Ns):
    sample = df_data.drop(axis="columns", labels=[str(i)]).mean(axis=1)
    samples[i,:] = sample
means = np.average(samples, axis=0)
std_devs = np.sqrt(Ns-1)*np.std(samples, axis = 0)
df["mean"] = means
df["std dev"] = std_devs

E0_fits    = []
E0_errs    = []
E0_chi2    = []
A0_fits    = []
m_eff_fits = []
m_eff_errs = []
m_eff_chi2 = []
# objective function for fitting
def perform_fit(lower_lim, upper_lim, plot_raw = False, plot_meff = False):
    def obj_func(t,A0, E0):
        y = A0*np.exp(-t*E0) + A0*np.exp(-(Nt-t)*E0)  
        return y

    # curve fitting and calculating chi squared
    p0 = (10**7, 0.06)
    bounds = ([0,0.05],[np.inf,0.2])
    popt, pcov = curve_fit(
                    obj_func, df["t"].iloc[lower_lim:upper_lim], 
                    df["mean"].iloc[lower_lim:upper_lim], 
                    sigma=df["std dev"].iloc[lower_lim:upper_lim],
                    p0=p0, 
                    bounds=bounds)
    chi2 = np.sum(
            (df["mean"].iloc[lower_lim:upper_lim]-
            obj_func(df["t"].iloc[lower_lim:upper_lim],*popt))**2/
            (df["std dev"].iloc[lower_lim:upper_lim])**2)

    A0_fits.append(popt[0])
    E0_fits.append(popt[1])
    E0_errs.append(np.sqrt(pcov[1,1]))
    E0_chi2.append(chi2)
    # #data saving routine
    if save == True:
        new_data = np.array((popt[1],pcov[1,1],chi2, popt[0]))
        # data = np.concatenate((prev_data, new_data))
        np.save(f"stats/fit_results/E0data_1state_Pz{Pz}.npy", new_data)


    # solving equation for effective mass
    def solvingfunc(m,t,Nt,ratio_i):
        result = np.cosh(m*(t-0.5*Nt))/np.cosh(m*(t+1-0.5*Nt)) - ratio_i
        return result

    # intialize empty arrays
    roots = np.zeros((Nt,))
    roots_err = np.zeros((Nt,))


    # for effective mass calculations
    # jack-knife calculations for error
    roots_all_samples = np.zeros((Ns,Nt))
    for i in range(0,Ns):
        for t in range(0,126): #solve for m_eff at each time step
            root = fsolve(solvingfunc, 
                        x0=popt[1], 
                        args=(t,Nt,samples[i,t]/samples[i,t+1]))
            roots_all_samples[i,t] = root
    roots = np.average(roots_all_samples, axis=0)
    roots_err = np.sqrt(Ns-1)*np.std(roots_all_samples, axis=0)


    def obj_func_meff(t, m_eff):
        return m_eff

    popt_meff, pcov_meff = curve_fit(
                    obj_func_meff, 
                    df["t"].iloc[lower_lim:upper_lim], 
                    roots[lower_lim:upper_lim],
                    sigma=roots_err[lower_lim:upper_lim])

    chi2_meff = np.sum((roots[lower_lim:upper_lim]-obj_func_meff(df["t"].iloc[lower_lim:upper_lim],*popt_meff))**2 / roots_err[lower_lim:upper_lim]**2)
    m_eff_fits.append(popt_meff)
    m_eff_errs.append(np.sqrt(pcov_meff[0,0]))
    m_eff_chi2.append(chi2_meff)

    #######PLOTS#######
    if plot_raw == True:
    # raw data plots w/ error bars
        plt.errorbar(df["t"].iloc[lower_lim:upper_lim], 
                    df["mean"].iloc[lower_lim:upper_lim],
                    yerr=df["std dev"].iloc[lower_lim:upper_lim], 
                    capsize=3, 
                    fmt=".")
        plt.plot(df["t"].iloc[lower_lim:upper_lim], obj_func(df["t"].iloc[lower_lim:upper_lim],*popt))
        # plt.yscale("log")
        tab_cols = ("Value", "Error")
        tab_rows = ("A0", "E0", "$\chi^2$")
        cells = [["%.2e" %popt[0], "%.2e" %np.sqrt(pcov[0,0])]\
                ,["%.2e" %popt[1],"%.2e" %np.sqrt(pcov[1,1])]\
                ,["%.3f" %chi2, "n/a"]]
        plt.table(cellText=cells, rowLabels=tab_rows, colLabels=tab_cols, loc="upper center", colWidths=[0.2,0.2])
        plt.xlabel("$n_t$")
        plt.ylabel("$C(n_t)$")
        plt.title(f"Nz = {Pz}")
        plt.savefig(f"stats/fit_results/Pz{Pz}_corrfunc.png")
        plt.show()


    if plot_meff == True:
        # effective mass plots
        plt.figure()
        plt.errorbar(df["t"].iloc[lower_lim:upper_lim],
                    roots[lower_lim:upper_lim], 
                    yerr= np.abs(roots_err[lower_lim:upper_lim]),
                    capsize=3,
                    fmt="s")

        plt.plot(df["t"].iloc[lower_lim:upper_lim], 
                np.ones((upper_lim-lower_lim,))*obj_func_meff(df["t"].iloc[lower_lim:upper_lim], 
                *popt_meff))

        plt.xlabel("$n_t$")
        plt.ylabel("$m_eff$")
        plt.title(f"Nz = {Pz}")
        tab_cols = ("Value", "Error")
        tab_rows = ("$m_{eff}$", "$\chi^2$")
        cells = [["%.2e" %popt_meff[0], "%.2e" %np.sqrt(pcov_meff[0,0])]\
                ,["%.3f" %chi2_meff, "n/a"]]
        plt.table(cellText=cells, rowLabels=tab_rows, colLabels=tab_cols, loc="upper center", colWidths=[0.2,0.2])
        plt.savefig(f"stats/fit_results/Pz{Pz}_meff.png")
        plt.show()
    return chi2


lower_t = 3
upper_t = 20
lower_window = 10
upper_window = 20
best_chi2 = 1000
best_i = 0
best_j = 0
for i in range(lower_t, upper_t):
    for j in range(lower_window, upper_window):
        chi2 = perform_fit(i,i+j, plot_raw=False, plot_meff=False)
        if abs(chi2 - 1)  < abs(best_chi2 - 1):
            best_chi2 = chi2
            best_i = i
            best_j = j 
            print(best_chi2, best_i, best_j)


print(best_chi2, best_i, best_j)
perform_fit(best_i, best_i + best_j, plot_raw = True, plot_meff=False)
# plt.figure()
# plt.plot(np.arange(lower, upper), E0_fits)
# plt.title("E0_fits")
# plt.show()
# plt.figure()
# plt.plot(np.arange(lower, upper), E0_chi2)
# plt.title("E0 Chi Square")
# plt.show()
# plt.figure()
# plt.plot(np.arange(lower, upper), m_eff_fits)
# plt.title("M_eff fits")
# plt.show()
# plt.figure()
# plt.plot(np.arange(lower, upper), m_eff_chi2)
# plt.title("M_eff chi square")
# plt.show()



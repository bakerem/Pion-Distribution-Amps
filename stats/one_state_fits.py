import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, fsolve


lower_lim = 15  
upper_lim = 35

Ns = 29
Nt = 128
# create list of columns for df
columns = ["t"]
for i in range(0,Ns):
    columns.append(f"{i}")

# read in and create mean and std columns 
df = pd.read_csv("/home/bakerem/ANL/stats/c2pt-data/64IGSRC_W40_k0.ama.c2pt.PX0PY0PZ3.real.cfg.csv",names=columns, dtype=np.float64)
df_data = df.drop(axis="columns", labels=["t"])

# jacknife error
samples = np.zeros((Ns,Nt))
for i in range(0,Ns):
    sample = df_data.drop(axis="columns", labels=[str(i)]).mean(axis=1)
    samples[i,:] = sample
means = np.average(samples, axis=0)
std_devs = np.sqrt(Ns-1)*np.std(samples, axis = 0)
df["mean"] = means
df["std dev"] = std_devs


# objective function for fitting
def obj_func(t,A0, E0):
    y = A0*np.exp(-t*E0) + A0*np.exp(-(Nt-t)*E0)  
    return y

# curve fitting and calculating chi squared
p0 = (10**7, 0.06)
bounds = ([0,0.05],[np.inf,3])
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

#data saving routine
prev_data = np.load("stats/E0data_1state.npy")
new_data = np.array((popt[1],pcov[1,1],chi2))
data = np.concatenate((prev_data, new_data))
np.save("stats/E0data_1state.npy", data)


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


#######PLOTS#######


# raw data plots w/o error bars
# plt.figure()
# plt.plot(df["t"].iloc[lower_lim:upper_lim], df["mean"].iloc[lower_lim:upper_lim], ".")
# plt.plot(df["t"].iloc[lower_lim:upper_lim], obj_func(df["t"].iloc[lower_lim:upper_lim],*popt))
# # plt.yscale("log")
# tab_cols = ("Value", "Error")
# tab_rows = ("A0","E0", "$\chi^2$")
# cells = [["%.2e" %popt[0], "%.2e" %np.sqrt(pcov[0,0])]\
#          ,["%.2e" %popt[1],"%.2e" %np.sqrt(pcov[1,1])]\
#          ,["%.3f" %chi2, "n/a"]]
# plt.table(cellText=cells, rowLabels=tab_rows, colLabels=tab_cols, loc="upper center", colWidths=[0.2,0.2])
# plt.show()


# # raw data plots w/ error bars
plt.errorbar(df["t"].iloc[lower_lim:upper_lim], 
             df["mean"].iloc[lower_lim:upper_lim],
             yerr=df["std dev"].iloc[lower_lim:upper_lim], 
             capsize=3, 
             fmt=".")
plt.plot(df["t"].iloc[lower_lim:upper_lim], obj_func(df["t"].iloc[lower_lim:upper_lim],*popt))
plt.yscale("log")
tab_cols = ("Value", "Error")
tab_rows = ("A0", "E0", "$\chi^2$")
cells = [["%.2e" %popt[0], "%.2e" %np.sqrt(pcov[0,0])]\
         ,["%.2e" %popt[1],"%.2e" %np.sqrt(pcov[1,1])]\
         ,["%.3f" %chi2, "n/a"]]
plt.table(cellText=cells, rowLabels=tab_rows, colLabels=tab_cols, loc="upper center", colWidths=[0.2,0.2])
plt.xlabel("$n_t$")
plt.ylabel("$C(n_t)$")
plt.savefig("stats/Pz2corrfunc.png")
plt.show()

# effective mass plots
# plt.figure()
# plt.errorbar(df["t"].iloc[lower_lim:upper_lim],
#               roots[lower_lim:upper_lim], 
#               yerr= np.abs(roots_err[lower_lim:upper_lim]),
#               capsize=3,
#               fmt="s")

# plt.plot(df["t"].iloc[lower_lim:upper_lim], 
#          np.ones((upper_lim-lower_lim,))*obj_func_meff(df["t"].iloc[lower_lim:upper_lim], 
#          *popt_meff))

# plt.xlabel("$n_t$")
# plt.ylabel("$m_eff$")
# tab_cols = ("Value", "Error")
# tab_rows = ("$m_{eff}$", "$\chi^2$")
# cells = [["%.2e" %popt_meff[0], "%.2e" %np.sqrt(pcov_meff[0,0])]\
#          ,["%.3f" %chi2_meff, "n/a"]]
# plt.table(cellText=cells, rowLabels=tab_rows, colLabels=tab_cols, loc="upper center", colWidths=[0.2,0.2])
# plt.savefig("stats/1_state_meff_bad.png")
# plt.show()


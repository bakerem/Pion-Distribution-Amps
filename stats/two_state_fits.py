import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, fsolve


# create list of columns for df
columns = ["t"]
for i in range(0,30):
    columns.append(f"{i}")

# read in and creat mean and std columns and delete all others (except time)
df = pd.read_csv("/home/bakerem/ANL/stats/c2pt-data/64IGSRC_W40_k6.ama.c2pt.PX0PY0PZ2.real.cfg.csv",names=columns, dtype=np.float64)
df['mean'] = df.mean(axis=1)
df["std dev"] = df.std(axis=1)
df = df.drop(axis="columns", labels=columns[1:])

Nt = 128
# objective function for fitting
def obj_func(t,A0,A1, E0, E1):
    y = A0*np.exp(-t*E0) + A0*np.exp(-(Nt-t)*E0)  \
      + A1*np.exp(-t*E1) + A1*np.exp(-(Nt-t)*E1) 
    return y

# curve fitting and calculating chi squared
p0 = (10**5, 10**5,0.39, 0.5)
bounds = ([0,0,0.35, 0.35],[np.inf, np.inf,0.41, 1])
popt, pcov = curve_fit(obj_func, df["t"], df["mean"],sigma=df["std dev"],p0=p0, bounds=bounds)
chi2 = np.sum((df["mean"]-obj_func(df["t"],*popt))**2/df["std dev"]**2)

# prev_data = np.load("stats/E0data_2state.npy")
# new_data = np.array((popt[2],pcov[2,2]))
# data = np.concatenate((prev_data, new_data))
# np.save("stats/E0data_2state.npy", data)
# for effective mass calculations
Cn = np.array(obj_func(df["t"], *popt))
Cn1 = np.array(obj_func(df["t"].add(1), *popt))
ratio_fit = Cn/Cn1


# errors (not really working)
# Cn_err1 = np.array(obj_func(df["t"], popt[0]+np.sqrt(pcov[0,0]), popt[1])) - Cn
# Cn_err2 = np.array(obj_func(df["t"], popt[0], popt[1]+np.sqrt(pcov[1,1]))) - Cn
# Cn_err = np.sqrt(Cn_err1**2 + Cn_err2**2)
# Cn1_err1 = np.array(obj_func(df["t"], popt[0]+np.sqrt(pcov[0,0]), popt[1])) - Cn1
# Cn1_err2 = np.array(obj_func(df["t"], popt[0], popt[1]+np.sqrt(pcov[1,1]))) - Cn1
# Cn1_err = np.sqrt(Cn1_err1**2 + Cn1_err2**2)
# ratio_fit = Cn/Cn1
# ratio_err1 = ((Cn+Cn_err)/Cn1) - ratio_fit
# ratio_err2 = (Cn/(Cn1+Cn1_err)) - ratio_fit
# ratio_err_tot = np.sqrt(ratio_err1**2 + ratio_err2**2)


# solving equation for effective mass
def solvingfunc(m,t,Nt,ratio_fit_i):
    result = np.cosh(m*(t-0.5*Nt))/np.cosh(m*(t+1-0.5*Nt)) - ratio_fit_i
    return result
roots = np.empty((Nt,))
roots_err = np.empty((Nt,))
for t in range(0,128): #solve for m_eff at each time step
    root = fsolve(solvingfunc, x0=0.6, args=(t,Nt,ratio_fit[t]))
    roots[t] = root

# raw data plots w/o error bars
# plt.figure()
# plt.plot(df["t"], df["mean"], ".")
# plt.plot(df["t"], obj_func(df["t"],*popt))
# plt.yscale("log")
# tab_cols = ("Value", "Error")
# tab_rows = ("A0","A1", "E0", "E1", "$\chi^2$")
# cells = [["%.2e" %popt[0], "%.2e" %pcov[0,0]]\
#          ,["%.2e" %popt[1],"%.2e" %pcov[1,1]]\
#          ,["%.2e" %popt[2],"%.2e" %pcov[2,2]]\
#          ,["%.2e" %popt[3],"%.2e" %pcov[3,3]]\
#          ,["%.3f" %chi2, "n/a"]]
# plt.table(cellText=cells, rowLabels=tab_rows, colLabels=tab_cols, loc="upper center", colWidths=[0.2,0.2])
# plt.show()

# raw data plots w/ error bars
plt.figure()
plt.errorbar(df["t"], df["mean"],yerr=df["std dev"], capsize=1, fmt=".")
plt.plot(df["t"], obj_func(df["t"],*popt))
plt.yscale("log")
tab_cols = ("Value", "Error")
tab_rows = ("A0","A1", "E0", "E1", "$\chi^2$")
cells = [["%.2e" %popt[0], "%.2e" %pcov[0,0]]\
         ,["%.2e" %popt[1],"%.2e" %pcov[1,1]]\
         ,["%.2e" %popt[2],"%.2e" %pcov[2,2]]\
         ,["%.2e" %popt[3],"%.2e" %pcov[3,3]]\
         ,["%.3f" %chi2, "n/a"]]
plt.table(cellText=cells, rowLabels=tab_rows, colLabels=tab_cols, loc="upper center", colWidths=[0.2,0.2])
plt.show()

# effective mass plots
# plt.figure()
# plt.plot(df["t"], abs(roots))

# plt.show()
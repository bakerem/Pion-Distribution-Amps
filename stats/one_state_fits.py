import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, fsolve





# create list of columns for df

columns = ["t"]
for i in range(0,30):
    columns.append(f"{i}")

# read in and creat mean and std columns and delete all others (except time)
old_df = pd.read_csv("/home/bakerem/ANL/stats/c2pt-data/64IGSRC_W40_k0.ama.c2pt.PX0PY0PZ0.real.cfg.csv",names=columns, dtype=np.float64)
old_df['mean'] = old_df.mean(axis=1)
old_df["std dev"] = old_df.drop(axis="columns",labels=["t"]).std(axis=1)
df = old_df.drop(axis="columns", labels=columns[1:])


Nt = 128
# objective function for fitting
def obj_func(t,A0, E0):
    y = A0*np.exp(-t*E0) + A0*np.exp(-(Nt-t)*E0)  
    return y

# curve fitting and calculating chi squared
p0 = (10**3,0.3)
bounds = ([0,0],[np.inf,1])
popt, pcov = curve_fit(obj_func, df["t"].iloc[20:109], df["mean"].iloc[20:109],sigma=df["std dev"].iloc[20:109],p0=p0, bounds=bounds)
chi2 = np.sum((df["mean"].iloc[20:109]-obj_func(df["t"].iloc[20:109],*popt))**2/df["std dev"].iloc[20:109]**2)

# prev_data = np.load("stats/E0data_1state.npy")
# new_data = np.array((popt[1],pcov[1,1]))
# data = np.concatenate((prev_data, new_data))
# np.save("stats/E0data_1state.npy", data)


# for effective mass calculations
Cn = np.array(obj_func(df["t"].iloc[20:109], *popt))
Cn1 = np.array(obj_func(df["t"].iloc[20:109].add(1), *popt))
ratio_fit = Cn/Cn1
ratio_actual = df["mean"].shift(1,fill_value=df["mean"].iloc[0])/df["mean"]
ratio_actual_err = ratio_actual * np.sqrt((df["std dev"]/df["mean"])**2 \
                    + (df["std dev"].shift(1,fill_value=df["std dev"].iloc[0])/df["mean"].shift(1,fill_value=df["mean"].iloc[0])**2))




# solving equation for effective mass
def solvingfunc(m,t,Nt,ratio_i):
    result = np.cosh(m*(t-0.5*Nt))/np.cosh(m*(t+1-0.5*Nt)) - ratio_i
    return result
roots = np.empty((Nt,))
roots_err = np.empty((Nt,))
for t in range(20,109): #solve for m_eff at each time step
    root = fsolve(solvingfunc, x0=0.06, args=(t,Nt,ratio_actual[t]))
    root_diff = fsolve(solvingfunc, x0=0.06, args=(t,Nt, ratio_actual[t] + ratio_actual_err[t]))
    roots[t] = root
    roots_err[t] = root - root_diff


print(roots_err[10:15])
def obj_func_meff(t, m_eff):
    return m_eff

popt_meff, pcov_meff = curve_fit(
                obj_func_meff, 
                df["t"].iloc[20:109].drop([63,64,65]), 
                np.concatenate((roots[20:63],roots[66:109])),
                sigma=np.concatenate((roots_err[20:63],roots_err[66:109])))

chi2_meff = np.sum(
            (np.concatenate((roots[20:63],roots[66:109]))-obj_func_meff(df["t"].iloc[20:109],*popt_meff))**2\
            /np.concatenate((roots_err[20:63],roots_err[66:109]))**2)


# mass plateau error
# each_cn = np.empty((30,89))
# each_cn1 = np.empty((30,89))

# for i in range(0,29):
#     popt, pcov = curve_fit(obj_func, old_df["t"].iloc[20:109], old_df[f"{i}"].iloc[20:109],p0=p0, bounds=bounds)
#     each_cn[i,:] = np.array(obj_func(old_df["t"].iloc[20:109], *popt))
#     each_cn1[i,:] = np.array(obj_func(old_df["t"].add(1).iloc[20:109], *popt))
# avg_cn = np.average(each_cn, axis=0)
# avg_cn1 = np.average(each_cn1, axis=0)

# cov = np.empty((89,89))
# cov1 = np.empty((89,89))
# for i in range(1,89):
#     for j in range(1,89):
#         N_tests = 29
#         cov[i,j] = (1/N_tests-1)*np.average((each_cn[:,i] - avg_cn[i])*(each_cn[:,j]-avg_cn[j]))
#         cov1[i,j] = (1/N_tests-1)*np.average((each_cn1[:,i] - avg_cn1[i])*(each_cn1[:,j]-avg_cn1[j]))


# cov = np.abs(np.diag(cov))
# cov1 = np.abs(np.diag(cov1))
# ratio_real = avg_cn/avg_cn1
# ratio_real_err = ratio_real*np.sqrt((cov1/avg_cn1**2)+(cov/avg_cn**2))
# print(ratio_real_err[:5])

################################################################################################################
#################################################### PLOTS #####################################################
################################################################################################################

# raw data plots w/o error bars
# plt.figure()
# plt.plot(df["t"].iloc[20:109], df["mean"].iloc[20:109], ".")
# plt.plot(df["t"].iloc[20:109], obj_func(df["t"].iloc[20:109],*popt))
# plt.yscale("log")
# tab_cols = ("Value", "Error")
# tab_rows = ("A0","E0", "$\chi^2$")
# cells = [["%.2e" %popt[0], "%.2e" %np.sqrt(pcov[0,0])]\
#          ,["%.2e" %popt[1],"%.2e" %np.sqrt(pcov[1,1])]\
#          ,["%.3f" %chi2, "n/a"]]
# plt.table(cellText=cells, rowLabels=tab_rows, colLabels=tab_cols, loc="upper center", colWidths=[0.2,0.2])
# plt.show()


# # raw data plots w/ error bars
# plt.errorbar(df["t"].iloc[20:109], df["mean"].iloc[20:109],yerr=df["std dev"].iloc[20:109], capsize=1, fmt=".")
# plt.plot(df["t"].iloc[20:109], obj_func(df["t"].iloc[20:109],*popt))
# plt.yscale("log")
# tab_cols = ("Value", "Error")
# tab_rows = ("A0", "E0", "$\chi^2$")
# cells = [["%.2e" %popt[0], "%.2e" %np.sqrt(pcov[0,0])]\
#          ,["%.2e" %popt[1],"%.2e" %np.sqrt(pcov[1,1])]\
#          ,["%.3f" %chi2, "n/a"]]
# plt.table(cellText=cells, rowLabels=tab_rows, colLabels=tab_cols, loc="upper center", colWidths=[0.2,0.2])
# plt.show()

# effective mass plots
plt.figure()
plt.errorbar(df["t"].iloc[20:109].drop([63,64,65]),
              np.concatenate((roots[20:63],roots[66:109])), 
              yerr= np.abs(np.concatenate((roots_err[20:63],roots_err[66:109]))),
              fmt="v")
plt.plot(df["t"].iloc[20:109], np.ones((89,))*obj_func_meff(df["t"].iloc[20:109], *popt_meff))
plt.xlabel("$n_t$")
plt.ylabel("$m_eff$")
tab_cols = ("Value", "Error")
tab_rows = ("$m_eff$", "$\chi^2$")
cells = [["%.2e" %popt_meff[0], "%.2e" %np.sqrt(pcov_meff[0,0])]\
         ,["%.3f" %chi2_meff, "n/a"]]
plt.table(cellText=cells, rowLabels=tab_rows, colLabels=tab_cols, loc="upper center", colWidths=[0.2,0.2])
# plt.savefig("stats/1_state_meff.pdf")
plt.show()


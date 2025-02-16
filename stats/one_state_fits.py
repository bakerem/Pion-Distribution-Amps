from networkx import faster_could_be_isomorphic
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit, fsolve
import os
import scienceplots

plt.style.use('science')

"""
Ethan Baker, Haverford College/ANL
Uses a one-state two-parameter fit for fitting C2pt data. Pz gives the
momentum of the data in simulation units, p0 is an initial guess for 
fitting and bounds are bounds for the fit. This is mostly used to check the
data to see if it conforms to the dispersion relation, but there may be some
errors since it is only a one parameter fit.

The results of this fit are used in "one_state_plots.py" in this directory
to produce a plot of the dispersion relation. That relies on a list called 
bestE0s, which is put together manually by finding the t_min value that
produces the fit with the lowest chi-squared during the fit. These are printed
when the script is run to allow for easy computation. 
"""

Nt = 128
Ns = 55
a = 1/2.359 # GeV^-1

Pz = 0
save = False
p0 = (1e8, 0.1)
bounds = ([1e6,0],[1e10,10])

lower_t = 2
upper_t = 40
window = 14
# create list of columns for df
save_path = f"final_results/one_state_fits/Pz{Pz}"
os.makedirs(save_path, exist_ok=True)


columns = ["t"] + [str(i) for i in range(0, Ns)]
# read in data as pandas dataframe 

if Pz < 4:
    df = pd.read_csv(f"0-data/c2pt_cfgs/64IGSRC_W40_k0.ama.c2pt.PX0PY0PZ{Pz}.real.cfg.csv",names=columns, dtype=np.float64)
else:
    df = pd.read_csv(f"0-data/c2pt_cfgs/64IGSRC_W40_k6.ama.c2pt.PX0PY0PZ{Pz}.real.cfg.csv",names=columns, dtype=np.float64)


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


# plt.figure()
# plt.plot(df["t"], 
#          np.abs(np.log(abs(df["mean"]/df["mean"].shift(-1,fill_value=df["mean"].iloc[0])))),
#         )
# title = r"ln$(C(n)/C(n+1))$" + f" for Nz={Pz}"
# plt.title(title)
# plt.xlabel(r"n_t")
# plt.ylabel(r"ln$(C(n)/C(n+1))$")
# plt.savefig(f"{save_path}/Pz{Pz}_logplot.png")
# plt.show()


# main function for fitting routine
def perform_fit(lower_lim, upper_lim, plot_raw = False, plot_meff = False):
    def obj_func(t,A0, E0):
        y = A0*np.exp(-t*E0) + A0*np.exp(-(Nt-t)*E0)  
        return y

    # curve fitting and calculating chi squared

    popt, pcov = curve_fit(
                    obj_func, df["t"].iloc[lower_lim:upper_lim], 
                    df["mean"].iloc[lower_lim:upper_lim], 
                    sigma = cov[lower_lim:upper_lim,lower_lim:upper_lim],
                    p0=p0, 
                    bounds=bounds,
                    maxfev=2000,
                    # method="dogbox",
                    )

    chi2 = np.sum(
            (df["mean"].iloc[lower_lim:upper_lim]-
            obj_func(df["t"].iloc[lower_lim:upper_lim],*popt))**2/
            (df["std dev"].iloc[lower_lim:upper_lim])**2) / (upper_lim - lower_lim - len(popt))


    # solving equation for effective mass
    def solvingfunc(m,t,Nt,ratio_i):
        result = np.cosh(m*(t-0.5*Nt))/np.cosh(m*(t+1-0.5*Nt)) - ratio_i
        return result
    
    def f_prime(m,t,Nt,ratio_i):
        result = ((1/np.cosh(m*(-0.5*Nt+t+1)))
                  *((t-0.5*Nt)*np.sinh(m*(t-0.5*Nt))+0.5*(Nt-2*(t+1))*np.cosh(m*(t-0.5*Nt))
                  *np.tanh(m*(-0.5*Nt+t+1)))

        )
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
                        #   fprime=f_prime,
                          x0=popt[1], 
                          args=(t,Nt,samples[i,t]/samples[i,t+1]))
            roots_all_samples[i,t] = root
    roots = np.average(roots_all_samples, axis=0)
    roots_err = np.sqrt(Ns-1)*np.std(roots_all_samples, axis=0)


    # function for fitting m_eff 
    def obj_func_meff(t, m_eff):
        return m_eff

    # performs fits and calculates chi square
    popt_meff, pcov_meff = curve_fit(
                    obj_func_meff, 
                    df["t"].iloc[lower_lim:upper_lim], 
                    roots[lower_lim:upper_lim],
                    sigma=roots_err[lower_lim:upper_lim],
                    method="lm")

    chi2_meff = np.sum((roots[lower_lim:upper_lim]-obj_func_meff(df["t"].iloc[lower_lim:upper_lim],*popt_meff))**2 \
                       / roots_err[lower_lim:upper_lim]**2) / (upper_lim - lower_lim - len(popt))


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
        tab_rows = ("A0", "E0", r"$\chi^2$")
        cells = [["%.2e" %popt[0], "%.2e" %np.sqrt(pcov[0,0])]\
                ,["%.2e" %(popt[1]/a),"%.2e" %(np.sqrt(pcov[1,1])/a)]\
                ,["%.3f" %chi2, "n/a"]]
        plt.table(cellText=cells, rowLabels=tab_rows, colLabels=tab_cols, loc="upper center", colWidths=[0.2,0.2])
        plt.xlabel("$n_t$")
        plt.ylabel("$C(n_t)$")
        plt.title(f"Nz = {Pz}")
        if save == True:
            plt.savefig(f"{save_path}/Pz{Pz}_corrfunc.pdf")
        plt.show()


    if plot_meff == True:
        # effective mass plots
        plt.figure()
        plt.errorbar(df["t"].iloc[lower_lim:upper_lim],
                    roots[lower_lim:upper_lim]/a, 
                    yerr=np.abs(roots_err[lower_lim:upper_lim])/a,
                    capsize=3,
                    fmt="bs",
                    markerfacecolor="none"
                    )


        plt.xlabel("$t/a$")
        plt.ylabel(r"$m_{eff}$ (GeV)")
        # plt.title(r"$m_{eff}$ " + f"at Nz = {Pz}")
        # plt.ylim(0,2)
        plt.fill_between([20,40],0.13,0.15, alpha=0.2, color="blue",)
        # tab_cols = ("Value", "Error")
        # tab_rows = ("$m_{eff}$", r"$\chi^2$")
        # cells = [["%.2e" %(popt_meff[0]/a), "%.2e" %(np.sqrt(pcov_meff[0,0])/a)]\
        #         ,["%.3f" %chi2_meff, "n/a"]]
        # plt.table(cellText=cells, rowLabels=tab_rows, colLabels=tab_cols, loc="upper center", colWidths=[0.2,0.2])
        if save == True:
            plt.savefig(f"{save_path}/Pz{Pz}_meff_filled.pdf")
        plt.show()
    return chi2, popt[1], np.sqrt(pcov[1,1]), popt[0], np.sqrt(pcov[0,0])


# creates plot of E_0 fitted at each t_min ranging from lower_t to upper_t.
fit_results = np.zeros((5,upper_t-window-lower_t))
perform_fit(2,50,plot_meff=True)
# plt.ylim(0,2)
for i in range(lower_t, upper_t-window):
    results = perform_fit(i,upper_t, plot_raw=False)
    fit_results[0,i-lower_t] = results[1] # save E0
    fit_results[1,i-lower_t] = results[2] # save E0 err
    fit_results[2,i-lower_t] = results[3] # save A0
    fit_results[3,i-lower_t] = results[4] # save A0 err
    fit_results[4,i-lower_t] = results[0] # save chi2
    print(i, results[0], results[1]/a, results[2]/a)

plt.figure()
plt.errorbar(np.arange(lower_t, upper_t-window), 
             np.array(fit_results[0]/a), 
             yerr=(np.array(fit_results[1])/a), 
             fmt="rs", 
             capsize=4,
             markerfacecolor='none',)
plt.xlabel(r"$t_\text{min}/a$")
plt.ylabel(r"$E_0(P_z)$ (Gev)")
# plt.ylim(0,4)
# plt.text(15,0.141, "Pz = %.2f GeV" %phys_p(a, Pz), fontfamily="sans-serif", fontsize="large", fontstyle="normal")
# plt.title(r"Fitted $E_0$ from [$t_{min}/a$, " + f"{upper_t}]")
if save:
    plt.savefig(f"{save_path}/Pz{Pz}window_length{window}.pdf")
    np.save(f"{save_path}/E0_fits_Pz{Pz}.npy", fit_results)
plt.show()





from networkx import multi_source_dijkstra
import numpy as np
import h5py
import pandas as pd
from math import factorial
from scipy.special import gamma, jv, gegenbauer
import matplotlib.pyplot as plt


C_F = 4/3
gamma_E = 0.57721
a = 2.359
mu_0 = 2
alpha_0 = 0.303
beta_0 = (33-2*3)/(6*2*np.pi)



def read_data(Ns: int, Nt: int, init_char: str, Pz: int, bz: int, smear:str) -> list:
    """
    Reads in the data from h5 file and pandas dataframe and calculates errors
    and means
    """

    columns = ["t"] + [str(i) for i in range(0,Ns)]
    if Pz < 4:
        file = h5py.File(f"0-data/qDA_cfgs/64I.qDA.ama.GSRC_W40_k0_flow{smear}eps01.{init_char}.eta0.PX0PY0PZ{Pz}.h5")
        DA_data = file[f"b_X/bT0/bz{bz}"]
 
        # read in 2pt correlation data
        c2pt = pd.read_csv(f"0-data/c2pt_cfgs/64IGSRC_W40_k0.ama.c2pt.PX0PY0PZ{Pz}.real.cfg.csv", names=columns)
        c2pt_data = np.array(c2pt.drop(axis="columns", labels=["t"])).transpose()
    else:
        file = h5py.File(f"0-data/qDA_cfgs/64I.qDA.ama.GSRC_W40_k6_flow{smear}eps01.{init_char}.eta0.PX0PY0PZ{Pz}.h5")
        DA_data = file[f"b_X/bT0/bz{bz}"]
        # read in 2pt correlation data
        c2pt = pd.read_csv(f"0-data/c2pt_cfgs/64IGSRC_W40_k6.ama.c2pt.PX0PY0PZ{Pz}.real.cfg.csv", names=columns)
        c2pt_data = np.array(c2pt.drop(axis="columns", labels=["t"])).transpose()

    # calculate means and error in ratio of DA/c2pt
    real_samples      = np.zeros((Ns,Nt))
    imag_samples      = np.zeros((Ns,Nt))
    real_sample_errs  = np.zeros((Ns,Nt))
    imag_sample_errs = np.zeros((Ns,Nt))
    ratio = DA_data/c2pt_data
    if init_char == "Z5":
        ratio *= -1j
   
    real_ratio = np.real(ratio)
    imag_ratio = np.imag(ratio)
    
    for i in range(0,Ns):
        real_sample = np.mean(np.delete(real_ratio,i,axis=0), axis=0)
        imag_sample = np.mean(np.delete(imag_ratio,i,axis=0), axis=0)
        real_samples[i,:] = real_sample
        imag_samples[i,:] = imag_sample
        real_sample_errs[i,:] = np.std(np.delete(real_ratio,i,axis=0), axis=0)
        imag_sample_errs[i,:] = np.std(np.delete(imag_ratio,i,axis=0), axis=0)

    real_ratio_means = np.mean(real_samples, axis=0)
    imag_ratio_means = np.mean(imag_samples, axis=0)
    
    real_ratio_stds = np.sqrt(Ns-1)*np.std(real_samples, axis = 0)
    imag_ratio_stds = np.sqrt(Ns-1)*np.std(imag_samples, axis = 0)


    return (
        real_ratio_means, 
        imag_ratio_means, 
        real_ratio_stds, 
        imag_ratio_stds,
        real_samples,
        real_sample_errs,
        imag_samples,
        imag_sample_errs)



def phys_p(a: float, n: int):
    """takes a in GeV^-1 and returns physical momentum assosicated with n """
    return 2 * np.pi * n * a / 64

def alpha_func(mu):
    a_s = (303)/(1000+beta_0*303*np.log(0.25*mu**2))
    return a_s
### Definitions for Mellin OPE ###
def m_ope(bz, mm2, mm4, mm6, mm8, l, h0, h1, N_max, N_ht, Pz, alpha_s, mu,  init_char):
    """
    Mellin OPE. Eq. (6) from  Gao et al. 2022, arxiv:2206.04084v2
    There are two sets of coefficients depending on the situation, hence 
    the if statement.
    """
    m_moms = [1, 0, mm2, 0, mm4, 0, mm6, 0, mm8]
    ev2 = (mu/2)**(-25/(6*4*np.pi)*alpha_s*C_F)
    ev4 = (mu/2)**(-91/(15*4*np.pi)*alpha_s*C_F)

    # ev2 = (alpha_s/alpha_0)**(-25/(6*4*np.pi*beta_0)*C_F)
    # ev4 = (alpha_s/alpha_0)**(-91/(15*4*np.pi*beta_0)*C_F)

    ev_facs = [1,0,ev2, 0, ev4]

    lam = bz*Pz
    h_tw2 = 0
    if init_char == "Z5":
        for n in range(0,N_max+1):
            term1 = (-1j*lam/2)**n/factorial(n)
            term2 = 0
            for m in range(0,n+1):
                term2 += m_coeff_Z5(n,m, bz, alpha_s, mu)*m_moms[m]*ev_facs[m]
            h_tw2 += term1*term2
    

    elif init_char == "T5":
        for n in range(0,N_max+1):
            term1 = (-1j*lam/2)**n/factorial(n)
            term2 = 0
            for m in range(0,n+1):
                term2 += m_coeff_T5(n,m, bz, alpha_s, mu)*m_moms[m]*ev_facs[m]
            h_tw2 += term1*term2
    
    l_corr = l*(Pz*a)**2
    h_corr = 0
    hs = [h0,h1]
    for i in range(0, N_ht):
        h_corr += bz**2*hs[i]*F_n(i, bz, Pz, init_char, 0, mu)

    return h_tw2 + l_corr + h_corr

def m_coeff_Z5(n,m, bz, alpha_s, mu):
    """
    Mellin Coefficents for Z5
    """
    L = np.log((bz)**2*mu**2*np.exp(2*gamma_E)/4)
    # determination of C0
    if n == m:
        C0 = 1
    else:
        C0 = 0
    # determination of C1_mn
    if   n==0:
        C1 = 1.5*L + 3.5

    
    elif n==1 and m==1:
        C1 = (17/6)*L - 0.5

    elif n==2 and m==0:
        C1 = (-5/12)*L + (11/12)
    elif n==2 and m==2:
        C1 = (43/12)*L - (37/12)

    
    elif n==3 and m==1:
        C1 = (-11/20)*L + (79/60)
    elif n==3 and m==3:
        C1 = (247/60)*L - (923/180)

    elif n==4 and m==0:
        C1 = (-2/15)*L + 0.25
    elif n==4 and m==2:
        C1 = (-19/30)*L + (5/3)
    elif n==4 and m==4:
        C1 = (68/15)*L - (247/36)

    elif n==5 and m==1:
        C1 = (2235/6300) - (1200/6300)*L
    elif n==5 and m==3:
        C1 = (12420/6300) - (30*145/6300)*L
    elif n==5 and m==5:
        C1 = (-52813/6300) + (30720/6300)*L

    elif n==6 and m==0:
        C1 = (1475/12600) - (825/12600)*L
    elif n==6 and m==2:
        C1 = (15*383/12600) - (15*195/12600)*L
    elif n==6 and m==4:
        C1 = (45*627/12600) - (45*205/12600)*L
    elif n==6 and m==6:
        C1 = (-122831/12600)+(65115/12600)*L


    elif n==7 and m==1:
        C1 = (175*83/88200) - (175*49/88200)*L
    elif n==7 and m==3:
        C1 = (35*1389/88200) - (35*665/88200)*L
    elif n==7 and m==5:
        C1 = (35*6243/88200) - (35*1925/88200)*L
    elif n==7 and m==7:
        C1 = (-969497/88200) + (478205/88200)*L

    elif n==8 and m==0:
        C1 = (1715*7/176400) - (4*1715/176400)*L
    elif n==8 and m==2:
        C1 = (140*267/176400) - (140*154/176400)*L
    elif n==8 and m==4:
        C1 = (70*1613/176400) - (70*728/176400)*L
    elif n==8 and m==6:
        C1 = (140*3391/176400) - (140*994/176400)*L
    elif n==8 and m==8:
        C1 = (-2140979/176400) + (996100/176400)*L
    else:
        C1 = 0


    coeff = C0 + alpha_s*C_F/(2*np.pi)*C1
    return coeff


def m_coeff_T5(n,m, bz, alpha_s, mu):
    """
    Wilson Coefficents for T5
    """
    L = np.log((bz)**2*mu**2*np.exp(2*gamma_E)/4)
    # determination of C0
    if n == m:
        C0 = 1
    else:
        C0 = 0
    # determination of C1_mn
    if   n==0:
        C1 = 2.5 + 1.5*L

    elif n==1 and m==1:
        C1 = (-5/6) + (17/6)*L

    elif n==2 and m==0:
        C1 = 0.75 - (5/12)*L
    elif n==2 and m==2:
        C1 = (-13/4) + (43/12)*L

    
    elif n==3 and m==1:
        C1 = (73/60) - (11/20)*L
    elif n==3 and m==3:
        C1 = (-941/180) + (247/60)*L

    elif n==4 and m==0:
        C1 = (33/180) - (24/180)*L
    elif n==4 and m==2:
        C1 = (6*48/180) - (6*19/180)*L
    elif n==4 and m==4:
        C1 = (-1247/180) + (816/180)*L

    elif n==5 and m==1:
        C1 = (15*129/6300) - (15*80/6300)*L
    elif n==5 and m==3:
        C1 = (30*404/6300) - (30*145/6300)*L
    elif n==5 and m==5:
        C1 = (-53113/6300) + (30720/6300)*L

    elif n==6 and m==0:
        C1 = (1025/12600) - (825/12600)*L
    elif n==6 and m==2:
        C1 = (15*353/12600) - (15*195/12600)*L
    elif n==6 and m==4:
        C1 = (45*617/12600) - (45*205/12600)*L
    elif n==6 and m==6:
        C1 = (-123281/12600) + (65115/12600)*L


    elif n==7 and m==1:
        C1 = (175*69/88200) - (175*49/88200)*L
    elif n==7 and m==3:
        C1 = (35*1319/88200) - (35*665/88200)*L
    elif n==7 and m==5:
        C1 = (35*6173/88200) - (35*1925/88200)*L
    elif n==7 and m==7:
        C1 = (-971947/88200) + (478205/88200)*L

    elif n==8 and m==0:
        C1 = (8085/176400) - (6860/176400)*L
    elif n==8 and m==2:
        C1 = (140*239/176400) - (140*154/176400)*L
    elif n==8 and m==4:
        C1 = (70*1557/176400) - (70*728/176400)*L
    elif n==8 and m==6:
        C1 = (140*3363/176400) - (140*994/176400)*L
    elif n==8 and m==8:
        C1 = (-2144899/176400) + (996100/176400)*L
    else:
        C1 = 0

    coeff = C0 + alpha_s*C_F/(2*np.pi)*C1
    return coeff

def x_dep1_m_ope(bz, alpha, l, h0, h1, N_max, N_ht, Pz, alpha_s, mu, init_char):
    
    mm2 = 1/(3+2*alpha)
    mm4 = 3/(15+16*alpha+4*alpha**2)
    mm6 = 15/(105 + 142*alpha + 60*alpha**2 + 8*alpha*3)
    mm8 = 105/((3 + 2*alpha)*(5 + 2*alpha)*(7 + 2*alpha)*(9 + 2*alpha))
    
    ev2 = (mu/2)**(-25/(6*4*np.pi)*alpha_s*C_F)
    ev4 = (mu/2)**(-91/(15*4*np.pi)*alpha_s*C_F)
    # ev2 = (alpha_s/alpha_0)**(-25/(6*4*np.pi*beta_0)*C_F)
    # ev4 = (alpha_s/alpha_0)**(-91/(15*4*np.pi*beta_0)*C_F)
    ev_facs = [1, 0, ev2, 0, ev4]
    m_moms  = [1, 0, mm2, 0, mm4,0, mm6, 0, mm8]
    lam = bz*Pz
    h_tw2 = 0

    if init_char == "Z5":
        for n in range(0,N_max+1):
            term1 = (-1j*lam/2)**n/factorial(n)
            term2 = 0
            for m in range(0,n+1):
                term2 += m_coeff_Z5(n,m, bz, alpha_s, mu)*m_moms[m]*ev_facs[m]
            h_tw2 += term1*term2

    elif init_char == "T5":
        for n in range(0,N_max+1):
            term1 = (-1j*lam/2)**n/factorial(n)
            term2 = 0
            for m in range(0,n+1):
                term2 += m_coeff_T5(n,m, bz, alpha_s, mu)*m_moms[m]*ev_facs[m]
            h_tw2 += term1*term2
    
    l_corr = l*(Pz*a)**2
    h_corr = 0
    hs = [h0,h1]
    for i in range(0, N_ht):
        h_corr += bz**2*hs[i]*F_n(i, bz, Pz,init_char, 0, mu)

    return h_tw2 + l_corr + h_corr

def x_dep_many_m_ope(bz, alpha, s1, s2, l, h0, h1, N_max, N_ht, Pz, alpha_s, mu, init_char, Ng):
    if Ng == 1:
        mm2 = (5+2*alpha+2*s1+6*alpha*s1+4*alpha**2*s1)/(15+16*alpha+4*alpha**2)
        mm4 = (3*(7+2*alpha+4*(1+alpha)*(1+2*alpha)*s1))/((3+2*alpha)*(5+2*alpha)*(7+2*alpha))
    if Ng == 2:
        mm2 = (5+2*alpha+2*s1+6*alpha*s1+4*alpha**2*s1)/(15+16*alpha+4*alpha**2)
        mm4 = ((27+6*alpha)*(7+2*alpha+(4+4*alpha)*(1+2*alpha)*s1)+
              4*(1+alpha)*(2+alpha)+(1+2*alpha)*(3+2*alpha)*s2*gamma(1.5+alpha))/(16*gamma(5.5+alpha))
    elif Ng == 3:
        mm2 = (5+2*alpha+2*s1+6*alpha*s1+4*alpha**2*s1)/(15+16*alpha+4*alpha**2)
        mm4 = (3*(9+2*alpha)*(7+2*alpha+4*(1+a)*(1+2*alpha)*s1) + 
              4*(1+alpha)*(2+alpha)*(1+2*alpha)*(3+2*alpha)*s2)/((3+2*alpha)*(5+2*alpha)*(7+2*alpha)*(9+2*alpha))
    
    # gm6  = A*2**(3-2*alpha)*np.sqrt(np.pi)*gamma(4+alpha)/((7+2*alpha)*(9+2*alpha)*(11+2*alpha)*(13+2*alpha)*gamma(0.5+alpha))
    # gm8  = 105*2**(-3-2*alpha)*(1+2*alpha)*np.sqrt(np.pi)*gamma(2+alpha)/gamma(6.5+alpha)
    # gm10 = 4725*2**(-6-2*alpha)*(1+2*alpha)*np.sqrt(np.pi)*gamma(2+alpha)/gamma(7.5+alpha)
    
    ev2 = (mu/2)**(-25/(6*4*np.pi)*alpha_s*C_F)
    ev4 = (mu/2)**(-91/(15*4*np.pi)*alpha_s*C_F)
    ev_facs = [1,0,ev2, 0, ev4]

    m_moms = [1, 0, mm2, 0, mm4,]
    lam = bz*Pz
    h_tw2 = 0
    if init_char == "Z5":
        for n in range(0,N_max+1):
            term1 = (-1j*lam/2)**n/factorial(n)
            term2 = 0
            for m in range(0,n+1):
                term2 += m_coeff_Z5(n,m, bz, alpha_s, mu)*m_moms[m]*ev_facs[m]
            h_tw2 += term1*term2

    elif init_char == "T5":
        for n in range(0,N_max+1):
            term1 = (-1j*lam/2)**n/factorial(n)
            term2 = 0
            for m in range(0,n+1):
                term2 += m_coeff_T5(n,m, bz, alpha_s, mu)*m_moms[m]*ev_facs[m]
            h_tw2 += term1*term2
    
    l_corr = l*(Pz*a)**2
    h_corr = 0
    hs = [h0,h1]
    for i in range(0, N_ht):
        h_corr += bz**2*hs[i]*F_n(i, bz, Pz, init_char, 0, mu)

    return h_tw2 + l_corr + h_corr


# z_range = np.arange(0,10,0.01)
# mu_range = 2*np.exp(-gamma_E)/(z_range/a)
# # plt.plot(z_range,m_coeff_T5(2,0,z_range/a,alpha_func(mu_range), mu_range)*(alpha_func(mu_range)/alpha_0)**(-25/(6*4*np.pi*beta_0)*C_F), label="C(2,0)")

# # plt.plot(z_range,m_coeff_T5(4,0,z_range/a,alpha_func(mu_range), mu_range)*(alpha_func(mu_range)/alpha_0)**(-91/(15*4*np.pi*beta_0)*C_F), label="C(4,0)")

# # plt.plot(z_range,
# #          2+m_coeff_T5(2,0,z_range/a,alpha_func(mu_range), mu_range) - 
# #          m_coeff_T5(0,0,z_range/a,alpha_func(mu_range), mu_range), 
# #          label="1+C(2,0)-C(0,0)")
# plt.plot(z_range, alpha_func(mu_range))
# # plt.ylim(0, 2)
# plt.legend()
# plt.xlabel("z/a")
# plt.ylabel(r"$\alpha_s$")
# plt.title("Evolution of $alpha_s$")
# plt.savefig("alpha_ev.png")
# plt.show()

### Definitions for Conformal OPE ###
def c_ope(bz, an2, an4, an6, an8, l, h0, h1, N_max, N_ht, Pz, alpha, mu, init_char):
    """
    The conformal OPE for h_tw2. F_n are the conformal partial waves
    and a_n are the Gegenbauer moments. a_n will be the fitting parameters. Eq. 
    (17) from  Gao et al. 2022, arxiv:2206.04084v2
    """
    h_tw2 = 0
    ans = [1, 0, an2, 0, an4, 0, an6, 0, an8]
         # 0, 1,   2, 3,   4, 5
    for n in range(0,N_max+1,2):
        h_tw2 += ans[n]*F_n(n, bz, Pz, init_char, alpha, mu)

    l_corr = l*(Pz*a)**2
    h_corr = 0
    hs = [h0,h1]
    for i in range(0, N_ht):
        h_corr += bz**2*hs[i]*F_n(i, bz, Pz,init_char, 0 , mu)

    return h_tw2 + l_corr + h_corr

def c_n(n, init_char, alpha_s):
    if init_char == "Z5":
        delta = 1
    else:
        delta = 0
    Hn = sum(1/i for i in range(1,n+1))
    Hn2 = sum(1/i**2 for i in range(1,n+1))
    cn = 1 + (alpha_s*C_F/(2*np.pi))*(((5+2*n)/(2+3*n+n**2)) 
                                  + (2*delta)/(2+3*n+n**2)
                                  + 2*(1-Hn)*Hn- 2*Hn2
                                  )
    return cn

def F_n(n, bz, Pz, init_char, alpha_s, mu):
    """
    Expression for the conformal partial waves. Eq. 
    (18) from  Gao et al. 2022, arxiv:2206.04084v2
    """
    cn = c_n(n, init_char, alpha_s)
    Hn1 = sum(1/i for i in range(1,n+2))
    lam = bz*Pz/2

    gamma_n = alpha_s*C_F/(4*np.pi)*(4*Hn1 - (2/(n**2+3*n+2))-3)
    gamma_O = 3*alpha_s*C_F/(4*np.pi)

    Fn = (cn * (mu**2*bz**2)**(gamma_n+gamma_O) * (gamma(2-gamma_n)*gamma(1+n))/gamma(1+n+gamma_n)
        * 0.375 * 1j**n * np.sqrt(np.pi) * (n**2+3*n+2) * gamma(n+gamma_n+2.5)/gamma(n+2.5)
        * (lam/2)**(-1.5-gamma_n)*jv(n+gamma_n+1.5,lam)
    )
    return Fn

### Modified solver for z-dependence ###
def chi2(params:list, bzPz_range:np.array, delta:float, cov_inv:np.array, ratio, matrix_els):
    bz, Pz = bzPz_range
    chi2 = 0
    for i, bzPz in enumerate(bzPz_range.transpose()):
        bz, Pz = bzPz
        for j, bzPz1 in enumerate(bzPz_range.transpose()):
            bz1, Pz1 = bzPz1
            sns = params[0:2]
            chi2 += ((ratio(bz, Pz, *params) - matrix_els[Pz, bz])*cov_inv[i,j]*(ratio(bz1, Pz1, *params) - matrix_els[Pz1, bz1]))
    chi2 += np.sum((sns/delta)**2)
    # print(chi2)
    return chi2

def chi2_orig(params:list, bzPz_range:np.array, cov:np.array, ratio, matrix_els):
    bz, Pz = bzPz_range
    cov_inv = np.linalg.inv(cov)
    chi2 = 0
    for i, bzPz in enumerate(bzPz_range.transpose()):
        bz, Pz = bzPz
        for j, bzPz1 in enumerate(bzPz_range.transpose()):
            bz1, Pz1 = bzPz1
            sns = params[0:2]
            chi2 += ((ratio(bz, Pz, *params) - matrix_els[bz, Pz])*cov_inv[i,j]*(ratio(bz1, Pz1, *params) - matrix_els[bz1, Pz1]))
    # print(chi2)
    return chi2

def phi(u, alpha, s1, s2):
    A = 2**(1+2*alpha)*gamma(1.5+alpha)/(np.sqrt(np.pi)*gamma(1+alpha))
    y = u**alpha*(1-u)**alpha*(1+
                                s1*gegenbauer(2,0.5+alpha)(1-2*u)+
                                s2*gegenbauer(4,0.5+alpha)(1-2*u))
    return A*y
    
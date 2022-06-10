# -*- coding: utf-8 -*-
"""
Created on Mon May  2 13:29:49 2022

@author: ievdv
"""
import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
import func_univariate
import pandas as pd

def aic_bic(logL, k, n):
    AIC = 2*k - 2*logL
    BIC = k*np.log(n) - 2*logL

    return AIC, BIC

def func_aic_bic(k,dat,var1):
    for i in range(len(var1)):
        n = len(dat[var1[i]])  # length of observations
    
        par_nrm_tsmp = st.norm.fit(dat[var1[i]])
        par_exp_tsmp = st.expon.fit(dat[var1[i]])

        logL_norm = np.sum(st.norm.logpdf(
            dat[var1[1]], par_nrm_tsmp[0], par_nrm_tsmp[1]))
        logL_exp = np.sum(st.expon.logpdf(
            dat[var1[1]], par_exp_tsmp[0], par_exp_tsmp[1]))

        AIC_norm, BIC_norm = aic_bic(logL_norm, k, n)
        AIC_exp, BIC_exp = aic_bic(logL_exp, k, n)

        n_obs = n + 1
        dat_sorted = np.sort(dat[var1[i]])
    
        ecdf_Q = np.linspace(1, len(dat[var1[i]]), len(dat[var1[i]]))/(n_obs)
        f_Q = st.norm.cdf(dat_sorted, loc=par_nrm_tsmp[0], scale=par_nrm_tsmp[1])

        plt.figure(figsize=(14, 8))
        x = np.linspace(0, 0.12)
        plt.plot(ecdf_Q, f_Q, color='r', linestyle='None', marker='o', markersize=0.1)
        plt.plot([0, 1], [0, 1], '--', color='k')
        plt.xlabel('Empirical quantiles')
        plt.ylabel('Theoretical quantiles')
        plt.title(f'{var1[i]} QQ-plot')
        plt.grid(linestyle='--', alpha=.3)
        plt.show()
    
    return


def func_prob(dat, var1, qntl): 
    x = np.linspace(min(dat[var1[0]]), max(dat[var1[0]]), len(dat[var1[0]]))
    xx = np.linspace(min(dat[var1[1]]), max(dat[var1[1]]), len(dat[var1[1]]))

    dat_q = dat.quantile(qntl)
    
    x1, f1 = func_univariate.ecdf(dat[var1[0]])
    x2, f2 = func_univariate.ecdf(dat[var1[1]])
    
    p_tssh = 1-f1[np.where(x1 >= dat_q[0])[0][0]]
    p_tsmp = 1-f2[np.where(x2 >= dat_q[1])[0][0]]
    
    emp = [p_tssh, p_tsmp]
    
    prob = pd.DataFrame(index= ['Empirical', 'Gaussian','Exponential', 
                                'Lognormal', 'Logistic'],
                        columns= [f'P({var1[0]} >= {str(round(dat_q[0], 2))})',
                                  f'P({var1[1]} >= {str(round(dat_q[1], 2))})'
                                  ])
    
    for i in range(len(var1)):
        par_nrm_tssh = st.norm.fit(dat[var1[i]])
        par_exp_tssh = st.expon.fit(dat[var1[i]])
        par_logn_tssh = st.lognorm.fit(dat[var1[i]])
        par_logi_tssh = st.logistic.fit(dat[var1[i]])

        # Create a linspace vector for empirical distributions
        x = np.linspace(min(dat[var1[i]]), max(dat[var1[i]]), len(dat[var1[i]]))  

        rdm_nrm_tssh = st.norm.cdf(x, loc=par_nrm_tssh[0], scale=par_nrm_tssh[1])
        rdm_exp_tssh = st.expon.cdf(x, par_exp_tssh[0], scale=par_exp_tssh[1])
        rdm_logn_tssh = st.lognorm.cdf(x, s=par_logn_tssh[0],
                                       loc=par_logn_tssh[1], scale=par_logn_tssh[2])
        rdm_logi_tssh = st.logistic.cdf(x, loc=par_logi_tssh[0],
                                        scale=par_logi_tssh[1])
        
        p_nrm_tssh = 1 - rdm_nrm_tssh[np.where(x >= dat_q[i])[0][0]]
        p_exp_tssh = 1 - rdm_exp_tssh[np.where(x >= dat_q[i])[0][0]]
        p_logn_tssh = 1 - rdm_logn_tssh[np.where(x >= dat_q[i])[0][0]]
        p_logi_tssh = 1 - rdm_logi_tssh[np.where(x >= dat_q[i])[0][0]]

        prob.iloc[0][i] = emp[i]
        prob.iloc[1][i] = p_nrm_tssh
        prob.iloc[2][i] = p_exp_tssh
        prob.iloc[3][i] = p_logn_tssh
        prob.iloc[4][i] = p_logi_tssh
    
    return prob.T, dat_q
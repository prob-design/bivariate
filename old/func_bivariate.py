# -*- coding: utf-8 -*-
"""
Created on Mon May  2 16:49:33 2022

@author: ievdv
"""
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import numpy as np
import scipy.stats as st
import pandas as pd

def plot(dat,var1,var_lab):
    plt.figure(figsize=(6, 6))
    plt.plot(dat[var1[0]], dat[var1[1]], '.b')
    plt.xlabel(var_lab[0])
    plt.ylabel(var_lab[1])
    plt.grid(linestyle='--', linewidth=.5)
    plt.show()

    h = sns.jointplot(data=dat, x=var1[0], y=var1[1])
    h.set_axis_labels(var_lab[0], var_lab[1])
    h.figure.tight_layout()

    g = sns.displot(data=dat, x=var1[0], y=var1[1], kind='kde')
    g.set_axis_labels(var_lab[0], var_lab[1])
    g.figure.tight_layout()

    plt.show()
    
    return
    
def cov_cor(dat,var1):
    covariance = np.cov(dat[var1[0]], dat[var1[1]])
    corr, _ = pearsonr(dat[var1[0]], dat[var1[1]])
    
    return covariance, corr

def fit(dat,var1,var_lab):
    dat1 = dat[var1]  # Create a data frame without the timestamp

    mean = np.mean(dat1, axis=0)
    cov = np.cov(dat1, rowvar=0)

    N = dat1.shape[0]  # Number of samples
    # Draw random samples from multivariate normal
    r_norm = st.multivariate_normal.rvs(mean, cov, N)  
    df_r_norm = pd.DataFrame(r_norm, columns=(var1[0], var1[1]))

    df_r_norm.head()

    plt.figure(figsize=(6, 6))
    plt.scatter(r_norm[:, 0], r_norm[:, 1])  # Bivariate normal simulations
    plt.xlabel(var_lab[0])
    plt.ylabel(var_lab[1])
    plt.show()

    h1 = sns.jointplot(data=df_r_norm, x=var1[0], y=var1[1])
    h1.set_axis_labels(var_lab[0], var_lab[1])
    h1.figure.tight_layout()
    h1.fig.suptitle('Bivariate fit')
    h1.fig.subplots_adjust(top=0.95)

    g1 = sns.displot(data=df_r_norm, x=var1[0], y=var1[1], kind='kde')
    g1.set_axis_labels(var_lab[0], var_lab[1])
    g1.figure.tight_layout()
    
    plt.figure(figsize=(10, 10))
    plt.scatter(dat1[var1[0]], dat1[var1[1]], label='Empirical Data')
    plt.scatter(r_norm[:, 0], r_norm[:, 1], label='Bivariate Normal Data')
    plt.xlabel(var_lab[0])
    plt.ylabel(var_lab[1])
    plt.legend()
    plt.show()
    
    mean_lnorm = np.log(np.mean(dat1, axis=0))
    cov = np.cov(dat1, rowvar=0)

    N = dat1.shape[0]
    # r_lnorm = np.exp(st.multivariate_normal.rvs(mean_lnorm, cov, N))
    # df_r_norm = pd.DataFrame(r_lnorm, columns=(var1[0], var1[1]))

    # df_r_norm.head()

    # fig, ax = plt.subplots(figsize = (6, 6))

    # ax.scatter(r_lnorm[:,0], r_lnorm[:,1])  # Bivariate lognormal simulations
    # ax.set_xscale('log')
    # ax.set_yscale('log')
    # ax.set_xlabel(var_lab[0])
    # ax.set_ylabel(var_lab[1])
    # plt.show()

    # h1=sns.jointplot(data=df_r_norm, x=var1[0], y=var1[1])
    # h1.set_axis_labels(var_lab[0], var_lab[1]) 
    # h1.figure.tight_layout()

    # g1=sns.displot(data=df_r_norm, x=var1[0], y=var1[1], kind='kde')
    # g1.set_axis_labels(var_lab[0], var_lab[1])    
    # g1.figure.tight_layout()
    # plt.show()

    return dat1, df_r_norm

def func_and_or(AND, OR, data, fit, var_lab, var1, title):
    fig3, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(20, 5))
    ax[0].scatter(data[var1[0]], data[var1[1]], edgecolor='w')
    ax[0].scatter(AND[var1[0]], AND[var1[1]], edgecolor='w')
    ax[0].hlines(fit[1], fit[0], max(data[var1[0]]), 'k', linewidth=2)
    ax[0].vlines(fit[0], fit[1], max(data[var1[1]]), 'k', linewidth=2)
    ax[0].set_title('AND scenario')
    ax[0].set_xlabel(var_lab[0])
    ax[0].set_ylabel(var_lab[1])
    ax[0].grid(linestyle='--', alpha=.3)

    ax[1].scatter(data[var1[0]], data[var1[1]], edgecolor='w')
    ax[1].scatter(OR[var1[0]], OR[var1[1]], edgecolor='w')
    ax[1].hlines(fit[1], 0, fit[0], 'k', linewidth=2)
    ax[1].vlines(fit[0], 0, fit[1], 'k', linewidth=2)
    ax[1].set_title('OR scenario')
    ax[1].set_xlabel(var_lab[0])
    ax[1].set_ylabel(var_lab[1])
    ax[1].grid(linestyle='--', alpha=.3)
    plt.suptitle(title)
    plt.show()

    return

def probabilities(dat1, var1, dat, dat_q, qntl, var_lab, df_r_norm):
    AND_SC = dat1[(dat1[var1[0]] >= dat_q[0]) & (dat1[var1[1]] >= dat_q[1])]
    OR_SC = dat1[(dat1[var1[0]] >= dat_q[0]) | (dat1[var1[1]] >= dat_q[1])]

    func_and_or(AND_SC, OR_SC, dat, dat_q, var_lab, var1, 'Empirical data')
    
    fit_q = df_r_norm.quantile(qntl)  # Compute quantile for bivariate normal data

    ANDf_SC = df_r_norm[(df_r_norm[var1[0]] >= fit_q[0]) &
                        (df_r_norm[var1[1]] >= fit_q[1])]
    ORf_SC = df_r_norm[(df_r_norm[var1[0]] >= fit_q[0]) |
                       (df_r_norm[var1[1]] >= fit_q[1])]

    func_and_or(ANDf_SC, ORf_SC, df_r_norm, fit_q, var_lab, var1, 
                'Bivariate normal data')

    P_emp_AND = AND_SC.shape[0]/dat1.shape[0]
    P_fit_AND = ANDf_SC.shape[0]/dat1.shape[0]

    P_emp_OR = OR_SC.shape[0]/dat1.shape[0]
    P_fit_OR = ORf_SC.shape[0]/dat1.shape[0]

    P_biv = pd.DataFrame(np.array([[P_fit_OR, P_fit_AND], [P_emp_OR, P_emp_AND]])).transpose()
    P_biv.columns = ['Empirical', 'Bivariate Normal']
    P_biv.index = ['AND', 'OR']

    P_biv.head()
    
    return

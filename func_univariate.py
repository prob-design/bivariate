# -*- coding: utf-8 -*-
"""
Created on Mon May  2 13:00:19 2022

@author: ievdv
"""
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st
from statsmodels.distributions.empirical_distribution import ECDF

def func_ecdf(var, color, var_name):
    fig, ax = plt.subplots(1, 3, sharex=True, figsize=(18, 5))
    plt.suptitle('Python ECDF', y=1.02)
    ax[0].step(var.x, var.y, color=color, linewidth=4)  # Plot F(x)
    ax[0].set_title('$F(x)$')
    ax[0].set_xlabel(var_name)
    ax[0].set_ylabel('Cumulative Probability')

    ax[1].step(var.x, 1-var.y, color=color, linewidth=4)  # Plot 1-F(x)
    ax[1].set_xlabel(var_name)
    ax[1].set_title('1-$F(x)$')

    ax[2].semilogy(var.x, 1-var.y, color=color,
                   linewidth=4)  # Plot logY 1-F(x)
    ax[2].set_xlabel(var_name)
    ax[2].set_title('1-$F(x)$. Y axis log scale')
    plt.show()

    return

def func_ecdf2(var1,var2,color1,color2,var_name,xlabel):
    fig, ax = plt.subplots(1, 3, sharex=True, figsize=(18, 5))
    plt.suptitle('Python ECDF', y=1.02)
    ax[0].step(var1.x, var1.y, color=color1, linewidth=4,label=var_name[0])  # Plot F(x)
    ax[0].step(var2.x, var2.y, color=color2, linewidth=4,label=var_name[1])  # Plot F(x)
    ax[0].set_title('$F(x)$')
    ax[0].legend()
    ax[0].set_xlabel(xlabel)
    ax[0].set_ylabel('Cumulative Probability')

    ax[1].step(var1.x, 1-var1.y, color=color1, linewidth=4,label=var_name[0])  # Plot 1-F(x)
    ax[1].step(var2.x, 1-var2.y, color=color2, linewidth=4,label=var_name[1])  # Plot 1-F(x)
    ax[1].legend()
    ax[1].set_xlabel(xlabel)
    ax[1].set_title('1-$F(x)$')

    ax[2].semilogy(var1.x, 1-var1.y, color=color1,
                   linewidth=4,label=var_name[0])  # Plot logY 1-F(x)
    ax[2].semilogy(var2.x, 1-var2.y, color=color2,
                   linewidth=4,label=var_name[1]) 
    ax[2].set_xlabel(xlabel)
    ax[2].set_title('1-$F(x)$. Y axis log scale')
    plt.show()

    return

def ecdf(y):
    x = np.sort(y)
    n = x.size
    f = np.arange(1, n+1) / n
    return(x, f)

def func_ecdf_emp(var1, var2, color, var_name):
    fig, ax = plt.subplots(1, 3, sharex=True, figsize=(18, 5))
    plt.suptitle('User defined ECDF', y=1.02)
    ax[0].step(var1, var2, color=color, linewidth=4)
    ax[0].set_title('$F(x)$')
    ax[0].set_xlabel(var_name)
    ax[0].set_ylabel('Cumulative Probability')

    ax[1].step(var1, 1-var2, color=color, linewidth=4)  # Plot 1-F(x)
    ax[1].set_xlabel(var_name)
    ax[1].set_title('1-$F(x)$')

    ax[2].semilogy(var1, 1-var2, color=color, linewidth=4)  # Plot logY 1-F(x)
    ax[2].set_xlabel(var_name)
    ax[2].set_title('1-$F(x)$. Y axis log scale')
    plt.show()

    return

def func_ecdf_emp2(var1,p1, var2, p2,color1,color2,var_name,xlabel):
    fig, ax = plt.subplots(1, 3, sharex=True, figsize=(18, 5))
    plt.suptitle('User defined ECDF', y=1.02)
    ax[0].step(var1, var2, color=color1, linewidth=4,label=var_name[0])
    ax[0].step(p1, p2, color=color2, linewidth=4,label=var_name[1])
    ax[0].set_title('$F(x)$')
    ax[0].set_xlabel(xlabel)
    ax[0].legend()
    ax[0].set_ylabel('Cumulative Probability')

    ax[1].step(var1, 1-var2, color=color1, linewidth=4,label=var_name[0])  # Plot 1-F(x)
    ax[1].step(p1,1-p2,color=color2,linewidth=4,label=var_name[1])
    ax[1].set_xlabel(xlabel)
    ax[1].legend()
    ax[1].set_title('1-$F(x)$')

    ax[2].semilogy(var1, 1-var2, color=color1, linewidth=4,label=var_name[0])  # Plot logY 1-F(x)
    ax[2].semilogy(p1, 1-p2, color=color2, linewidth=4,label=var_name[1])
    ax[2].set_xlabel(xlabel)
    ax[2].legend()
    ax[2].set_title('1-$F(x)$. Y axis log scale')
    plt.show()

    return

def emprical(var1, dat, xlabel, plot=False, together=False):
    empcdf = ECDF(dat[var1[0]])  # ECDF function from Python
    empcdf1 = ECDF(dat[var1[1]])

    x1, f1 = ecdf(dat[var1[0]])
    x2, f2 = ecdf(dat[var1[1]])
    
    if plot==True:
        if together==False:
            func_ecdf(empcdf, 'royalblue', var1[0])
            func_ecdf(empcdf1, 'violet', var1[1])
    
            func_ecdf_emp(x1, f1, 'royalblue', var1[0])
            func_ecdf_emp(x2, f2, 'violet', var1[1])

            plt.step(empcdf.x[1:50], 1-empcdf.y[1:50])  # Zoomed version of ECDF
            plt.semilogy
            plt.xlabel(var1[0])
            plt.title('1-$F(x)$')
            plt.show()
        else:
            func_ecdf2(empcdf,empcdf1,'royalblue','violet',var1,xlabel)
    
            func_ecdf_emp2(x1,x2, f1,f2, 'royalblue','violet', var1,xlabel)

            plt.step(empcdf.x[1:50], 1-empcdf.y[1:50])  # Zoomed version of ECDF
            plt.semilogy
            plt.xlabel(var1[0])
            plt.title('1-$F(x)$')
            plt.show()
    
    return x1, f1, x2, f2
    

def func_dist(x, x1, f1, norm, expon, logn, logi, xlabel):
    fig1, ax1 = plt.subplots(1, 4, sharex=True, figsize=(20, 5))
    ax1[0].semilogy(x1, 1-f1, label='Empirical Data',
                    linestyle='None', marker='o', markersize=2)
    ax1[0].semilogy(x, 1-norm, color='violet', label='Fitted Data', 
                    linestyle='None', marker='o', markersize=2)
    ax1[0].set_xlabel(xlabel)
    ax1[0].set_ylabel('Cumulative Probability')
    ax1[0].set_title('Fit Gaussian Distribution')
    ax1[0].legend()

    ax1[1].semilogy(x1, 1-f1, label='Empirical Data',
                    linestyle='None', marker='o', markersize=2)
    ax1[1].semilogy(x, 1-expon, color='violet', label='Fitted Data', 
                    linestyle='None', marker='o', markersize=2)
    ax1[1].set_xlabel(xlabel)
    ax1[1].set_title('Fit Exponential Distribution')
    ax1[1].legend()

    ax1[2].semilogy(x1, 1-f1, label='Empirical Data',
                    linestyle='None', marker='o', markersize=2)
    ax1[2].semilogy(x, 1-logn, color='violet', label='Fitted Data', 
                    linestyle='None', marker='o', markersize=0.1)
    ax1[2].set_xlabel(xlabel)
    ax1[2].set_title('Fit Lognormal Distribution')
    ax1[2].legend()

    ax1[3].semilogy(x1, 1-f1, label='Empirical Data',
                    linestyle='None', marker='o', markersize=2)
    ax1[3].semilogy(x, 1-logi, color='violet', label='Fitted Data', 
                    linestyle='None', marker='o', markersize=2)
    ax1[3].set_xlabel(xlabel)
    ax1[3].set_title('Fit Logistic Distribution')
    ax1[3].legend()
    plt.show()

    return


def func_all_dist(x, x1, f1, norm, expon, logn, logi, var_lab):
    plt.figure(figsize=(8, 5))  # Plot all the distributions together
    plt.plot(x1, 1-f1, label="Empirical Distribution",
              color='k', linestyle='dashed')
    plt.plot(x, 1-norm, label="Gaussian Distribution")
    plt.plot(x, 1-expon, label="Exponential Distribution")
    plt.plot(x, 1-logn, label="Lognormal Distribution")
    plt.plot(x, 1-logi, label="Logistic Distribution")
    plt.yscale('log')
    plt.xlabel(var_lab)
    plt.ylabel("Cumulative Probability")
    plt.legend()
    plt.show()

    return


def theoretical_distribution(var1, dat, var_lab,xlabel):
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
        
        x_emp1, f_emp1, x_emp2, f_emp2 = emprical(var1,dat,xlabel)
        x_emp = [x_emp1,x_emp2]
        f_emp = [f_emp1,f_emp2]
        
        x1 = x_emp[i]
        f1 = f_emp[i]
    
        func_dist(x, x1, f1, rdm_nrm_tssh, rdm_exp_tssh,
                  rdm_logn_tssh, rdm_logi_tssh, var1[i])
        func_all_dist(x, x1, f1, rdm_nrm_tssh, rdm_exp_tssh,
                      rdm_logn_tssh, rdm_logi_tssh, var_lab[i])
    
    return

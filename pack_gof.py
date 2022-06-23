# -*- coding: utf-8 -*-
"""
Created on Fri May 27 16:14:15 2022

@author: jelle
"""

import numpy as np
import matplotlib.pyplot as plt

import pack_univariate as univ
import pack_helpers as helpers


def aic_bic(pdf, k, n):
    """Calculates the AIC- and BIC-value of a fitted distribution
    Arguments:
        LogL (float): The sum of the log of the pdf of the fitted distribution
        k (int): The number of parameters in the distribution
        n (int): The number of data points
    Returns:
        The AIC- and BIC-scores
    """
    
    logL = np.sum(np.log10(pdf))
    AIC = 2*k - 2*logL
    BIC = k*np.log(n) - 2*logL

    return AIC, BIC


def aic_bic_fit(var, distribution):
    """Calculates the AIC and BIC of a fitted distribution for a given variable
    Arguments:
        var (series): the variable
        distribution (str): the distribution to fit on the variable
    Returns:
        The AIC- and BIC-scores
    """
    pars, cdf = univ.fit_distribution(var, distribution, plot=False)
    dist = helpers.scipy_dist(distribution)

    logL = dist.pdf(var, *pars)
    k = len(pars)
    n = len(var)

    AIC, BIC = aic_bic(logL, k, n)

    return AIC, BIC


def QQ_plot(var, distribution, **kwargs):
    """Creates a QQ-plot of a given variable and distribution
    Arguments:
        var (series): the variable
        distribution (str): the distribution to fit on the variable
    """
    pars, cdf = univ.fit_distribution(var, distribution, plot=False)
    dist = helpers.scipy_dist(distribution)
    n = len(var)
    var_sorted = np.sort(var)

    ecdf_Q = np.linspace(1, n, n)/(n + 1)
    f_Q = dist.cdf(var_sorted, *pars)

    fig, ax = plt.subplots(figsize=(14, 8))
    ax.plot([0, 1], [0, 1], '--', color='k')
    ax.plot(ecdf_Q, f_Q, **kwargs)
    ax.set_xlabel('Empirical quantiles')
    ax.set_ylabel('Theoretical quantiles')
    ax.set_title(f"QQ-plot of fitted {distribution} distribution")
    ax.grid()
    plt.show()


def quantile_compare(var, distribution, quantile):
    """Function to compare quantiles of a variable and a distribution
    Arguments:
        var (series): the variable
        distribution (str): the distribution to fit on the variable
        quantile: the quantiles to be compared
    Returns:
        x_emp: the empirical value of the given quamtile
        x_fitted: the value of the fitted distribution for the given quantile
        q_fitted: the CDF of the fitted distribution at the empirical quantile
    """
    x_emp = var.quantile(quantile)
    pars, cdf = univ.fit_distribution(var, distribution, plot=False)
    dist = helpers.scipy_dist(distribution)
    x_fitted = dist.ppf(quantile, *pars)
    q_fitted = dist.cdf(x_emp, *pars)
    if type(quantile) == list or tuple:
        x_emp = x_emp.to_numpy()
    return x_emp, x_fitted, q_fitted

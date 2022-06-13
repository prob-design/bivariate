# -*- coding: utf-8 -*-
"""
Created on Thu May 26 13:35:02 2022

@author: jelle
"""

import matplotlib.pyplot as plt
import numpy as np
import pack_helpers as helpers


def ecdf(var):
    """Returns the empirical cumulative distribution of a given variable"""
    x = np.sort(var)
    n = x.size
    f = np.arange(1, n+1) / n
    return x, f


def plot_ecdf(var, label=None, **kwargs):
    """Plots the empirical cumulative distribution function of a variable
    Arguments:
        var (series): the variable
        label (str): optional label to put in the title
    """
    x, f = ecdf(var)
    fig, ax = plt.subplots(1, 4, sharex=True, figsize=(24, 5))
    if label:
        plt.suptitle(f'ECDF of {label}', y=0.99)

    ax[0].step(x, f, linewidth=4, **kwargs)  # Plot F(x)
    ax[0].set_title('$F(x)$')
    ax[0].set_ylabel('Cumulative Probability')
    ax[0].grid()

    ax[1].step(x, 1 - f, linewidth=4, **kwargs)  # Plot 1-F(x)
    ax[1].set_title('$1- F(x)$')
    ax[1].grid()

    ax[2].semilogy(x, f, linewidth=4, **kwargs)  # Plot logY 1-F(x)
    ax[2].set_title('$F(x)$. Y axis log scale')
    ax[2].grid()

    ax[3].semilogy(x, 1 - f, linewidth=4, **kwargs)  # Plot logY 1-F(x)
    ax[3].set_title('$1- F(x)$. Y axis log scale')
    ax[3].grid()
    plt.show()

    return


def fit_distribution(var, distribution="Normal", plot=True, label=None, 
                     **kwargs):
    """Fits and optionally plots a variable to a given distribution
    Arguments:
        var (series): the variable
        distribution (str): the distribution to fit on the variable
        plot (bool): whether to plot the empirical and fitted distribution
        label (str): optional label to put in the title of the plot
    Returns:
        parameters of the fitted distribution
        fitted cdf of the distribution F(x)
    Currently supported distributions:
        Normal, Exponential, Lognormal, Logistic, Extreme Values
        Function checks the first four characters of the inputted string
        (case independent)
    """
    x, f = ecdf(var)

    dist = helpers.scipy_dist(distribution)

    fit_pars = dist.fit(var)
    fit_cdf = dist.cdf(x, *fit_pars)

    if plot:
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.plot(x, f, label="Empirical Distribution", **kwargs)
        ax.plot(x, fit_cdf, label=f"Fitted {distribution} distribution",
                **kwargs)
        ax.set_xlabel("Value")
        ax.set_ylabel("F(X)")
        if label:
            plt.suptitle(f'CDF of {label}')
        ax.legend()
        ax.grid()
        plt.show()

    return fit_pars, fit_cdf


def plot_distributions(var, seperate=True, label=None, **kwargs):
    """Plots fitted distributions on a given variable in a single figure
    Currently uses Normal, Exponential, Lognormal, Logistic distributions
    Arguments:
        var (series): the variable
        separate (bool): whether to plot the distributions in seperate plots
        label (str): optional label to put in the title of the plot
    """
    x, f_emp = ecdf(var)
    f_norm = fit_distribution(var, "Normal", plot=False)[1]
    f_exp = fit_distribution(var, "Exponential", plot=False)[1]
    f_lognorm = fit_distribution(var, "Lognormal", plot=False)[1]
    f_logit = fit_distribution(var, "Logistic", plot=False)[1]

    fitted = [f_norm, f_exp, f_lognorm, f_logit]
    dist_names = ["Normal", "Exponential", "Lognormal", "Logistic"]

    if seperate:
        fig, ax = plt.subplots(1, 4, sharex=True, sharey=True, figsize=(20, 6))

        for i in range(len(fitted)):
            ax[i].plot(x, f_emp, label="Empirical Distribution", **kwargs)
            ax[i].plot(x, fitted[i],
                       label=f"Fitted {dist_names[i]} distribution", **kwargs)
            ax[i].set_xlabel("Value")
            ax[i].grid()
            ax[i].legend(bbox_to_anchor=(0.5, -0.1), loc='upper center')
        ax[0].set_ylabel("F(X)")
        # plt.tight_layout()

    else:
        fig, ax = plt.subplots(sharex=True, sharey=True, figsize=(20, 10))
        ax.plot(x, f_emp, label="Empirical Distribution", **kwargs)
        for i in range(len(fitted)):
            ax.plot(x, fitted[i], label=f"Fitted {dist_names[i]} distribution",
                    **kwargs)
        ax.set_xlabel("Value")
        ax.set_ylabel("F(X)")
        ax.legend(bbox_to_anchor=(0.5, -0.1), ncol=len(fitted) + 1, 
                  loc='upper center')
        plt.grid()

    if label:
        plt.suptitle(f"CDF's of {label}")

    plt.show()

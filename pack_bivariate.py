# -*- coding: utf-8 -*-
"""
Created on Fri Jun  3 14:06:37 2022

@author: jelle
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats as st
import pandas as pd


def bivar_plot(df, vars_used, labels=None):
    """Creates several plots of two variables from a dataframe
    Arguments:
        df (dataframe): the dataframe to use
        vars_used (list of 2 str): the columns containing the variables
        labels (optional list of 2 str): labels to use in the plots
    """
    plt.figure(figsize=(6, 6))
    plt.plot(df[vars_used[0]], df[vars_used[1]], ".")
    plt.grid()
    if labels:
        plt.xlabel(labels[0])
        plt.ylabel(labels[1])
    plt.show()

    h = sns.jointplot(data=df, x=vars_used[0], y=vars_used[1])
    if labels:
        h.set_axis_labels(labels[0], labels[1])
    h.figure.tight_layout()

    g = sns.displot(data=df, x=vars_used[0], y=vars_used[1], kind='kde')
    if labels:
        g.set_axis_labels(labels[0], labels[1])
    g.figure.tight_layout()

    plt.show()


def cov_cor(df, vars_used):
    """Calculates the covariance and Pearson's correlation coefficient of two
    variables from a dataframe.
    Arguments:
        df (dataframe): the dataframe to use
        vars_used (list of 2 str): the columns containing the variables
    Returns:
        the covariance and Pearson's correlation coefficient
    """
    # Docstring is kinda useless here...
    covariance = np.cov(df[vars_used[0]], df[vars_used[1]])
    corr, _ = st.pearsonr(df[vars_used[0]], df[vars_used[1]])
    return covariance, corr


def bivar_fit(df, vars_used, plot=True, labels=None, N=None):
    """Fits a bivariate normal distribution, including covariance, on two
    variable from a dataframe, and draw samples from this fit.
    Arguments:
        df (dataframe): the dataframe to use
        vars_used (list of 2 str): the columns containing the variables
        plot (bool): whether to create a plot of the fitted distribution
        labels (optional list of 2 str): labels to use in the plots
        N (int): number of samples to draw. If None, equal to number in dataset
    Returns:
        A dataframe with the drawn samples
    """
    dat = df[vars_used]  # Create a data frame without the timestamp

    mean = np.mean(dat, axis=0)
    cov = np.cov(dat, rowvar=0)

    if not N:
        N = df.shape[0]  # Number of samples = number in df if not given
    # Draw random samples from multivariate normal
    r_norm = st.multivariate_normal.rvs(mean, cov, N)
    df_r_norm = pd.DataFrame(r_norm, columns=vars_used)

    # df_r_norm.head()
    if plot:
        plt.figure(figsize=(6, 6))
        plt.scatter(r_norm[:, 0], r_norm[:, 1])  # Bivariate normal simulations
        if labels:
            plt.xlabel(labels[0])
            plt.ylabel(labels[1])
        plt.show()

        h1 = sns.jointplot(data=df_r_norm, x=vars_used[0], y=vars_used[1])
        if labels:
            h1.set_axis_labels(labels[0], labels[1])
        h1.figure.tight_layout()
        h1.fig.suptitle('Bivariate fit')
        h1.fig.subplots_adjust(top=0.95)

        g = sns.displot(data=df_r_norm, x=vars_used[0], y=vars_used[1],
                        kind='kde')
        if labels:
            g.set_axis_labels(labels[0], labels[1])
        g.figure.tight_layout()

        plt.figure(figsize=(10, 10))
        plt.scatter(dat[vars_used[0]], dat[vars_used[1]],
                    label='Empirical Data')
        plt.scatter(r_norm[:, 0], r_norm[:, 1], label='Bivariate Normal Data')
        if labels:
            plt.xlabel(labels[0])
            plt.ylabel(labels[1])
        plt.legend()
        plt.show()

    return df_r_norm


def and_or_probabilities(df, vars_used, quantiles, plot=True, title=None,
                         labels=None):
    df_quantiles = [df[vars_used[0]].quantile(quantiles[0]),
                    df[vars_used[1]].quantile(quantiles[1])]
    AND_SC = df[(df[vars_used[0]] >= df_quantiles[0]) &
                (df[vars_used[1]] >= df_quantiles[1])]
    OR_SC = df[(df[vars_used[0]] >= df_quantiles[0]) |
               (df[vars_used[1]] >= df_quantiles[1])]

    P_AND = len(AND_SC) / len(df)
    P_OR = len(OR_SC) / len(df)

    if plot:
        fig, ax = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(20, 5))
        ax[0].scatter(df[vars_used[0]], df[vars_used[1]])
        ax[0].scatter(AND_SC[vars_used[0]], AND_SC[vars_used[1]])
        ax[0].axvline(df_quantiles[0], color='k')
        ax[0].axhline(df_quantiles[1], color='k')
        ax[0].set_title(f'AND scenario, probability {P_AND:.2f}')
        if labels:
            ax[0].set_xlabel(labels[0])
            ax[0].set_ylabel(labels[1])
        ax[0].grid()

        ax[1].scatter(df[vars_used[0]], df[vars_used[1]])
        ax[1].scatter(OR_SC[vars_used[0]], OR_SC[vars_used[1]])
        ax[1].axvline(df_quantiles[0], color='k')
        ax[1].axhline(df_quantiles[1], color='k')
        ax[1].set_title(f'OR scenario, probability {P_OR:.2f}')
        if labels:
            ax[1].set_xlabel(labels[0])
            ax[1].set_ylabel(labels[1])
        ax[1].grid()

    return P_AND, P_OR

# def and_or_plot(df, vars_used, quantiles):
    

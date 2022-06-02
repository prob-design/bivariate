# -*- coding: utf-8 -*-
"""
Created on Mon May 30 15:56:47 2022

@author: jelle
"""

import pack_helpers as helpers
import pack_univariate as univ
import pack_gof as gof


def create_ev(df, period):
    """Creates a dataframe with blockwise extreme values from a given dataframe
    and period. Also drops any NaN-values
    Arguments:
        df (dataframe): the dataframe to extract extreme values from
        period (str): the period to use for blocks.
            Supported: [W]eekly, [A]nnual, [D]aily (pandas uses first char)
    Returns:
        a dataframe with the extreme values asked
    """
    time_col = helpers.find_datetime_col(df)
    extremes = df.resample(period[0].upper(),
                           on=time_col).max().dropna().reset_index(drop=True)
    return extremes


def fit_ev(ex_var, plot=True, label=None):
    """Fits an extreme value distribution to a variable of extreme values, and
    optionally plots it.
    Arguments:
        ex_var (series): the variable of extreme values
        plot (bool): whether to plot the empirical and fitted distribution
        label (str): optional label to put in the title of the plot
    Returns:
        parameters of the fitted distribution
        fitted cdf of the distribution F(x)
    Note: this function simply passes through the fit_distribution() of the
    univariate module
    """
    fit_pars, fit_cdf = univ.fit_distribution(ex_var, distribution="Extreme",
                                              plot=plot, label=label)
    return fit_pars, fit_cdf


def AIC_BIC_ev(ex_var):
    """Calculates the AIC and BIC of an extreme value distribution for a given
    variable of extreme values.
    Arguments:
        ex_var (series): the variable of extreme values
    Returns:
        The AIC- and BIC-scores
    Note: this function simply passes through the AIC_BIC_fit() of the
    univariate module
    """
    AIC, BIC = gof.aic_bic_fit(ex_var, distribution="Extreme")
    return AIC, BIC


def QQ_plot_ev(ex_var):
    """Creates a QQ-plot of given extreme values and distribution
    Arguments:
        ex_var (series): the variable of extreme values
    Note: this function simply passes through the QQ_plot() of the
    univariate module
    """
    gof.QQ_plot(ex_var, "Extreme")

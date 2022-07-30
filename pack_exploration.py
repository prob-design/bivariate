# -*- coding: utf-8 -*-
"""
Created on Wed May 25 15:23:48 2022

@author: jelle
"""

import matplotlib.pyplot as plt

from IPython.display import display

from . import pack_helpers as helpers


def data_summary(df, cols=None):
    """Displays the summary of a dataframe"""
    if not cols:
        cols = list(df.columns)
    display(df[cols].describe())


def time_plot(df, cols=None, together=False, zoom=None, **kwargs):
    
    """Plots the values of the given columns in a dataframe against time
    Arguments:
        df (dataframe): the dataframe
        cols (list of str): the cols to plot. If None (default), plots them all
        together (bool): plot everything in one figure. Default False
        zoom: (2-tuple of int): optional domain for zoomed-in plot
        **kwargs: optional arguments for ax.scatter()
    Returns:
        Figure and axes.
    """
    time_col = helpers.find_datetime_col(df)

    if not cols:
        cols = list(df.drop(columns=time_col).columns)

    figsize = (10, 10) if together else (10, 5*len(cols))
   
    ax = df.plot(x=time_col,
                 y=cols,
                 xlim=zoom,
                 subplots=not together,
                 sharex=True,
                 figsize=figsize,
                 marker='o',
                 ls='none',
                 title=cols if not together else None,
                 legend=together,
                 grid=True,
                 **kwargs)

    return plt.gcf(), ax


def hist_plot(df, cols=None, together=False, **kwargs):
    """Create a histogram of the given columns in a dataframe
    Arguments:
        df (dataframe): the dataframe
        cols (list of str): the cols to plot. If None (default), plots them all
        together (bool): plot everything in one figure. Default False
        **kwargs: optional arguments for ax.hist()
    """
    time_col = helpers.find_datetime_col(df)
    
    if not cols:
        cols = list(df.drop(columns=time_col).columns)
    
    figsize = (10, 10) if together else (10, 5*len(cols))

    ax = df.plot(y=cols,
                 kind='hist',
                 subplots=not together,
                 figsize=figsize,
                 title=cols if not together else None,
                 legend=together,
                 grid=True,
                 **kwargs)

    return plt.gcf(), ax
# -*- coding: utf-8 -*-
"""
Created on Wed May 25 15:23:48 2022

@author: jelle
"""

import matplotlib.pyplot as plt

import pack_helpers as helpers

from IPython.display import display


def data_summary(df):
    """Displays the summary of a dataframe"""
    display(df.describe())


def time_plot(df, cols=None, zoom=None):
    """Plots the values of the given columns in a dataframe against time
    Arguments:
        df (dataframe): the dataframe
        cols (list of str): the cols to plot. If None (default), plots them all
        zoom: (2-list of int): optional domain for zoomed-in plot
    """
    time_col = helpers.find_datetime_col(df)
    if not cols:
        cols = list(df.columns)
    if not zoom:
        zoom = [0, len(df)]
    for col_name in df:
        if col_name == time_col or col_name not in cols:
            continue
        fig, ax = plt.subplots(figsize=(10, 10), sharex=True)
        ax.plot(time_col, col_name, data=df.iloc[zoom[0]:zoom[1]],
                linestyle='None', marker='o', markersize=1)
        ax.set_title(col_name)
        ax.grid(True)
        ax.xaxis.set_major_locator(plt.MaxNLocator(7))
        ax.set(xlabel='Date', ylabel=col_name)
        fig.autofmt_xdate()
        plt.show()


def hist_plot(df, cols=None, bins=None):
    """Create a histogram of the given columns in a dataframe
    Arguments:
        df (dataframe): the dataframe
        cols (list of str): the cols to plot. If None (default), plots them all
        bins (int or lost): defines bins, same as plt.hist()
    """
    time_col = helpers.find_datetime_col(df)
    if not cols:
        cols = list(df.columns)
    for col_name in df:
        if col_name == time_col or col_name not in cols:
            continue
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.hist(df[col_name].values, density=False, bins=bins)
        ax.set_title(col_name)
        ax.grid(True)
        ax.xaxis.set_major_locator(plt.MaxNLocator(10))
        ax.set(xlabel=col_name, ylabel='Frequency')
        plt.show()

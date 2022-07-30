# -*- coding: utf-8 -*-
"""
Created on Wed May 25 15:23:48 2022

@author: jelle
"""

import matplotlib.pyplot as plt

from IPython.display import display


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
        zoom: (2-list of int): optional domain for zoomed-in plot
        **kwargs: optional arguments for ax.scatter()
    """
    time_col = helpers.find_datetime_col(df)
    if not cols:
        cols = list(df.columns)
    if not zoom:
        zoom = [0, len(df)]
    
    if together:
        fig, ax = plt.subplots(figsize=(10, 10), sharex=True)
        ax.set_title(cols)
        for col_name in df:
            if col_name == time_col or col_name not in cols:
                continue
            ax.scatter(time_col, col_name, data=df.iloc[zoom[0]:zoom[1]],
                       label = col_name, **kwargs)
        ax.grid(True)
        ax.set(xlabel='Date / Time')
        ax.legend()
        fig.autofmt_xdate()
        plt.show()
    
    else:
        for col_name in df:
            if col_name == time_col or col_name not in cols:
                continue
            fig, ax = plt.subplots(figsize=(10, 10), sharex=True)
            ax.scatter(time_col, col_name, data=df.iloc[zoom[0]:zoom[1]], **kwargs)
            ax.set_title(col_name)
            ax.grid(True)
            ax.set(xlabel='Date / Time', ylabel=col_name)
            fig.autofmt_xdate()
            plt.show()
            

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
        cols = list(df.columns)
    
    if together:
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_title(cols)
        for col_name in df:
            if col_name == time_col or col_name not in cols:
                continue
            ax.hist(df[col_name].values, label=col_name, **kwargs)
        ax.grid(True)
        ax.set(xlabel=col_name)
        ax.legend()
        plt.show()
    
    else:
        for col_name in df:
            if col_name == time_col or col_name not in cols:
                continue
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.hist(df[col_name].values, **kwargs)
            ax.set_title(col_name)
            ax.grid(True)
            ax.xaxis.set_major_locator(plt.MaxNLocator(10))
            ax.set(xlabel=col_name, ylabel='Frequency')
            plt.show()

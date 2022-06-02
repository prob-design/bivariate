# -*- coding: utf-8 -*-
"""
Created on Wed May 25 14:30:34 2022

@author: jelle
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def set_TUDstyle():
    TUcolor = {"cyan": "#00A6D6", "darkgreen": "#009B77", "purple": "#6F1D77",
               "darkred": "#A50034", "darkblue": "#0C2340",
               "orange": "#EC6842", "green": "#6CC24A",
               "lightcyan": "#00B8C8", "red": "#E03C31", "pink": "#EF60A3",
               "yellow": "#FFB81C", "blue": "#0076C2"}
    plt.rcParams.update({'axes.prop_cycle': plt.cycler(color=TUcolor.values()),
                         'font.size': 16, "lines.linewidth": 4})
    return TUcolor


def load_dataset(filename, var_time, vars_used=None):
    """Function to load the dataset and specify variables of interest
    Arguments:
        filename (str): the name of the file containing the dataset
        var_time (str): the column pandas should interpret as datetime
        var_used (list of str): the column(s) containing the data of interest.
        If not specified, al columns are used.
    Returns:
        A dataframe with the specified columns
    """
    data = pd.read_csv(filename, parse_dates=[var_time])
    if vars_used:
        cols_used = [var_time] + vars_used
        data = data[cols_used]
    return data


def clean_dataset(df, thres=3):
    """Function to clean a dataset by dropping NaN values and outliers
    Arguments:
        df (dataframe): a dataframe, preferably the output from load_dataset()
        thres (float / int): z-score threshold for removing outliers, default 3
    Returns:
        A dataframe with NaN values and outliers removed
    """
    df = df.dropna().reset_index(drop=True)

    for col_name in df:
        if pd.api.types.is_datetime64_any_dtype(df[col_name]):
            continue
        col = df[col_name]
        col_mean = col.mean()
        col_std = col.std()
        z = (col - col_mean) / col_std
        col_out_idx = col[np.abs(z) > thres].index.values.tolist()
        df = df.drop(index=col_out_idx).reset_index(drop=True)

    return df
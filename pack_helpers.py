# -*- coding: utf-8 -*-
"""
Created on Thu May 26 13:39:11 2022

@author: jelle
"""

import pandas as pd
import scipy.stats as st


def find_datetime_col(df):
    """Helper function to find the datetime column name of a given dataframe"""
    for col_name in df:
        if pd.api.types.is_datetime64_any_dtype(df[col_name]):
            return col_name


def scipy_dist(distribution):
    "Helper function to turn the name of a distribution into the scipy class"
    if distribution.lower()[:4] == "norm":
        dist = st.norm
    if distribution.lower()[:3] == "exp":
        dist = st.expon
    if distribution.lower()[:4] == "logn":
        dist = st.lognorm
    if distribution.lower()[:4] == "logi":
        dist = st.logistic
    if distribution.lower()[:4] == "extr":
        dist = st.genextreme
    else:
        Exception("Distribtution not found!")
    return dist

# -*- coding: utf-8 -*-
"""
Created on Tue May 17 13:38:10 2022

@author: ievdv
"""
# -*- coding: utf-8 -*-
"""
Created on Wed May  4 10:33:28 2022

@author: ievdv
"""

#%% Import necessary packages
import datetime
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import os
import pandas as pd
import scipy 
import scipy.stats as st
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf

from matplotlib.dates import DateFormatter
from scipy.stats import pearsonr
from scipy.stats import genextreme
from scipy.stats import kstest
from scipy.stats._continuous_distns import _distn_names

plt.rcParams.update({'font.size':16})

import func_data_exploration
import func_univariate
import func_goodness_of_fit
import func_ev
import func_bivariate

#%% Clean the dataset, dataset specific
"""
You can think of:
    - Check the frequency
    - Extract the variables of interest, two
    - Checking for and removing NaN values
    - Checking for and removing outliers
    - Describe the data
    - Make one time column
"""

# User defined things
filename = 'Climate_Data_Washington.csv'
var1 = 'POWER'
var2 = 'CLOUD_BROKEN'
var_time = 'DATE_TIME'
data = pd.read_csv(filename)

# Zoom domain
x1 = 500
x2 = 10000

label_var1 = '[m]'
label_var2 = '[m]'

xlim_var1 = 1.5
xlim_var2 = 15

# Labels for the two variables including their unit
var_lab = ['Power Usage',
           'Percent Clouds Broken']
# General x-axis label
var_lab2 = ['wave height [m]']

k = 3
qntl = .90
bins = 50

no_nan_values = data.isna().sum()

#%%
# Remove rows with nan values
data = data.dropna().reset_index(drop=True)

# Check for nan again
no_nan_values = data.isna().sum()
data_final = data[[var1, var2]]

n = data_final.shape[0]

data_final_min = data_final.min()
data_final_max = data_final.max()
data_final_mean = data_final.mean()
data_final_std = data_final.std()
data_final_P25 = data_final.quantile(.25)
data_final_P50 = data_final.quantile(.15)
data_final_P75 = data_final.quantile(.75)
data_final_P95 = data_final.quantile(.95)

data_final_summary = pd.DataFrame([data_final_min, data_final_max, 
                                   data_final_mean, data_final_std, 
                                   data_final_P25, data_final_P50, 
                                   data_final_P75, data_final_P95]).transpose().round(2)
data_final_summary.columns = ['min', 'max', 'mean', 'std', 'P25', 'P50', 'P75', 'P95']
data_final_summary.head()

# Outliers ax1
var1_out_index = data_final[var1][abs(
    (data_final[var1]-data_final_mean[var1])/data_final_std[var1]) > 3].index.values.tolist()

# Outliers ax2
var2_out_index = data_final[var2][abs(
    (data_final[var2]-data_final_mean[var2])/data_final_std[var2]) > 3].index.values.tolist()

# Unique outliers ax1 and ax2
data_final_out_index = np.unique(var1_out_index+var2_out_index).tolist()

# Remove outliers from data
data_final = data_final.drop(index=data_final_out_index).reset_index(drop=True)

n = data_final.shape[0]

data_final_min = data_final.min()
data_final_max = data_final.max()
data_final_mean = data_final.mean()
data_final_std = data_final.std()
data_final_P25 = data_final.quantile(.25)
data_final_P50 = data_final.quantile(.15)
data_final_P75 = data_final.quantile(.75)
data_final_P95 = data_final.quantile(.95)

data_final_summary_clean = pd.DataFrame(
    [data_final_min, data_final_max, data_final_mean, data_final_std, 
     data_final_P25, data_final_P50, data_final_P75, data_final_P95]).transpose().round(2)
data_final_summary_clean.columns = ['min', 'max', 'mean', 'std', 'P25', 'P50', 
                                    'P75', 'P95']

dat = data_final
dates = pd.to_datetime(data[var_time])
dat['time'] = dates

zoom = pd.DataFrame()
zoom = dat.iloc[x1:x2, :]

var = [var1,var2]
data = dat

#%%
func_data_exploration.data_exploration(var,data,zoom,label_var1, label_var2,
                                       xlim_var1,xlim_var2,bins)

#%% Univariate analysis
func_univariate.emprical(var, data,var_lab2[0],plot=True,together=False)
func_univariate.theoretical_distribution(var, data, var_lab,var_lab2[0])

#%% Goodness of Fit test
func_goodness_of_fit.func_aic_bic(k,data,var)

#%% Computations of probabilities
prob, data_q = func_goodness_of_fit.func_prob(data,var,qntl)
print(prob)

#%% Extreme value analysis
# data['GVW'] = data[['AX1', 'AX2']].sum(axis=1)
# var_new = ['GVW']
# unit = ['[kN]']

values_max, shape, loc, scale, test_GEV, AIC, BIC = func_ev.func_gev(k,data,
                                                                     var,
                                                                     label_var1,
                                                                     bins=20)

max_interest = values_max[var[0]]

print(f'Largest absolute difference between CDF and ECDF: {test_GEV[0]:.3f}')
print(f'p-value Kolmogorov-Smirnov test: {test_GEV[1]:.10f}')

print(f'The AIC value is: {AIC:.3f}')
print(f'The BIC value is: {BIC:.3f}')

print(f'Number of extremes: {len(values_max)} \n'
      f'Mean: {max_interest.mean():.3f} m \n'
      f'Median: {max_interest.median():.3f} m \n'
      f'Min: {max_interest.min():.3f} m \n'
      f'Max: {max_interest.max():.3f} m \n'
      f'Standard deviation: {max_interest.std():.3f} m')
print('')
print(f'shape: {shape:.3f} \n'
      f'scale: {scale:.3f} \n'
      f'location: {loc:.3f}')

#%% bivariate
func_bivariate.plot(data,var,var_lab)

covariance, corr = func_bivariate.cov_cor(data,var)
print(f'The Pearson\'s correlation coefficient for the pair {var[0]}-{var[1]} is: {corr}')
print(f'The covariance for the pair {var[0]}-{var[1]} is: \n {covariance}')

data1, df_r_norm = func_bivariate.fit(data,var,var_lab)
func_bivariate.probabilities(data1, var, data, data_q, qntl, var_lab, df_r_norm)
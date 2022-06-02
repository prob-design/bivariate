# -*- coding: utf-8 -*-
"""
Created on Mon May  2 16:30:49 2022

@author: ievdv
"""
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from scipy.stats import genextreme
import numpy as np
from scipy.stats import kstest

def func_gev(k,dat,var1,unit,bins=50,format_date='%Y %m %d %H:%M:%S'):
    gev = dat.copy()
    gev['datetime'] = pd.to_datetime(gev['time'], format=format_date)
    # gev = gev.set_index(pd.DatetimeIndex(gev['datetime']))
    gev['Date'] = gev['datetime'].dt.date

    loc_values_max = list(gev.groupby(['Date']))

    values_max = pd.DataFrame([l[1].loc[l[1][var1[0]].idxmax()]
                               for l in loc_values_max]).reset_index(drop=True)

    plt.figure(figsize=(14, 8))
    plt.plot(gev['Date'],gev[var1[0]],'.')
    plt.plot(values_max['Date'], values_max[var1[0]], 'x', markersize=15)
    plt.xticks(rotation=45)
    plt.grid(linestyle='--', alpha=.3)
    plt.xlabel('Date')
    plt.ylabel(var1[0])
    plt.title(f'{var1[0]}, daily maxima')
    # plt.xlim([datetime.date(2014, 5, 1), datetime.date(2014, 10, 1)])
    plt.show()

    shape, loc, scale = genextreme.fit(values_max[var1[0]])
    
    a = values_max[var1[0]].min()
    b = 1.001*values_max[var1[0]].max()
    N = 200

    input_val = np.linspace(a, b, N)
    
    pdf_gev = genextreme.pdf(input_val, shape, loc=loc, scale=scale)
    cdf_gev = genextreme.cdf(input_val, shape, loc=loc, scale=scale)

    plt.figure(figsize=(14, 8))
    plt.hist(values_max[var1[0]], density=True, bins=bins,
             label='Max Obs.')
    plt.plot(input_val, pdf_gev, linewidth=3, label='GEV')
    plt.xlabel(f'{var1[0]} {unit} ')
    plt.ylabel('Density')
    plt.title('GEV fit for the maxima', fontsize=18)
    plt.grid(linestyle='--', alpha=.3)
    plt.legend()
    plt.show()
    
    x = np.sort(values_max[var1[0]])  # Sort the values from small to large
    n = x.size  # Determine the number of datapoints
    # Make a list of n values corresponding to the probability
    y = np.arange(1, n+1) / n

    plt.figure(figsize=(14, 8))
    plt.plot(input_val, cdf_gev, linewidth=2, label='CDF')
    plt.plot(x, y, '.', label='Obs.')
    plt.xlabel(f'{var1[0]} {unit}')
    plt.ylabel('Cumulative probability')
    plt.title('Cumulative distribution function')
    plt.grid(linestyle='--', alpha=.3)
    plt.legend()
    plt.show()
    
    test_GEV = kstest(x, 'genextreme', args=(shape, loc, scale))
    
    n = len(values_max[var1[0]])  # Determine number of daily maxima

    logL = np.sum(genextreme.logpdf(values_max[var1[0]],
                                    shape, loc=loc, scale=scale))  

    AIC = 2*k - 2*logL
    BIC = k*np.log(n) - 2*logL

    Nobs = len(values_max[var1[0]]) + 1
    daily_max_sorted = np.sort(values_max[var1[0]])  # Sort the original data

    ecdf_max = np.linspace(1, len(values_max[var1[0]]), len(values_max[var1[0]]))/(Nobs+1)
    f_max = genextreme.cdf(daily_max_sorted, shape, loc=loc, scale=scale)

    plt.figure(figsize=(14, 8))
    plt.plot(ecdf_max, f_max, '.', color='r')
    plt.plot([0, 1], [0, 1], '--', color='k')
    plt.xlabel('Empirical quantiles')
    plt.ylabel('Theoretical quantiles')
    plt.title('QQ-plot')
    plt.grid(linestyle='--', alpha=.3)
    plt.show()
    
    inv_50 = genextreme.ppf(0.5, shape, loc=loc, scale=scale)
    inv_90 = genextreme.ppf(0.9, shape, loc=loc, scale=scale)
    inv_95 = genextreme.ppf(0.95, shape, loc=loc, scale=scale)

    T = 100
    T_ret = 1 - 1/100
    T_100 = genextreme.ppf(T_ret, shape, loc=loc, scale=scale)

    a = values_max[var1].min()
    b = 1.8*values_max[var1].max()
    N = 200

    x = np.linspace(a, b, N)
    cdf_gev = genextreme.cdf(x, shape, loc=loc, scale=scale)

    # Make a plot of the CDF inverse and the various quantiles
    plt.figure(figsize=(14, 8))
    plt.plot(x, 1-cdf_gev, color='k')
    plt.axvline(inv_50, linestyle='--', color='r',
                label=f'50th quantile = {inv_50:.2f}')
    plt.axvline(inv_90, linestyle='--', color='m',
                label=f'90th quantile = {inv_90:.2f}')
    plt.axvline(inv_95, linestyle='--', color='g',
                label=f'95th quantile = {inv_95:.2f}')
    plt.axvline(T_100, linestyle='--', color='b',
                label=f'100-yr event = {T_100:.2f}')
    plt.xlabel(f'{var1[0]} {unit}')
    plt.ylabel('Exceedance probability')
    plt.legend(bbox_to_anchor=(1, 1.02))
    plt.grid(linestyle='--', alpha=.3)
    plt.xlim(x.min(),T_100*1.2)
    plt.show()
    
    return values_max, shape, loc, scale, test_GEV, AIC, BIC


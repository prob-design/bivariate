# -*- coding: utf-8 -*-
"""
Created on Wed May 25 15:29:42 2022

@author: jelle
"""
import pack_init as init
import pack_exploration as expl
import pack_univariate as univ
import pack_gof as gof
import pack_extreme as ext
import pack_bivariate as bivar


file = "Climate_Data_Washington.csv"
col_names = ["POWER", "CLOUD_BROKEN"]
col_time = "DATE_TIME"
labels = ["Power Usage", "Percentage broken clouds"]
test_dist = "Normal"
ev_frequency = "Weekly"

TU_color = init.set_TUDstyle()

data = init.load_dataset(file, col_time, col_names)
data = init.clean_dataset(data)

expl.data_summary(data)

expl.time_plot(data)
expl.time_plot(data, cols=col_names[0], zoom=[1000, 2000])

expl.hist_plot(data)

univ.plot_ecdf(data[col_names[0]], labels[0])

univ.plot_distributions(data[col_names[0]], label=labels[0])

univ.plot_distributions(data[col_names[0]], seperate=False, label=labels[0])

print(gof.aic_bic_fit(data[col_names[0]], test_dist))

gof.QQ_plot(data[col_names[0]], test_dist)

print(gof.quantile_compare(data[col_names[0]], test_dist, 0.90))

extreme = ext.create_ev(data, ev_frequency)

expl.time_plot(extreme, cols=[col_names[0]])

ext.fit_ev(extreme[col_names[0]])

print(ext.AIC_BIC_ev(extreme[col_names[0]]))

ext.QQ_plot_ev(extreme[col_names[0]])

bivar.bivar_plot(data, col_names, labels)

bivar.bivar_fit(data, col_names, labels)

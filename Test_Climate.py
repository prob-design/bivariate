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

test_file = "https://surfdrive.surf.nl/files/index.php/s/m7KhpkIRkBJm6QB"
test_folder = "https://surfdrive.surf.nl/files/index.php/s/Wg6SWc38zn8jqVg"
test_path = "subfolder01/subfolder02/Climate_Data_Washington_Metric.csv"

file = "https://surfdrive.surf.nl/files/index.php/s/Wg6SWc38zn8jqVg/download?path=%2Fsubfolder01%2Fsubfolder02&files=Climate_Data_Washington_Metric.csv"
col_names = ["POWER", "CLOUD_BROKEN"]
col_time = "DATE_TIME"
labels = ["Power Usage", "Percentage broken clouds"]
test_dist = "Normal"
ev_frequency = "Weekly"

TU_color = init.set_TUDstyle()

data = init.load_SURFdrive_path(test_folder, test_path, col_time, col_names)
data = init.clean_dataset(data)

expl.data_summary(data)

expl.time_plot(data, alpha=0.5, marker="o")
expl.time_plot(data, zoom=[500, 1000], together=True)

expl.hist_plot(data)
expl.hist_plot(data, together=True)

univ.plot_ecdf(data[col_names[0]], labels[0], color="red")

univ.plot_distributions(data[col_names[0]], label=labels[0], linestyle="--")

univ.plot_distributions(data[col_names[0]], seperate=False, label=labels[0])

print(gof.aic_bic_fit(data[col_names[0]], test_dist))

gof.QQ_plot(data[col_names[0]], test_dist, color="red")

print(gof.quantile_compare(data[col_names[0]], test_dist, (0.90, 0.95, 0.99)))

extreme = ext.create_ev(data, ev_frequency)

expl.time_plot(extreme, cols=[col_names[0]])

ext.fit_ev(extreme[col_names[0]])

print(ext.AIC_BIC_ev(extreme[col_names[0]]))

ext.QQ_plot_ev(extreme[col_names[0]])

bivar.bivar_plot(data, col_names, labels=labels)

print(bivar.cov_cor(data, col_names))

fit = bivar.bivar_fit(data, col_names, labels=labels)

print(fit)

print(bivar.and_or_probabilities(data, col_names, [0.9, 0.9], labels=labels))

print(bivar.and_or_probabilities(fit, col_names, [0.9, 0.9], labels=labels))

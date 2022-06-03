# Bivariate Package for MUDE

This directory is used to develop the Bivariate Package for the MUDE module.

The module currently contains 6 sub-packages:
- pack_init: used to load and clean datasets, and set the TU Delft plotting style.
- pack_exploration: used to create some simple time and distribution plots of a single variable
- pack_univariate: used to fit and plot distributions to single variables
- pack_gof: used to evaluate goodness of fit of ditributions of single variables
- pack_extreme: used to create extreme value analyses
- pack_helpers: some general function used by the other sub-packages

A sub-package specifically for bivariate analysis is currently being worked on in a separate branch, and further functionality could be added in the future.

The files beginning with "func_" belong to an old implementation and can be removed when alle of their functionaluty is present in the "pack_"-files.

There are also two test files in this directory, more could be added with different datasets.

The current list of methods in every subpackage is given below:
- pack_init
	* set_TUDstyle
	* load_dataset
	* clean_dataset
- pack_exploration
	* data_summary
	* time_plot
	* hist_plot
- pack_univariate:
	* ecdf
	* plot_ecdf
	* fit_distribution
	* plot_distributions
- pack_gof:
	* aic_bic
	* aic_bic_fit
	* QQ_plot
	* quantile_compare
- pack_extreme:
	* create_ev
	* fit_ev
	* AIC_BIV_ev
	* QQ_plot_ev
- pack_helpers
	* find_datetime
	* scipy_dist
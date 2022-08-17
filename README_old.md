# Bivariate Package MUDE
*version 1.0*

*Created by Thirza Feenstra, Jelle Knibbe and Irene van der Veer*

Welcome to the manual of the Python package used for the MUDE module. This document will show you how to use the different functions in this package.

## Overview

The module currently contains 6 sub-packages:
- pack_init: used to load and clean datasets, and set the TU Delft plotting style.
- pack_exploration: used to create some simple time and distribution plots of a single variable
- pack_univariate: used to fit and plot distributions to single variables
- pack_gof: used to evaluate goodness of fit of ditributions of single variables
- pack_extreme: used to create extreme value analyses
- pack_helpers: some general function used by the other sub-packages

Currently, the user manual is being written by Jelle. Everyone is invited to proofread it and see if they can figure out the package by themselves.

The current list of methods in every subpackage is given below:
- pack_init
	* set_TUDstyle
	* load_dataset
	* clean_dataset
	* load_SURFdrive_path
	* load_SURFdrive_file
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

## Installing the package
To pull the bivariate package from the gitlab repository, copy-paste the following code:

`!git clone https://gitlab.tudelft.nl/mude/bivariate_package.git`

Then, import the different subpackages as follows:
`from bivariate_package import pack_init`

`from bivariate_package import pack_exploration`

`from bivariate_package import pack_univariate`

`from bivariate_package import pack_gof`

`from bivariate_package import pack_extreme`

`from bivariate_package import pack_bivariate`


## Starting point: the `init`-subpackage
**Loading and cleaning a dataset**

First, in order to load a dataset, use the function `load_dataset` from the `init` subpackage. Use the name or path of the file as arguments, and the name of the column(s) that contains the time variable. This will be passed to the `parse_dates` argument of pandas. Optionally, add a list of the variables `vars_used` can be given as a keyword argument. In this case, the function only loads the columns you actually want to use. If this is missing, all of them will be used, which could cause problems while cleaning the data.

Speaking of which, use the `clean_dataset` function of the same package to remove any rows containing `Nan` values or outliers, which are determined using the absolute value of the Z-score. The threshold of the Z-score is 3 by default, but this can be changed manually using the keyword argument `thres`.  The first argument of this function should generally be the dataframe generated from `load_dataset`.

Finally, in order to use the TU Delft colours and other default plotting styles, simply call the `set_TUDstyle`-function without any arguments.

If you've been given SURFdrive links instead of files, you can use `load_SURFdrive_file` and `load_SURFdrive_path` instead. Use the first one if your link leads directly to a file, and the second one if it leads to a directory. In the latter case, you also need to supply a path to the file you want. For example, to access a file `filename.csv` in the top-folder, simply supply that name as a second argument. If it's located in a folder, use `folder/filename.csv`. Both of these functions also use the column arguments that `load_dataset` uses.

## First steps: the `exploration`-subpackage

**Checking your data and creating simple plots**

This subpackage (currently) contains three functions. All three of them can be used on a complete DataFrame `df`, though you can optionnaly specify which columns you want to use with a list of strings `cols`. The first, `data_summary`, simply gives a description of the values in the DataFrame. It is a pass-through of pandas' `describe`-function, meaning you can pass any keyword argument pandas uses to this one as well. This is something you will see often in this manual.

The `time_plot`-function creates a markerplot of the columns in your dataframe against time. By default, it creates seperate figures for every column, but you change this by setting `together=True`. You can alos focus focus on a certain segment between two rows of your dataset by supplying a list of 2 integers `zoom`. Finally, this function passes through `ax.scatter`, so any more keyword that work with that function can be used here. Examples include the markerstyle and transparency.

Finally, `hist_plot` creates a histogram. The arguments are the same as `time_plot`, but without `zoom`. The pass-through is `ax.hist()`, so you could supply bins or whether to use the frequency or density.

## Fitting distributions: the `univariate`-subpackage

**Single Variables**

All of the functions in this subpackage take a single column of d dataframe (also known as a Series) as main argument `var`.  The first, `ecdf` simply creates an empirical cumulative distribution of the given variable. It returns the sorted values of the variable and the distribution (both as numoy-arrays) for easy plotting.

The `plot_ecdf`-function plots the empirical cdf of the given variable, the empirical probablilty of exceedance (1 minus the previous value), and both of these with a vertical log scale for four plots total.

The next functions extensively use a function from the 'hidden' `helpers` called `scipy-dist`. While you'll likely not interact directly with this function yourself, understanding it will make the next functins easier to use, so it is still described here.

This function takes a single string as input and interprets it as one of several distributions, which is then returned as `scipy.stats`-class, which then contains a number of functions used in this sub-package. Only the first few characters of the case-independent string are interpreted, though note the exact string you supply is used as a title of label in some of the plots. Currently supported distributions are:

| Distribution | String interpreted  |
|-------|---|
|  Normal     | norm  |
|  Exponential     | exp  |
|  Lognormal     | logn  |
|  Logistic| logi|
|Extreme Values| extr|

More will be added in the future. Now, let's move on to the other functions in `pack_bivariate`.

The `fit_distribution`, of course, fits a distribution to the given `var`. Thanks to the helper function, you only have to supply the name of the distribution as a string. Optionally, you can plot the distribution, including an optional `label` and any pass-throughs for `plt.plot`.

Finally, the `plot-distributions`-function performs the previous function four times on a single variable, using a normal, exponential, lognormal and logistic distribution. It can also create a single figure with all four distributions by setting `seperate=False`, and passes through `plt.plot` in both cases.


## Evaluating fits: the `gof`-subpackage
**Scores and quantiles**

After fitting a distribution to a variable, you'll want to determine the quality of the fit. This subpackage provides several ways to do this.

First, the Akaike Information Criterion (AIC) and Bayesian Information Criterion (BIC) of a fitted distribution can be calculated using the `aic_bic`-function. This takes the fitted `pdf` (not the cdf!) as input, along with the number of parameters `k` and number of data points `n` as input, and returns both model scores.

The `aic_bic_fit`-function can be used in much the same way as some of the functions in the `univariate`-subpackage. It takes a Series and a string specifying the distribution as input, and uses the same `helpers`-function to fit a distribution and then return the model scores.

The `QQ-plot`-function also uses the same arguments as the univariate functions. It creates a plot of the fitted quantiles against the empirical ones (which is the pass-through). Of course, if the fit is good, this should be e nearly straight line from the origin to the point (1,1), which is als plotted to compare.

Finally, the `quantile-compare`-function. Once again it takes a Series and a distribution as argument, and now a single quantile or list of qunatiles too. It returns both the fitted and empirical value of the variable belonging to this quantile, as well as the quantile of the fitted distribution belonging to the empirical value of the original quantile. If this sounds confusing, consider it as a single point on the QQ-plot.

## Extreme values: the `extreme`-subpackage
**Periodical Peaks**

This package is used to do extreme value analyses on existing datasets. Currently, it only supports `scipy`'s General Extreme Value distribution. In fact, three of the four functions are just previously covered functions that take a distribution as argument, with the value of this string set to `"Extreme`. The only argument needed is then, of course, a Series of extreme values. You can generate this (an entire DataFrame, in fact!) with the `create_ev`-function.

This function takes a DataFrame and a string `period` as arguments, and returns a new DataFrame with the maximum values of the original one, evaluated on the user-defined period. This works through pandas' `resample`-method, meaning any string that defines a period for pandas works here too. Some examples are:

|String|Period|
|---|---|
|`"D"`|Daily|
|`"W"`|Weekly|
|`"M"`|Monthly|
|`"A"`/`"Y"`|Yearly|

As the way this package deals with distributions is not really different to the way the `gof`- and `univariate`-subpackages deal with them, the functions of those two are compatible with the resulting extreme-values dataframe.

## Two variables: the `bivariate`-subpackage
**The actual name of the package**

Finally, the bivariate analysis functions. The first one, `bivar_plot`, corresponds to the plotting functions in the `exploration`-subpackage. It takes a dataframe and a list of two strings of columns to use, and generates three different plots:
- A simple marker plot of all the datapoints
- The same, but with histograms of the two variables on the axes.
- A 2D density plot with iso-lines, which can take some time to run.

Currently, none of the three plots use pass-through arguments, though you can supply axis labels.

The second function, `cov_cor`, uses the same arguments as `bivar_plot`. It generates a covariance matrix and the Pearson's correlation coefficient between the two supplied variables.

The `bivar_fit`-function corresponds to the fitting of distributions in the `univariate`-subpackage. Currently, it only supports the Bivariate Normal distribution, and does not return the parameters but rather a 'draw' of the distribution, in the form of a DataFrame. The number of data points in this draw, `N`, is equal to the number of rows in the supplied dataframe by default, but you can give it a different value if you want. This function also plots the generated draw using the `bivar_plot` function, unless you supply `plot=False`.

Finally, the `and_or_probabilities` again uses a dataframe with two specified columns, plus a quantile per column, to compute the probability of both (AND) or one of the two (OR) variables being larger than its given quantile. These two probabilities are returned by the function, and by default a plot is also generated visualising both the AND and OR-scenarios. Note that this function can also be used with the output of `bivar_fit`.
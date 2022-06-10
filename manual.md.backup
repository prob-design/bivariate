# Bivariate Package MUDE
*version ???*

*Created by Thirza Feenstra, Jelle Knibbe and Irene van der Veer*

Welcome to the manual of the Pyhton package used for the MUDE module. This document will show you how to use the different functions in this package.

## Starting point: the `init`-subpackage
**Loading and cleaning a dataset**

First, in order to load a dataset, use the function `load_dataset` from the `init` subpackage. Use the name or path of the file as arguments, and the name of the column(s) that contains the time variable. This will be passed to the `parse_dates` argument of pandas. Optionally, add a list of the variables `vars_used` can be given as a keyword argument. In this case, the function only loads the columns you actually want to use. If this is missing, all of them will be used, which could cause problems while cleaning the data.

Speaking of which, use the `clean_dataset` function of the same package to remove any rows containing `Nan` values or outliers, which are determined using the absolute value of the Z-score. The threshold of the Z-score is 3 by default, but this can be changed manually using the keyword argument `thres`.  The first argument of this function should generally be the dataframe generated from `load_dataset`.

Finally, in order to use the TU Delft colours and other default plotting styles, simply call the `set_TUDstyle`-function without any arguments.

## First steps: the `exploration`-subpackage

**Checking your data and creating simple plots**

This subpackage (currently) contains three functions. All three of them can be used on a complete DataFrame `df`, though you can optionnaly specify which columns you want to use with a list of strings `cols`. The first, `data_summary`, simply gives a description of the values in the DataFrame. It is a pass-through of pandas' `describe`-function, meaning you can pass any keyword argument pandas uses to this one as well. This is something you will see often in this manual.

The `time_plot`-function creates a markerplot of the columns in your dataframe against time. By default, it creates seperate figures for every column, but you change this by setting `together=True`. You can alos focus focus on a certain segment between two rows of your dataset by supplying a list of 2 integers `zoom`. Finally, this function passes through `ax.scatter`, so any more keyword that work with that function can be used here. Examples include the markerstyle and transparency.

Finally, `hist_plot` creates a histogram. The arguments are the same as `time_plot`, but without `zoom`. The pass-through is `ax.hist()`, so you could supply bins or whether to use the frequency or density.

## Fitting distributions: the `univariate`-subpackage

## Evaluating fits: the `gof`-subpackage

## Extreme values: the `extreme`-subpackage

## Two variables: the `bivariate`-subpackage

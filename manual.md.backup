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

## Fitting distributions: the `univariate`-subpackage

## Evaluating fits: the `gof`-subpackage

## Extreme values: the `extreme`-subpackage

## Two variables: the `bivariate`-subpackage

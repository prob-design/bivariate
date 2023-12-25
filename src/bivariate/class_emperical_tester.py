import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as st
import seaborn as sns

from datetime import datetime
from IPython.display import display
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from typing import Any, Dict, List, Optional, Sequence, Tuple, Type, TypeVar, Union
from math import ceil, trunc

# This class is made by Siemen Algra and based of the class of Benjamin Rousse 

class Emperical_data():
    
    
    def __init__(self,
                 time_series_data: np.array,
                 data_title = None,
                 data_units = None
                 ):
        """Attributes of the emperical_data class.
        
        Parameters
        ----------
        time_series_data : `numpy.array`
            Numpy array containing time series data. This data is used to perform
            emperical statistical analysis on. 
        data_label : `str`   
            This is the label of the data. This is used for plotting purposes.
            
            
        Examples
        --------
        If one has a timeseries in the csv file 'data.csv' and wants to perform 
        statistical operations on it, one can use the following code:

        >>> data_array = np.genfromtxt('data.csv', delimiter=',')
        >>> data_label = "Measured data"
        >>> Emperical_data(data_array, data_label)
        
        """
        
        # Assign input to attributes
        self.data_array = time_series_data
        self.data_title = data_title
        self.data_units = data_units
        
        

        # To be computed
        self.extremes = None
        self._bivariate_vars = None
        self._bivar_r_norm = None
        self._cov = None
        self._cor = None
        self.statistical_summary = None


    def data_summary(self) -> None:
        """Generates statistical summary of inputted data.
        
        This function generates a statistical summary of the inputted data.
        This is done by using scipy.stats.describe. 
        
        
        Parameters
        ----------
        self : `data_array`
            Numpy array containing time series data. 
            Note that this is already an attribute of the class.
        
        Returns
        -------
        nobs : `int`
            Number of observations in the data.
        minmax : `tuple`
            Tuple containing the minimum and maximum value of the data.
        mean : `float`
            Mean of the data.
        variance : `float`
            Variance of the data.
        skewness : `float`
            Skewness of the data.
        kurtosis : `float`
            Kurtosis of the data.
        
        
        
        """

        # Generate statistical summary
        nobs, minmax, mean, variance, skewness, kurtosis = st.describe(self.data_array)

        # Print summary
        print(f"For the data: {self.data_title}")
        print(f"    Number of observations: {nobs}")
        print(f"    Minimum and maximum: {minmax}")
        print(f"    Mean: {mean}")
        print(f"    Variance: {variance}")
        print(f"    Skewness: {skewness}")
        print(f"    Kurtosis: {kurtosis}")
        
        self.statistical_summary = nobs, minmax, mean, variance, skewness, kurtosis
        
    
    def time_series_plot(self,
                         ax=None,
                         fig=None):
        """Plot the time series data

        Parameters
        ----------
        ax : `matplotlib.axes.Axes`
            Axes object to plot on. If not provided, a new figure will be created.

        Returns
        -------
        fig : `matplotlib.figure.Figure`
            Figure object containing the plot.
        ax : `matplotlib.axes.Axes`
            Axes object containing the plot.        
        
        """
        
        # Create figure and axes if not provided
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))


        ax.plot(self.data_array, label = f'{self.data_title}')
        ax.set_title(f"Time series plot of {self.data_title}")
        ax.set_xlabel(r"$Time$")
        ax.set_ylabel(f"{self.data_units}")
        ax.legend()
        
        return fig, ax
    
    def PDF_and_ECDF_plot(self,
                           axes=None,
                           fig=None):
        """Plot the histogram of the data

        Parameters
        ----------
        axes : `matplotlib.axes.Axes`
            Axes object to plot on. If not provided, a new figure will be created.
        fig : `matplotlib.figure.Figure`
            Figure object to plot on. If not provided, a new figure will be created.

        Returns
        -------
        fig : `matplotlib.figure.Figure`
            Figure object containing the plot.
        ax : `matplotlib.axes.Axes`
            Axes object containing the plot.        
        
        """
        
        # Create figure and axes if not provided
        if axes is None:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Plot the histogram of the probability density function
        axes[0].hist(self.data_array, edgecolor='k', linewidth=0.2, 
                     label=f'{self.data_title}', density = True)
        axes[0].set_xlabel(f'{self.data_units}')
        axes[0].set_ylabel('density')
        axes[0].set_title('PDF', fontsize=18)
        axes[0].legend()
        
        # Create function to compute the empirical cumulative distribution function
        def ecdf(var):
            x = np.sort(var) # sort the values from small to large
            n = x.size # determine the number of datapoints
            y = np.arange(1, n+1) / (n+1)
            return [y, x]
        
        # Plot the empirical cumulative distribution function
        axes[1].step(ecdf(self.data_array)[1], ecdf(self.data_array)[0], 
                     label=f'{self.data_title}')
        axes[1].set_xlabel(f'{self.data_units}')
        axes[1].set_ylabel('${P[X \leq x]}$')
        axes[1].set_title('CDF', fontsize=18)
        axes[1].legend()
        axes[1].grid()


        return fig, axes
    
    def fitting_distributions(self):
        """Fit distributions to the data

        This function uses scipy.stats to fit distributions to the data.
        The following distributions are fitted:
        - Normal distribution
        - Lognormal distribution
        - Exponential distribution
        - Weibull distribution
        
        Parameters
        ----------
        self : `data_array`
            Numpy array containing time series data. 
            Note that this is already an attribute of the class.
        
        Returns
        -------
        norm : `scipy.stats.norm`
            Normal distribution fitted to the data.
        lognorm : `scipy.stats.lognorm`
            Lognormal distribution fitted to the data.
        expon : `scipy.stats.expon`
            Exponential distribution fitted to the data.
        weibull : `scipy.stats.weibull_min`
            Weibull distribution fitted to the data.
        
        """
        
        self.params_logn = st.lognorm.fit(self.data_array, floc = 0)
        self.params_gumb = st.gumbel_r.fit(self.data_array)

        
        
    def graphical_assessing_goodness_of_fit(self,
                                         axes=None,
                                         fig=None):
        """Plots the Emperical CDF and the fitted CDF's
        

        Parameters
        ----------
        axes : `matplotlib.axes.Axes`
            Axes object to plot on. If not provided, a new figure will be created.
        fig : `matplotlib.figure.Figure`
            Figure object to plot on. If not provided, a new figure will be created.

        Returns
        -------
        fig : `matplotlib.figure.Figure`
            Figure object containing the plot.
        ax : `matplotlib.axes.Axes`
            Axes object containing the plot.        
        
        """

        # Create figure and axes if not provided
        if axes is None:
            fig, axes = plt.subplots(figsize=(5, 6))
        
        # Create function to compute the empirical cumulative distribution function
        def ecdf(var):
            x = np.sort(var)
            n = x.size
            y = np.arange(1, n+1) / (n+1)
            return [x, y]
        
        ### NEED TO GENERALIZE THIS CODE SUCH THAT IT USES AUGMENTED_RV_CONTINUOUS CLASS 
        ### NEED TO BE ABLE TO EXPAND THIS CODE TO MULTIPLE DISTRIBUTIONS
        
        # Plot the empirical cumulative distribution function
        axes.step(ecdf(self.data_array)[0], ecdf(self.data_array)[1], 
                  color = 'black', label=f'{self.data_title}')
        axes.plot(ecdf(self.data_array)[0], st.lognorm.cdf(ecdf(self.data_array)[0], *self.params_logn),
                    color='cornflowerblue', label='Lognormal')
        axes.plot(ecdf(self.data_array)[0], st.gumbel_r.cdf(ecdf(self.data_array)[0], *self.params_gumb),
                    '--', color = 'grey', label='Gumbel')
        
        
        axes.set_xlabel(f'{self.data_units}')
        axes.set_ylabel('${P[X \leq x]}$')
        axes.set_title(f'CDF emperical and fitted of {self.data_title}', fontsize=18)
        axes.set_yscale('log')
        axes.legend()
        axes.grid()
        
        return fig, axes
    
    def QQ_plot(self,
                axes=None,
                fig=None):
        """Plots the Emperical CDF and the fitted CDF's
        

        Parameters
        ----------
        axes : `matplotlib.axes.Axes`
            Axes object to plot on. If not provided, a new figure will be created.
        fig : `matplotlib.figure.Figure`
            Figure object to plot on. If not provided, a new figure will be created.

        Returns
        -------
        fig : `matplotlib.figure.Figure`
            Figure object containing the plot.
        ax : `matplotlib.axes.Axes`
            Axes object containing the plot.        
        
        """
        
        
        # Create figure and axes if not provided
        if axes is None:
            fig, axes = plt.subplots(figsize=(5, 6))
        
        
        def ecdf(var):
            x = np.sort(var)
            n = x.size
            y = np.arange(1, n+1) / (n+1)
            return [y, x]
        
        axes.plot([trunc(min(self.data_array)), ceil(max(self.data_array))], [trunc(min(self.data_array)), ceil(max(self.data_array))], 'k')
        axes.scatter(ecdf(self.data_array)[1], st.lognorm.ppf(ecdf(self.data_array)[0], *self.params_logn), 
                color='cornflowerblue', label='Lognormal')
        axes.scatter(ecdf(self.data_array)[1], st.gumbel_r.ppf(ecdf(self.data_array)[0], *self.params_gumb),
                color='grey', label='Gumbel')
        axes.set_xlabel(f'Observed {self.data_units}')
        axes.set_ylabel(f'Estimated {self.data_units}')
        axes.set_title(f'QQ-plot of {self.data_title}', fontsize=18)
        axes.set_xlim([trunc(min(self.data_array)), ceil(max(self.data_array))])
        axes.set_ylim([trunc(min(self.data_array)), ceil(max(self.data_array))])
        axes.legend()
        axes.grid()

        return fig, axes
    
    def KS_test(self):
        """Performs the Kolmogorov-Smirnov test
        

        Parameters
        ----------
        self : `data_array`
            Numpy array containing time series data. 
            Note that this is already an attribute of the class.

        Returns
        -------
        KS_test_logn : `scipy.stats.kstest`
            Kolmogorov-Smirnov test for the lognormal distribution.
        KS_test_gumb : `scipy.stats.kstest`
            Kolmogorov-Smirnov test for the Gumbel distribution.
        
        """
        
        _, self.KS_test_logn = st.kstest(self.data_array, 'lognorm', args=self.params_logn)
        _, self.KS_test_gumb = st.kstest(self.data_array, 'gumbel_r', args=self.params_gumb)
        
        print(f'The Kolmogorov-Smirnov test for the lognormal distribution of {self.data_title} gives a p-value of {np.round(self.KS_test_logn,3)}')
        print(f'The Kolmogorov-Smirnov test for the Gumbel distribution of {self.data_title} gives a p-value of {np.round(self.KS_test_gumb,3)}')
        
        

class augmented_rv_continuous(st.rv_continuous):
    """Class for the distribution 
    
    This class is based on
    
    
    Examples
    --------
    If one has a timeseries in the csv file 'data.csv' 
    and wants to fit a  distribution to it,
    one can use the following code:

    >>> emperical_data = np.genfromtxt('data.csv', delimiter=',')
    >>> fitted_lognormal_params = lognormal_rv.fit(emperical_data)
    >>> RV_lognormal = lognormal_rv(*fitted_lognormal_params)
    
    """
    def __init__(self,
                 time_series_data: np.array
                 type_distribution: str,
                 ):  

    # Assign input to attributes
    
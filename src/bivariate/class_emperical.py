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

# This class is made by Siemen Algra, based on study material provided by the MUDE teaching team 
# and a class made by Benjamin Rouse

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
        self.distributions = {}             # Create empty dictionary where the distribution can be stored in 


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
    
    def ecdf(self):
        """Compute the empirical cumulative distribution function
        
        """

        x = np.sort(self.data_array)
        n = x.size
        y = np.arange(1, n+1) / (n+1)
        
        return [x, y]
    
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
        
        # Calculate emperical CDF
        
        
        # Plot the empirical cumulative distribution 
        #rrrrrrrr I don't like these variables names
        x, y = self.ecdf()

        axes[1].step(x, y, label=f'{self.data_title}')
        axes[1].set_xlabel(f'{self.data_units}')
        axes[1].set_ylabel('${P[X \leq x]}$')
        axes[1].set_title('CDF', fontsize=18)
        axes[1].legend()
        axes[1].grid()


        return fig, axes
    
    def fitting_distributions(self, distribution_names: List[str]):
        """Fit distributions to the data

        This function uses scipy.stats to fit distributions to the data.
        The can be fitted:
        - Normal distribution
        - Lognormal distribution
        - Exponential distribution
        - Weibull distribution
        
        Parameters
        ----------
        self : `data_array`
            Numpy array containing time series data. 
            Note that this is already an attribute of the class.
        name_distribution : `str`
            Name of the distribution to be fitted.
        
        Returns
        -------

        
        """
        distribution_mapping = {
            'lognormal': st.lognorm,
            'gumbel': st.gumbel_r,
            'exponential': st.expon,
            'weibull': st.weibull_min,
            'normal': st.norm,
            'gamma': st.gamma
            # Add more distributions as needed
        }
        
        
        # This is necessary because of name convention in other packages
        distr_scipy_names = {
            'lognormal': 'lognorm',
            'gumbel': 'gumbel_r',
            'exponential': 'expon',
            'weibull': 'weibul_min',
            'normal': 'norm',
            'gamma': 'gamma'
            # Add more distributions as needed
        }
        
        # Loop over provided list of distributions
        for name in distribution_names:
            if name not in distribution_mapping:
                raise ValueError(f"Unsupported distribution: {name}")


            distribution_class = distribution_mapping[name]
            params = distribution_class.fit(self.data_array)
            rv = distribution_class(*params)
            self.distributions[name] = {'params': params, 'RV_': rv, 'scipy_name': distr_scipy_names[name]}

 
        
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
        # Specify x and y values for the ECDF
        x, y = self.ecdf()

        # Create figure and axes if not provided
        if axes is None:
            fig, axes = plt.subplots(figsize=(5, 6))   
        
        # Plot the empirical cumulative distribution function
        axes.step(x, y, 
                  color = 'black', label=f'{self.data_title}')
        
        # Plot all the fitted distributions, if none, prints statement    
        if not self.distributions:
            print("No fitted distributions found.")
        else:
            for name, distribution in self.distributions.items():
                axes.plot(x, distribution['RV_'].cdf(x), label=name)

        # Define the labels and text for the plot
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
        
        
        
        #SSSSSSS Documentation needs to be added
        x, y = self.ecdf()
        
        axes.plot([trunc(min(self.data_array)), ceil(max(self.data_array))], [trunc(min(self.data_array)), ceil(max(self.data_array))], 'k')
        
        # Plot the scatter plot for all the fitted distributions, if none, prints statement    
        if not self.distributions:
            print("No fitted distributions found.")
        else:
            for name, distribution in self.distributions.items():
                axes.scatter(x, distribution['RV_'].ppf(y), label=name)

        # Define the labels and text for the plot
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
        # Create empty dictionary to store results of KS_test in, with name as a key
        ks_results = {}
        
        # Loop over all the fitted distributions and perform KS_test
        for name, distribution in self.distributions.items():
            _, p_value = st.kstest(self.data_array, distribution['scipy_name'], args=distribution['params'])
            ks_results[name] = p_value
            print(f'The Kolmogorov-Smirnov test for the {name} distribution gives a p-value of {np.round(p_value, 3)}')
        
        return ks_results
        
    
    def tabulated_results(self):
        """Prints the tabulated results of the fitted distributions
        

        Parameters
        ----------
        self : `data_array`
            Numpy array containing time series data. 
            Note that this is already an attribute of the class.

        Returns
        -------
        tabulated_results : `pandas.DataFrame`
            Pandas dataframe containing the tabulated results of the fitted distributions.
        
        """
        
        # Define intervals for the assessment
        intervals = [0.05, 0.25, 0.5, 0.75, 0.95]
        
        # Make these interval into something that can be used as column names
        column_names = [f'{str(i)}' for i in intervals]
        
        # Set-up structure dataframe
        df = pd.DataFrame(columns=column_names)
        
        # Add row with emperical data
        for i, interval in enumerate(intervals):
            df.loc['Emperical'] = [np.round(np.percentile(self.data_array, interval*100), 3) for interval in intervals]

        # Add rows with fitted distributions
        for name, distribution in self.distributions.items():
            df.loc[name] = [np.round(distribution['RV_'].ppf(interval), 3) for interval in intervals]
            
            
        display(df)   





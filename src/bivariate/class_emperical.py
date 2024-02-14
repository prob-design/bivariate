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
        """This class is used to perform emperical statistical analysis on time series data.
        It can be used to perform:
        - statistical analysis
        - plot time series data
        - plot PDF and ECDF
        - fit distributions to the data
        - validate fitted distributions, using:
            -> graphical assessment of goodness of fit
            -> QQ-plot
            -> Kolmogorov-Smirnov test
            -> tabulated results of the fitted distributions

        
        Parameters
        ----------
        time_series_data : `numpy.array`
            Numpy array containing time series data. This data is used to perform
            emperical statistical analysis on. 
        data_title : `str`   
            This is the label of the data. This is used for plotting purposes.
        data_units : `str`
            This is the unit of the data. This is used for plotting purposes.
            
            
        Examples
        --------
        If one has a timeseries of pressures in the csv file 'data.csv' and wants to perform 
        statistical operations on it, one can use the following code:

        >>> data_array = np.genfromtxt('data.csv', delimiter=',')
        >>> title = "Pressure measurements Pipe A"
        >>> units = "Pa"
        >>> Emperical_data(time_series_data = data_array
                            , data_title = title
                            , data_units = units)
        
        """
        
        # Assign input to attributes
        self.data_array = time_series_data
        self.data_title = data_title
        self.data_units = data_units

        # Empty attributes
        self.statistical_summary = None
        
        # Create empty dictionary where the distribution dictionaries can be stored in 
        self.distributions = {}             

    def data_summary(self) -> None:
        """Generates statistical summary of inputted data.
        
        This function generates a statistical summary of the inputted data.
        This is done by using scipy.stats.describe. 
        
        
        Parameters
        ----------
        self : object
            The instance of the class.
        
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
        self : object
            The instance of the class.
        ax : `matplotlib.axes.Axes`
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
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))

        # Plot the time series data
        ax.plot(self.data_array, label = f'{self.data_title}')
        
        
        # Plotting configurations
        ax.set_title(f"Time series plot of {self.data_title}")
        ax.set_xlabel(r"$Time$")
        ax.set_ylabel(f"{self.data_units}")
        ax.legend()
        
        return fig, ax
    
    def ecdf(self):
        """Computes the empirical cumulative distribution function
        
        Parameters
        ----------
        self : object
            The instance of the class.
            
        Returns
        -------
        x : `numpy.array`	
            Sorted data   #rrrrr do not like this name
        y : `numpy.array`
            Empirical cumulative distribution function.
        
        """

        x = np.sort(self.data_array)
        n = x.size
        y = np.arange(1, n+1) / (n+1)
        
        return [x, y]
    
    def PDF_and_ECDF_plot(self,
                           axes=None,
                           fig=None):
        """Plot PDF and ECDF of the time series data

        Parameters
        ----------
        self : object
            The instance of the class.
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
    
    def fitting_distributions(self, distribution_names: List[str], fit_method: str = 'MLE'):
        """Fit distributions to the data

        This function uses scipy.stats to fit distributions to the data.
        All scipy.stats distributions are supported, but exampeles are:
        - Normal distribution
        - Lognormal distribution
        - Exponential distribution
        - Weibull distribution
        
        All 
        
        Parameters
        ----------
        self : `data_array`
            Numpy array containing time series data. 
            Note that this is already an attribute of the class.
        name_distribution : `str`
            Name of the distribution to be fitted.
        fit_method : `str`
            Method of fitting the distribution to the data.
            Options are:
            - 'MLE' Maximum Likelyhood Estimator
            - 'MM'  Method of moments
        
        Returns
        -------


        Examples
        --------
        If timeseries data is already assigned to the class, a variety
        of distributions can be fitted to the data. 
        
        This is done for exponential and lognormal by using the following code:
        >>> object.fitting_distributions(['exponential', 'lognormal'])
        
        Any distribution can be accessed by using the following code:
        >>> object.distributions['name_distribution']
        
        Which returns a dictionary containing:
        - 'params': fitted parameters are stored
        - 'RV_'   : instance of the rv_continuous_frozen class is stored
        - 'scipy_name': name used in scipy, done for operations
        - 'method_of_fitting': method used for fitting the distribution
        
        Note that the 'RV_' is a rv_continuous_frozen instance, which can be used
        to perform operations on the distribution. 
        Please see the scipy.stats documentation for more information.
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.rv_continuous.html
        """
        distribution_mapping = {
            'lognormal': st.lognorm,
            'gumbel': st.gumbel_r,
            'exponential': st.expon,
            'weibull': st.weibull_min,
            'normal': st.norm,
            'gamma': st.gamma
            # Add more distributions if needed
        }
        
        
        # This is necessary because of name convention in other packages
        distr_scipy_names = {
            'lognormal': 'lognorm',
            'gumbel': 'gumbel_r',
            'exponential': 'expon',
            'weibull': 'weibul_min',
            'normal': 'norm',
            'gamma': 'gamma'
            # Add more distributions if needed
        }
        
        # Loop over provided list of distributions
        for name in distribution_names:
            if name not in distribution_mapping:
                raise ValueError(f"Unsupported distribution: {name}")

            # Obtain correct distribution class, this use the name 
            # provided when creating the instance, and then obtains
            # the distribution rv_continuous class by using the 
            # distribution_mapping dictionary
            distribution_class = distribution_mapping[name]
            
            
            # Fit the distribution to the emperical data
            # Different methods can be used
            params = distribution_class.fit(self.data_array, method = fit_method)
            
            # Create a 'rv'/rv_continous_frozen instance using
            # the parameters from the above fitted distribution
            rv = distribution_class(*params)
            
            
            # Add the random variable into the {distribtions} dictionary
            # An example is given for a 'gamma' distribution:
            # The random variable is stored as a dictionary, in the {distribution} dictionary:
            #     - self.distributions['gamma']
            # This dictionary contains:
            #     - 'params': fitted parameters are stored
            #     - 'RV_'   : instance of the rv_continuous_frozen class is stored
            #     - 'scipy_name': name used in scipy, done for operations
            self.distributions[name] = {'params': params, 'RV_': rv, 'scipy_name': distr_scipy_names[name]
                                        , 'method_of_fitting': fit_method}

 
        
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
    
    
        # Plot the CDF's of all the fitted distributions, if none, prints statement    
        if not self.distributions:
            print("No fitted distributions defined.")
        else:
            # Loops over all items (which are also dictionaries) in the {distributions} dictionary
            # It extracts:
            #  - name of the speficifc distribution 
            #  - dictionary containing all information for that specific distribution
            for distribution_name, distribution_dict in self.distributions.items():
                axes.plot(x, distribution_dict['RV_'].cdf(x), label=distribution_name)

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
            # Loops over all items (which are also dictionaries) in the {distributions} dictionary
            # It extracts:
            #  - name of the speficifc distribution 
            #  - dictionary containing all information for that specific distribution
            for distribution_name, distribution_dict in self.distributions.items():
                axes.scatter(x, distribution_dict['RV_'].ppf(y), label=distribution_name)

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
        
        # Loop over all the fitted distributions and perform KS_test
        if not self.distributions:
            print("No fitted distributions found.")
        else:
            for distribution_dict_name, distribution_dict in self.distributions.items():
                _, p_value = st.kstest(self.data_array, distribution_dict['scipy_name'], args=distribution_dict['params'])
                print(f'The Kolmogorov-Smirnov test for the {distribution_dict_name} distribution gives a p-value of {np.round(p_value, 3)}')

    
    def tabulated_results(self,
                          intervals = None):
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
        
        # Define intervals for the assessment if they are not specified by the user
        if intervals is None:    
            intervals = [0.05, 0.25, 0.5, 0.75, 0.95]
        
        # Make these interval into something that can be used as column names
        
        #rrrrrrrrrrr Is this correct?
        column_names = [f'F(x) = {str(i)}' for i in intervals]
        
        # Set-up structure dataframe
        df = pd.DataFrame(columns=column_names)
        
        # Add row with emperical data
        for i, interval in enumerate(intervals):
            df.loc['Emperical'] = [np.round(np.percentile(self.data_array, interval*100), 3) for interval in intervals]

        # Add rows with fitted distributions
        for name, distribution in self.distributions.items():
            df.loc[name] = [np.round(distribution['RV_'].ppf(interval), 3) for interval in intervals]
            
            
        display(df)   




    @staticmethod
    def set_TUDstyle() -> Dict[str, str]:
        TUcolor = {"cyan":"#00A6D6",
                   "pink":"#EF60A3",
                   "green":"#6CC24A",
                   "yellow":"#FFB81C",
                   "blue":"#0076C2",
                   "purple":"#6F1D77",
                   "lightcyan":"#00B8C8",
                   "orange":"#EC6842",
                   "darkgreen":"#009B77",
                   "darkred":"#A50034",
                   "red":"#E03C31",
                   "darkblue":"#0C2340"}
        plt.rcParams.update({'axes.prop_cycle': plt.cycler(color=TUcolor.values()),
                             'font.size': 16, "lines.linewidth": 4})
        return TUcolor


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

from mpl_toolkits.mplot3d import axes3d
from scipy.optimize import fsolve
import pyvinecopulib as pyc

# This class is made by Siemen Algra, based on study material provided by the MUDE teaching team and Benjamin Rouse

class Region_of_interest():
    def __init__(self,
                 function,
                 random_samples : np.array = None
                 ):
        """
        Region_of_interest class for 2 dimensional probabilistic analysis.
        This class is used to define a Region of interest. 
        This region can be any arbitrary shape, and is defined by the user using a function.
        
        Random samples are given as input, and the region of interest is defined by the user.
        It is then checked how many of the random samples are within the region of interest.
        
        Parameters
        ----------
        function : function
            Function that defines the region of interest.
            The function should return True if the point is inside the region of interest, and False if it is not.
        random_samples : np.array
            Q-Dimensional array of random samples.
            Each row is 1 random sample of the Q-Dimensional distribution
            1st column is X1, 2nd column is X2,...., Qth column = Q.
            
        """
        self.function = function
        self.random_samples = random_samples
    
        
    def plot_function(self, 
                      xy_lim = None,
                      axes=None,
                      fig=None):
        """Plot the function region of interest.

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
            fig, ax = plt.subplots(1, 1, figsize=(14, 5))
        
        # Plot the function
        x = np.linspace(xy_lim[0], xy_lim[1], 100)
        y = np.linspace(xy_lim[2], xy_lim[3], 100)
        X, Y = np.meshgrid(x, y)
        Z = self.function(X, Y)
        ax.contour(X, Y, Z, levels=[0], linewidths=2, colors='r')
        
        xlim = [xy_lim[0], xy_lim[1]]
        ylim = [xy_lim[2], xy_lim[3]]
        
        ax.set_aspect("equal")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel(r"$X_1$", fontsize=14)
        ax.set_ylabel(r"$X_2$", fontsize=14)
        ax.set_title("Region of interest", fontsize = 14)
        
        return fig, ax
    
    def inside_function(self):
        """
        Check how many random samples are inside the function region of interest.
        
        Parameters
        ----------
        random_samples : np.array
            Q-Dimensional array of random samples.
            Each row is 1 random sample of the Q-Dimensional distribution
            1st column is X1, 2nd column is X2,...., Qth column = Q.
            
        Returns
        -------
        list
            list of boolean values, indicating if the random samples are inside the function region of interest.
            True if the random sample is inside the function region of interest, False if it is not.
        
        """
        
        # Check how many random samples are inside the function region of interest
        inside_random_samples = self.function(self.random_samples[:, 0], self.random_samples[:, 1]) <= 0
        
        # Create a dictionary to store the random samples inside the function region of interest
        self.function_dict = {'inside_random_samples': None}
        
        # Store the random samples inside the function region of interest in the dictionary
        self.function_dict['inside_random_samples'] = inside_random_samples


        # Calculate the amount of random samples inside the function region of interest
        # Be aware that these are stored as boolean values, so we need to sum them to get the amount
        amount_inside = np.sum(inside_random_samples)

        # Calculate the percentage of random samples inside the function region of interest
        percentage_inside = amount_inside / self.random_samples.shape[0]
        
        self.function_percentage_inside = percentage_inside

        return amount_inside, percentage_inside
    
    def plot_inside_function(self,
                             xy_lim = None,
                             axes=None,
                             fig=None):
        """
        Plot all the random samples, and highlight the ones inside the function region of interest.
        
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
        
        samples_inside_function = self.function_dict['inside_random_samples']
        
        # Create figure and axes if not provided
        if axes is None:
            fig, axes = plt.subplots(1, 1, figsize=(14, 5))
        
        # Plot the random samples
        axes.scatter(self.random_samples[:,0], self.random_samples[:,1],alpha=0.5, label='All samples')
        # Plot the random samples inside the function region of interest
        axes.scatter(self.random_samples[samples_inside_function, 0], self.random_samples[samples_inside_function, 1], color='pink', label='Samples inside')
        
        # Plot the function
        x = np.linspace(xy_lim[0], xy_lim[1], 100)
        y = np.linspace(xy_lim[2], xy_lim[3], 100)
        X, Y = np.meshgrid(x, y)
        Z = self.function(X, Y)
        axes.contour(X, Y, Z, levels=[0], linewidths=6, colors='r')
        
        
        # Set the x and y limits of the plot
        xlim = [xy_lim[0], xy_lim[1]]
        ylim = [xy_lim[2], xy_lim[3]]
        
        axes.set_xlim(xlim)
        axes.set_ylim(ylim)
        axes.legend()
        axes.set_xlabel(f'X1')
        axes.set_ylabel(f'X2')
        axes.set_title(f"Random samples inside the function region of interest \n {self.function_percentage_inside*100:.2f}% samples inside")
        axes.set_aspect("equal")
        axes.grid(True)
        axes.legend()   
        
        
        
        
        

        

class Bivariate():
    def __init__(self, 
                 rv_array: np.array, 
                 copulas_array: np.array,
                 conditional_copulas: np.array = None):
        """
        Bivariate class for 2 dimensional probabilistic analysis.
        
        This class is used to create a bivariate model based on the 2 given random variables and a copula.
        
        Parameters
        ----------
        rv_array : list
            list of rv_continuous_frozen objects
            Corresponding to the different random variables in the multivariate model.
            rv_array = [X1, X2, ...]
        copulas_array : list
            list of pyvinecopulib.Bicop objects
            Corresponding to the different copulas in the multivariate model.
            copulas_array = [copula12, copula23, ...]
            copula12 is the copula between rv1 and rv2, etc.
        conditional_copulas : list
            list of pyvinecopulib.Bicop objects
            Corresponding to the different conditional copulas in the multivariate model.
            conditional_copulas = [cond_copula13|2, ...]
            cond_copula13|2 is the conditional copula between rv1 and rv3 given rv2.
        
        
        """

        # Assign input to attributes
        self.rv_array = rv_array
        self.copulas_array = copulas_array
        self.conditional_copulas = conditional_copulas
        
    
    def random_samples(self,
                       size=None):
        """
        Generates random samples from the copula. 
        Since samples are in the unit interval, 
        they need to be transformed to the original space using the percent point function (ppf).
        
        Parameters
        ----------
        size : int
            Number of samples to generate.
            
        Returns
        -------
        np.array of shape (size, 2)
            2D array of random samples from the copula.
                    
        """
        
        # Generate random samples from the copula, using pyvinecopulib.Bicop.simulate
        samples = self.copulas_array.simulate(size)
        
        # Transform the samples back to the original space of RV using the percent point function (ppf)
        samples_X1 = self.rv_array[0].ppf(samples[:,0])
        samples_X2 = self.rv_array[1].ppf(samples[:,1])
        
        # Combine the samples into a 2D array
        samples_X1X2 = np.vstack([samples_X1, samples_X2]).T
        
        
        # Assign the samples to the attribute
        self.samples_X1X2 = samples_X1X2
        
        return samples_X1X2
    
        
    
    def plot_random_samples(self, axes=None, fig=None):
        """Plot random samples from the copula.

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
            fig, axes = plt.subplots(1, 1, figsize=(14, 5))

        
        # Plot the random samples
        axes.scatter(self.samples_X1X2[:,0], self.samples_X1X2[:,1], label='samples')
        axes.set_aspect("equal")
        axes.set_xlabel(f'X1')
        axes.set_ylabel('X2')
        axes.set_title('Random samples from the copula')
        axes.legend()
        axes.grid(True)
        
        
        
        return fig, axes
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
                 type: str = None,
                 random_samples : np.array = None
                 ):
        """
        Area_of_interest class for 2 dimensional probabilistic analysis.
        This class is used to define a Region of interest. 
        This region can be any arbitrary shape, and is defined by the user.
        
        Random samples are given as input, and the region of interest is defined by the user.
        It is then checked how many of the random samples are within the region of interest.
        
        Parameters
        ----------
        type : str
            Type of region of interest.
            Can be:
            - 'rectangle'
            - 'polygon'
            - 'function'
        random_samples : np.array
            Q-Dimensional array of random samples.
            Each row is 1 random sample of the Q-Dimensional distribution
            1st column is X1, 2nd column is X2,...., Qth column = Q.
            
        """
        self.type = type
        self.random_samples = random_samples
        
        
    def define_rectangle(self, x_left_lower, y_left_lower, x_right_upper, y_right_upper):
        """
        Define a rectangle region of interest.
        
        Parameters
        ----------
        x_left_lower : float
            X-coordinate of the left lower corner of the rectangle.
        y_left_lower : float
            Y-coordinate of the left lower corner of the rectangle.
        x_right_upper : float
            X-coordinate of the right upper corner of the rectangle.
        y_right_upper : float
            Y-coordinate of the right upper corner of the rectangle.
        
        """
        # Create a dictionary to store the rectangle coordinates
        self.rectangle_dict = {'x_left_lower': None
                               , 'y_left_lower': None
                               , 'x_right_upper': None
                               , 'y_right_upper': None}
        
        
        # Assign rectangle coordinates to dictionary
        self.rectangle_dict['x_left_lower'] = x_left_lower
        self.rectangle_dict['y_left_lower'] = y_left_lower
        self.rectangle_dict['x_right_upper'] = x_right_upper
        self.rectangle_dict['y_right_upper'] = y_right_upper
 
    
    def plot_rectangle(self, 
                       xy_lim = None,
                       axes=None,
                       fig=None):
        """Plot the rectangle region of interest.

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
        
   
        # Plot the rectangle
        rect = plt.Rectangle((self.rectangle_dict['x_left_lower'], self.rectangle_dict['y_left_lower']),
                                self.rectangle_dict['x_right_upper'] - self.rectangle_dict['x_left_lower'],
                                self.rectangle_dict['y_right_upper'] - self.rectangle_dict['y_left_lower'],
                                linewidth=1, edgecolor='r', facecolor='r')
        ax.add_patch(rect)
        
        
        xlim = [xy_lim[0], xy_lim[1]]
        ylim = [xy_lim[2], xy_lim[3]]
        
        ax.set_aspect("equal")
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_xlabel(r"$X_1$", fontsize=14)
        ax.set_ylabel(r"$X_2$", fontsize=14)
        ax.set_title("Region of interest", fontsize = 14)
        
        return fig, ax
        
    def _is_inside_rectangle(self, x1, x2):
        
        # Define coordinates of the bottom-left and top-right corners of the square
        x_left_lower = self.rectangle_dict['x_left_lower']
        x_right_upper = self.rectangle_dict['x_right_upper']
        y_left_lower =  self.rectangle_dict['y_left_lower']
        y_right_upper = self.rectangle_dict['y_right_upper']
    
        return x_left_lower <= x1 <= x_right_upper and y_left_lower <= x2 <= y_right_upper

    
    def inside_rectangle(self):
        """
        Check how many random samples are inside the rectangle.
        
        Parameters
        ----------
        random_samples : np.array
            Q-Dimensional array of random samples.
            Each row is 1 random sample of the Q-Dimensional distribution
            1st column is X1, 2nd column is X2,...., Qth column = Q.
            
        Returns
        -------
        int
            Number of random samples inside the rectangle.
        
        """
        
        
        
        # Check how many random samples are inside the rectangle
        inside_random_samples = self._is_inside_rectangle(self.random_samples[:, 0], self.random_samples[:, 1]) <= 0
        
        # Plot the random samples inside the rectangle
        # ax.scatter(random_samples[inside_indices, 0], random_samples[inside_indices, 1], color='b', label='inside')
        
        self.rectangle_dict['inside_random_samples'] = inside_random_samples

        # Calculate the amount of random samples inside the rectangle
        amount_inside = len(inside_random_samples)
        # Calculate the percentage of random samples inside the rectangle
        percentage_inside = amount_inside / random_samples.shape[0]

        return amount_inside, percentage_inside
    
    def plot_inside_rectangle(self,
                              random_samples=None,
                              axes=None,
                              fig=None):
        """Plot all the random samples, and highlight the ones inside the rectangle.
        
        Parameters
        ----------
        self : object
            The instance of the class.
        random_samples : np.array
            Q-Dimensional array of random samples.
            Each row is 1 random sample of the Q-Dimensional distribution
            1st column is X1, 2nd column is X2,...., Qth column = Q.
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
        
        inside_indices = self.rectangle_dict['inside_indices']
        
        # Create figure and axes if not provided
        if axes is None:
            fig, axes = plt.subplots(1, 1, figsize=(14, 5))
            
        axes.scatter(self.random_samples[:,0], self.random_samples[:,1],alpha=0.5, label='samples')
        axes.scatter(random_samples[inside_indices, 0], random_samples[inside_indices, 1], color='b', label='inside')
        
        axes.legend()
        axes.set_xlabel(f'X1')
        axes.set_ylabel(f'X2')

        
        
        
        

        

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
        axes.set_xlabel(f'X1')
        axes.set_ylabel('X2')
        
        
        return fig, axes
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
import matplotlib.cm as cm



from mpl_toolkits.mplot3d import axes3d
from sklearn.neighbors import KernelDensity
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
    
        # Create empty dictionary to store the emperical contour plot values
        self.emperical_contour = {}
    
        
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
        inside_random_samples = self.function(self.random_samples[:, 0], self.random_samples[:, 1]) 
        
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
    
    # def plot_emperical_contours(self,
                                xy_lim = None,
                                axes=None,
                                fig=None,
                                bandwidth=0.5):
        """
        Plot the emperical contours of the random samples using the KernelDensity.
        
        Parameters
        ----------
        self : object
            The instance of the class.
        axes : `matplotlib.axes.Axes`
            Axes object to plot on. If not provided, a new figure will be created.
        fig : `matplotlib.figure.Figure`
            Figure object to plot on. If not provided, a new figure will be created.
        bandwidth : float
            Bandwidth of the kernel density estimation.
            
            'The bandwidth parameter controls the smoothness of the estimated density function.
            Higher bandwidth values result in smoother density estimates, 
            while lower bandwidth values capture more detail but may be more 
            sensitive to noise in the data.' from ChatGPT
        
        Returns
        -------
        fig : `matplotlib.figure.Figure`
            Figure object containing the plot.
        ax : `matplotlib.axes.Axes`
            Axes object containing the plot.
        
        """

        # Assuming 'samples' is a 2D array with your random samples (shape: (n_samples, 2))
        # For example, you might have generated these samples using a copula as discussed earlier

        # Extract the X1 and X2 coordinates from the samples
        x1 = self.random_samples[:, 0]
        x2 = self.random_samples[:, 1]

        # Define the number of points for the grid
        num_points = 100

        # Create a grid of points for contour plot
        X1_values = np.linspace(min(x1), max(x1), num_points)
        X2_values = np.linspace(min(x2), max(x2), num_points)
        X, Y = np.meshgrid(X1_values, X2_values)
        grid = np.vstack([X.flatten(), Y.flatten()]).T

        # Estimate the density using kernel density estimation
        kde = KernelDensity(bandwidth=bandwidth)  # Adjust the bandwidth as needed
        kde.fit(self.random_samples)

        # Compute the log-density values on the grid
        log_density_values = kde.score_samples(grid)
        density_values = np.exp(log_density_values)

        # Reshape the density values to match the shape of the grid
        density_values = density_values.reshape(X.shape)
        
        # Create a dictionary to store the necessary values for the emperical contour plot
        self.emperical_contour_dict = {'X1_values': X1_values,
                                  'X2_values': X2_values,
                                  'density_values': density_values}      
        
        
        if axes is None:
            fig, axes = plt.subplots(1, 1, figsize=(14, 5))
        
        # Plot the contour plot of the emperical joint density   
        axes.contour(X, Y, density_values, levels=8, cmap=cm.Blues, alpha=0.8)
        # Plot the random samples
        axes.scatter(x1, x2, alpha=0.01, label='Samples', color='gray') 
        
        if xy_lim is None:
            xy_lim = [min(x1), max(x1), min(x2), max(x2)]
            
        # Set the x and y limits of the plot
        axes.set_xlim(xy_lim[0], xy_lim[1])
        axes.set_ylim(xy_lim[2], xy_lim[3])
         
        axes.set_xlabel('X1')
        axes.set_ylabel('X2')
        axes.set_title('Emperical Contour Plot of Sampled Data')
        axes.legend()  
        axes.grid(True)
        axes.set_aspect("equal")

        plt.show()
    
    
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
        axes.scatter(self.random_samples[:,0], self.random_samples[:,1],alpha=0.01, 
                     label='All samples', color = 'gray')
        
        # Plot the random samples inside the function region of interest
        # Notice that we use the boolean values to select the samples inside the function region of interest
        axes.scatter(self.random_samples[samples_inside_function, 0], self.random_samples[samples_inside_function, 1],
                     color='red', alpha=0.01, label='Samples inside', zorder=2)
        
        
        # Plot the function
        x = np.linspace(xy_lim[0], xy_lim[1], 100)
        y = np.linspace(xy_lim[2], xy_lim[3], 100)
        X, Y = np.meshgrid(x, y)
        Z = self.function(X, Y)
        axes.contour(X, Y, Z, levels=[0], linewidths=2.5, colors='black', zorder = 3)
        
        # If emperical_contour_dict exists, plot the emperical contour plot
        if 'emperical_contour_dict' in self.__dict__: 
            X1_values = self.emperical_contour_dict['X1_values']
            X2_values = self.emperical_contour_dict['X2_values']
            density_values = self.emperical_contour_dict['density_values']
            axes.contour(X1_values, X2_values, density_values, levels=8, cmap=cm.Blues, alpha=0.8, zorder = 4)
 
        
        # Set the x and y limits of the plot
        xlim = [xy_lim[0], xy_lim[1]]
        ylim = [xy_lim[2], xy_lim[3]]
        
        axes.set_xlim(xlim)
        axes.set_ylim(ylim)
        axes.set_zorder(4)
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
    
    def bivariatePdf(self, point: list[float]) -> float:
        '''
        Computes the bivariate joined probability density function evaluated in X1 and X2.

        point: list. Coordinates at which the bivariate PDF is evaluated.
        '''
        # Compute the ranks of the coordinates
        u0 = self.rv_array[0].cdf(point[0])
        u1 = self.rv_array[1].cdf(point[1])

        pdf = float(self.copulas_array.pdf([[u0, u1]])) * self.rv_array[0].pdf(point[0]) * self.rv_array[1].pdf(point[1])
        return pdf    
    
    def plot_histogram_2D(self):
        """
        Plot a 2D histogram of the random samples.
        This is a simple way to visualize the distribution of the random samples.
        
        Parameters
        ----------
        self : object
            The instance of the class.
        
        
        """
        # Extract the X1 and X2 coordinates from the samples
        x1 = self.samples_X1X2[:, 0]
        x2 = self.samples_X1X2[:, 1]

        # Define the number of bins for each dimension
        num_bins_x1 = 50
        num_bins_x2 = 50

        # Create the 2D histogram
        hist, x_edges, y_edges = np.histogram2d(x1, x2, bins=(num_bins_x1, num_bins_x2))

        # Create meshgrid for 3D plotting
        X, Y = np.meshgrid(x_edges[:-1], y_edges[:-1])

        # Plot the 3D histogram
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.bar3d(X.flatten(), Y.flatten(), np.zeros_like(X.flatten()), 
                (x_edges[1]-x_edges[0]), (y_edges[1]-y_edges[0]), hist.flatten()
                , zsort='average', color='blue')

        ax.set_xlabel('X1')
        ax.set_ylabel('X2')
        ax.set_zlabel('Frequency', labelpad=0.5)
        ax.set_title('2D Histogram of Random Samples')
        ax.grid(True)
        
        plt.show()

    
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


        # Define everything needed for the THEORETICAL contour plot
        X1_values = np.linspace(min(self.samples_X1X2[:,0]), max(self.samples_X1X2[:,0]), 100)  # Adjust the range as needed
        X2_values = np.linspace(min(self.samples_X1X2[:,1]), max(self.samples_X1X2[:,1]), 100)  # Adjust the range as needed
        X, Y = np.meshgrid(X1_values, X2_values)
        grid = np.vstack([X.flatten(), Y.flatten()]).T

        # Compute the PDF values on the grid
        pdf_values = np.array([self.bivariatePdf([x, y]) for x, y in grid])
        pdf_values = pdf_values.reshape(X.shape)
        
        # Plot the contour plot of the joint PDF
        axes.contour(X, Y, pdf_values, levels = 8, cmap=cm.Blues,
                     alpha=0.8)#, label = 'Theoratical contour lines')
        
        # Plot the random samples
        axes.scatter(self.samples_X1X2[:,0], self.samples_X1X2[:,1],
                     alpha = 0.01, label='samples', color = 'gray')

        axes.set_aspect("equal")
        axes.set_xlabel(f'X1')
        axes.set_ylabel('X2')
        axes.set_title('Random samples from the copula \n and the theoretical contour plot')
        axes.legend()
        axes.grid(True)
        
        plt.show()
        
        
        
        return fig, axes
    
    
class Multivariate():
    def __init__(self, 
                 rv_array: np.array, 
                 copulas_array: np.array,
                 conditional_copulas: np.array = None):
        """
        Multivariate class for Q dimensional probabilistic analysis.
        
        This class is used to create a multivariate model based on the Q given random variables and a copula.
        
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
        
    
    def random_samples_3D(self,
                       size=None):
        """
        Generates a random sample of the bivariate copula of X0 and X2.

        cop1: pyvinecopulib.Bicop. Copula between variables X0 and X1.
        cop2: pyvinecopulib.Bicop. Copula between variables X1 and X2.
        cond_cop: pyvinecopulib.Bicop. Conditional copula between X0 and X2 given X1.
        (default: independent copula).
        n: int. Number of samples.
        
        
        ---------------
        See sampling procedure in http://dx.doi.org/10.1016/j.insmatheco.2007.02.001
                      
        """
        
        # Generate random samples from the copulas, notice this is in univariate space
        u = np.random.rand(size, 3)
        x0 = u[:, 0]
        x1 = self.copulas_array[0].hinv1(np.concatenate((x0.reshape(-1, 1),
                                        u[:, 1].reshape(-1, 1)), axis=1))
        a = self.copulas_array[0].hfunc1(np.concatenate((x0.reshape(-1, 1),
                                        x1.reshape(-1, 1)), axis=1))
        b = self.conditional_copulas[0].hinv1(np.concatenate((a.reshape(-1, 1),
                                        u[:, 2].reshape(-1, 1)), axis=1))
        x2 = self.copulas_array[1].hinv1(np.concatenate((x1.reshape(-1, 1),
                                        b.reshape(-1, 1)), axis=1))
        random_samples_univariate = np.concatenate((x0.reshape(-1, 1),
                            x1.reshape(-1, 1), x2.reshape(-1, 1)), axis=1)
        
        # Transform the samples back to the original space of RV using the percent point function (ppf)
        samples_X0 = self.rv_array[0].ppf(random_samples_univariate[:,0])
        samples_X1 = self.rv_array[1].ppf(random_samples_univariate[:,1])
        samples_X2 = self.rv_array[2].ppf(random_samples_univariate[:,2])
        
        
        # Combine the samples into a 3D array
        random_samples = np.vstack([samples_X0, samples_X1, samples_X2]).T
        self.random_samples = random_samples
        
        return random_samples
    
    def plot_random_samples_3D(self,xyz_lim, axes=None, fig=None):
        """Plot random samples in 3D space.
        
    
        Parameters
        ----------
        self : object
            The instance of the class.
        
        """
        
        # Extract the x, y, and z coordinates from the array
        x = self.random_samples[:, 0]
        y = self.random_samples[:, 1]
        z = self.random_samples[:, 2]

        # Create a 3D plot
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Plot the points as scatter plot
        ax.scatter(x, y, z, alpha=0.2)

        # Set labels for the x, y, and z axes
        ax.set_xlim(xyz_lim[0], xyz_lim[1])
        ax.set_ylim(xyz_lim[2], xyz_lim[3])
        ax.set_zlim(xyz_lim[4], xyz_lim[5])
        ax.set_xlabel('X1')
        ax.set_ylabel('X2')
        ax.set_zlabel('X3')
        ax.set_title('3D Scatter Plot of random samples')
        ax.set_box_aspect([1, 1, 1])
        ax.grid(True)
        ax.legend()

        

class Region_of_interest_3D():
    def __init__(self,
                 function,
                 random_samples : np.array = None
                 ):
        """
        Region_of_interest class for 3 dimensional probabilistic analysis.
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


    def inside_function_3d(self):
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
        inside_random_samples = self.function(self.random_samples[:, 0], self.random_samples[:, 1], self.random_samples[:, 2])
        
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
    
    def plot_inside_function_3d(self, xyz_lim=None, axes=None, fig=None):
        """
        Plot all the random samples, and highlight the ones inside the function region of interest.
        
        Parameters
        ----------
        self : object
            The instance of the class.
        axes : `mpl_toolkits.mplot3d.Axes3D`, optional
            Axes object to plot on. If not provided, a new figure will be created.
        fig : `matplotlib.figure.Figure`, optional
            Figure object to plot on. If not provided, a new figure will be created.
            
        Returns
        -------
        fig : `matplotlib.figure.Figure`
            Figure object containing the plot.
        ax : `mpl_toolkits.mplot3d.Axes3D`
            Axes object containing the plot.
        
        """
        
        samples_inside_function = self.function_dict['inside_random_samples']
        
        # Create figure and axes if not provided
        if axes is None:
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
        
        # Plot the random samples
        ax.scatter(self.random_samples[:,0], self.random_samples[:,1], self.random_samples[:,2],
                   alpha=0.2, label='All samples')
        # Plot the random samples inside the function region of interest
        ax.scatter(self.random_samples[samples_inside_function, 0], 
                   self.random_samples[samples_inside_function, 1], 
                   self.random_samples[samples_inside_function, 2], 
                   color='red', label='Samples inside', alpha=0.2)
        
        # # Plot the function
        # x = np.linspace(xyz_lim[0], xyz_lim[1], 100)
        # y = np.linspace(xyz_lim[2], xyz_lim[3], 100)
        # z = np.linspace(xyz_lim[4], xyz_lim[5], 100)
        # X, Y, Z = np.meshgrid(x, y, z)
        # Z_function = self.function(X, Y, Z)
        # ax.contour3D(X, Y, Z, levels=Z_function.flatten(), cmap='viridis')
        
        # Set the x and y limits of the plot
        ax.set_xlim(xyz_lim[0], xyz_lim[1])
        ax.set_ylim(xyz_lim[2], xyz_lim[3])
        ax.set_zlim(xyz_lim[4], xyz_lim[5])
        ax.set_xlabel('X1')
        ax.set_ylabel('X2')
        ax.set_zlabel('X3')
        ax.set_box_aspect([1, 1, 1])
        ax.set_title(f'Random samples inside the function region of interest \n {self.function_percentage_inside*100:.2f}% samples inside')
        ax.grid(True)
        ax.legend()
        
        
    @staticmethod
    def set_Colorstyle() -> Dict[str, str]:
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
        
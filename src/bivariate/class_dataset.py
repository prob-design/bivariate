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


from bivariate.class_fitresults import FitResults

# Types for documentation
T = TypeVar('T', bound='Dataset')

class Dataset():

    distributions = ["Normal", "Exponential", "Lognormal", "Logistic"]
    
    # Constructors

    
    def __init__(
        self,
        dataframe: pd.DataFrame,
        cols: Optional[List[str]] = None,
        col_labels = None,
    ) -> None:
        """
        Parameters
        ----------
        dataframe : pandas.core.frame.DataFrame
            Dataframe that holds the data. Has to contain one datetime column.
        cols : list[str], default None
            List of columns to use. If None, use all columns.
        col_labels : list[str], list[list[str], list[str]] or dict, default None
            List of optional descriptive labels to use for the selected columns.
            Long labels are used for titles and axes, short labels for legends.
            If list[str], use given strings as both
            If list[list[str], list[str]], use first list as long labels, 
            second list for short labels
            If dict, key 'long' and 'short' should point to list[str]
            If none, use default column names.
        """
        
        # Main dataframe
        self.dataframe = dataframe
        
        # Columns to use
        self._time_col = self.find_datetime_col(self.dataframe)
        
        if cols is not None:
            self._cols = cols
        elif self._time_col is not None:
            self._cols = list(dataframe.drop(columns=self._time_col).columns)
        else:
            self._cols = list(dataframe.columns)

        # Descriptive columns labels
        self._col_labels = {}
        self._col_labels["short"] = self._cols.copy()
        self._col_labels["long"] = self._cols.copy() # nothing given
        
        if not col_labels:
            pass
        
        elif type(col_labels) == dict:
            if len(col_labels["short"]) == len(self._cols) and \
            len(col_labels["long"]) == len(self._cols):
                self._col_labels["long"] = col_labels["long"]
                self._col_labels["short"] = col_labels["short"]
            else:
                warnings.warn("No. of col_labels does not match no. of cols,\
                    using defaults from dataframe", UserWarning)
        
        elif type(col_labels[0]) == str: # list[str] given, use for both
            if len(col_labels) == len(self._cols):
                self._col_labels["long"] = col_labels
                self._col_labels["short"] = col_labels
            else:
                warnings.warn("No. of col_labels does not match no. of cols,\
                    using defaults from dataframe", UserWarning)
        
        elif type(col_labels[0]) == list:
            if len(col_labels[0]) == len(self._cols) and \
            len(col_labels[1]) == len(self._cols):
                self._col_labels["long"] = col_labels[0]
                self._col_labels["short"] = col_labels[1]
            else:
                warnings.warn("No. of col_labels does not match no. of cols,\
                    using defaults from dataframe", UserWarning)
        
        else:
            warnings.warn("Given col_labels does not have the right format,\
                using defaults from dataframe", UserWarning)

        # Helpers
        self._ncols = len(self._cols)

        if self._time_col is not None:
            self._has_timecol = True
        else:
            self._has_timecol = False

        # To be computed
        self.extremes = None
        self._bivariate_vars = None
        self._bivar_r_norm = None
        self._cov = None
        self._cor = None

        # Empty FitResults object for every variable
        self.results = {}
        for _col in self._cols:
            self.results[_col] = FitResults()

    @classmethod
    def import_from_filename(
        cls: Type[T],
        filename: str,
        var_time: Optional[str] = None,
        cols: Optional[List[str]] = None,
        col_labels: Optional[List[str]] = None,
        **kwargs
    ) -> T:
        """ Create a Dataset object from a given filename.
        
        Parameters
        ----------
        filename : str
            Filename or path of the dataset.
        var_time : str, default None
            Name of the column containing the timestamps. If none, don't use a
            time column.
        cols : list[str], default None
            List of columns to use. If None, use all columns.
        col_labels : list[str], default None
            List of optional descriptive labels to use for the selected columns.
            If none, use default column names.
        **kwargs
            Optional keyword arguments to be passed to Pandas `read_csv`.
            
        Returns
        -------
        Dataset
            Dataset object constructed from the given filename.
        """
        if isinstance(var_time, str):
            var_time = [var_time]
            if cols:
                cols_used = var_time + cols
            else:
                cols_used = None
        elif cols:
            var_time = False
            cols_used = cols
        else:
            var_time = False
            cols_used = None

        dataframe = pd.read_csv(
            filename,
            parse_dates=var_time,
            usecols=cols_used,
            **kwargs
        )

        return cls(dataframe, cols, col_labels)


    @classmethod
    def import_from_surfdrive_path(
        cls: Type[T],
        link: str,
        path: str,
        var_time: Optional[str] = None,
        cols: Optional[List[str]] = None,
        col_labels: Optional[List[str]] = None,
        **kwargs
    ) -> T:
        """ Create a Dataset object from a SURFdrive public access and link to 
        a directory and a path of subfolders.
        
        Parameters
        ----------
        link : str
            SURFdrive public access link.
        path : str
            Path to the subfolders.
        var_time : str
            Name of the column containing the timestamps.
        cols : list[str], default None
            List of columns to use. If None, use all columns.
        col_labels : list[str], default None
            List of optional descriptive labels to use for the selected columns.
            If none, use default column names.
        **kwargs
            Optional keyword arguments to be passed to Pandas `read_csv`.

        Returns
        -------
        Dataset
            Dataset object constructed from the given path.
        """

        link += r'/download?path=%2F'
        path_lst = path.split('/')
        for s in path_lst[:-1]:
            link += s + "%2F"
        link = link[:-3]
        link += "&files=" + path_lst[-1]

        return cls.import_from_filename(
            link,
            var_time,
            cols,
            col_labels,
            **kwargs
        )


    @classmethod
    def import_from_surfdrive_file(
        cls: Type[T],
        link: str,
        var_time: Optional[str] = None,
        cols: Optional[List[str]] = None,
        col_labels: Optional[List[str]] = None,
        **kwargs
    ) -> T:
        """ Create a Dataset object from a SURFdrive public access link.        

        Parameters
        ----------
        link : str
            SURFdrive public access link to dataset.
        var_time : str
            Name of the column containing the timestamps.
        cols : list[str], default None
            List of columns to use. If None, use all columns.
        col_labels : list[str], default None
            List of optional descriptive labels to use for the selected columns.
            If none, use default column names.
        **kwargs
            Optional keyword arguments to be passed to Pandas `read_csv`.

        Returns
        -------
        Dataset
            Dataset object constructed from the given link.
        """
        
        link += "/download"
        return cls.import_from_filename(
            link,
            var_time,
            cols,
            col_labels,
            **kwargs
        )


    # Cleaning dataset

    # TODO: see issue #5 on GitLab  
    def clean_dataset(self, z_score_threshold: Union[float, int] = 5) -> None:
        """Cleans the dataset by removing NaN's and outliers. Outliers are 
        removed based on the z-score method.

        Parameters
        ----------
        z_score_threshold : float or int
            Threshold of the Z-score to determine outliers.
        """
        dataframe = self.dataframe.dropna().reset_index(drop=True)

        for col_name in self._cols:
            col = dataframe[col_name]
            col_mean = col.mean()
            col_std = col.std()
            z_score = (col - col_mean) / col_std
            col_out_idx = (
                col[np.abs(z_score) > z_score_threshold].index.values.tolist())
            dataframe = dataframe.drop(index=col_out_idx).reset_index(drop=True)
   
        self.dataframe = dataframe


    # Data exploration


    def data_summary(self) -> None:
        """Displays a summary of the dataset.
        """

        display(self.dataframe[self._cols].describe())
        

    def time_plot(
        self,
        together: Optional[bool] = False,
        zoom: Optional[Tuple[datetime]] = None,
        **kwargs
    ):
        """Plot the data against time.
        
        Parameters
        ----------
        together: bool, default False
            Plot the data together in one plot. If None, make a subplot for 
            every column.
        zoom: tuple[datetime], default None
            Datetime range to zoom into.

        Returns
        -------
        matplotlib.figure.Figure
            Figure object
        matplotlib.axes.Axes or sequence of matplotlib.axes.Axes
            Axes object or Axes objects, if ```together=False```
        """
        
        figsize = (10, 10) if together else (10, 5*len(self._cols))

        ax = self.dataframe.plot(x=self._time_col,
                          y=self._cols,
                          xlim=zoom,
                          subplots=not together,
                          sharex=True,
                          figsize=figsize,
                          #marker='-',       #rrrrrrr, had to get rid of this because of convention used in MUDE
                          #ls='none',
                          grid=True,
                          markeredgecolor='k',
                          markeredgewidth=0.25,
                          **kwargs)
        if not together:
            for i, subplot in enumerate(ax):
                subplot.set_ylabel(self._col_labels["long"][i])
                subplot.legend([self._col_labels["short"][i]])
        
        else:
            ax.set_ylabel(self._col_labels["long"])
        
        return plt.gcf(), ax


    def hist_plot(self, **kwargs):
        """Make a histogram plot of the data.
        
        Parameters
        ----------
        together: bool, default False
            Plot the data together in one plot. If None, make a subplot for 
            every column.

        Returns
        -------
        matplotlib.figure.Figure
            Figure object
        matplotlib.axes.Axes or numpy.ndarray of matplotlib.axes.Axes
            Axes object or Axes objects, if ```together=False```
        """

        figsize = (10, 5*len(self._cols))

        ax = self.dataframe.hist(
            column=self._cols,
            figsize=figsize,
            layout=(self._ncols, 1),
            edgecolor='k',
            linewidth=1.0
        )

        return plt.gcf(), ax


    # Univariate fit
        

    def plot_ecdf(self, **kwargs):          #rrrrrrr, maybe it is nice to be able to specify which column
        """Make a plot of the Empirical Cumulative Distribution Function (ECDF)
        of the selected columns.

        Returns
        -------
        matplotlib.figure.Figure
            Figure object
        numpy.ndarray[matplotlib.figure.SubFigure]
            Numpy array containing subfigures.
        """

        # Make figure scale in height with the number of columns selected
        fig = plt.figure(figsize=(25, 5*self._ncols))

        # Make a new subfigure for every variable
        subfigs = fig.subfigures(self._ncols, 1)

        # Iterate over selected columns
        for i in range(self._ncols):
            
            # Calculate ECDF
            x, f = self.ecdf(self.dataframe[self._cols[i]])
            
            # Make subplots inside the subfig
            ax = subfigs[i].subplots(1, 4)
            subfigs[i].suptitle(self._col_labels["long"][i])
            
            ax[0].step(x, f, linewidth=4, **kwargs)  # Plot F(x)
            ax[0].set_title('$F(x)$')
            ax[0].set_ylabel('Cumulative Probability')
            ax[0].grid()

            ax[1].step(x, 1 - f, linewidth=4, **kwargs)  # Plot 1-F(x)
            ax[1].set_title('$1- F(x)$')
            ax[1].grid()

            ax[2].semilogy(x, f, linewidth=4, **kwargs)  # Plot logY 1-F(x)
            ax[2].set_title('$F(x)$. Y axis log scale')
            ax[2].grid()

            ax[3].semilogy(x, 1 - f, linewidth=4, **kwargs)  # Plot logY 1-F(x)
            ax[3].set_title('$1- F(x)$. Y axis log scale')
            ax[3].grid()
        
        return fig, subfigs


    def fit_distribution(self,
                         distribution: Union[str, List[str]] = 'all') -> None:
        """Fit a distribution to the selected columns.
        
        Parameters
        ----------
        distribution : str or list[str], default 'all'
            Name of the distribution to fit. Options are Normal, Exponential,
            Logistic or Lognormal. If 'all', fit all of the above.
        """

        # Make list of distritbutions to fit
        if distribution == 'all':
            distribution = self.distributions
        elif isinstance(distribution, str):
            distribution = [distribution]
        
        # Iterate over the distributions
        for _dist in distribution:

            # SciPy distribution from argument
            sp_dist = self.scipy_dist(_dist)

            
            # Iterate over selected columns
            for _col in self._cols:

                # Fit distribution to the data
                fit_pars = sp_dist.fit(self.dataframe[_col])

                # Calculate the GoF statistics
                aic, bic = self.aic_bic(sp_dist.pdf(self.dataframe[_col],
                                                    *fit_pars),
                                        len(fit_pars),
                                        len(self.dataframe[_col]))

                # Add distribution parameters and GoF to the results object
                self.results[_col].add_distribution(_dist,
                                                    fit_pars,
                                                    aic,
                                                    bic) 
        

    def plot_fitted_distributions(self, **kwargs):
        """Plot the distributions that are fitted using ```fit_distribution```.

        Returns
        -------
        matplotlib.figure.Figure
            Figure object.
        numpy.ndarray[matplotlib.figure.SubFigure]
            Numpy array containing subfigures.
        """
        
        # Number of fitted distributions
        ndists = len(self.results[self._cols[0]].distributions_fitted())
        
        # Create figure and subfigures
        fig = plt.figure(figsize=(5*ndists, 5*self._ncols))
        subfigs = fig.subfigures(self._ncols, 1)
        
        # Iterate over the selected variables
        for i, _col in enumerate(self._cols):

            # Add subplot for every fitted dist to subfigure
            ax = subfigs[i].subplots(1, ndists)
            subfigs[i].suptitle(self._col_labels["long"][i])

            # Calculate ECDF for the current variable
            x, f = self.ecdf(self.dataframe[_col])

            # Iterate over the fitted distributions
            for j, dist in enumerate(self.results[_col].distributions_fitted()):

                # Calculate the fitted cdf from parameters
                fit_pars = self.results[_col].distribution_parameters(dist)
                fit_cdf = self.scipy_dist(dist).cdf(x, *fit_pars)
                
                # Plot the ECDF 
                ecdf_plot,  = ax[j].plot(x, f, label="ECDF", **kwargs)                

                # Plot the fitted CDF
                fitted_plot,  = ax[j].plot(x, fit_cdf,
                                           label=f"Fitted {dist} distribution")

                # Set labels
                ax[j].set_xlabel("Value")
                ax[j].set_ylabel("F(X)")

                # Add AIC and BIC to the legend of the plot
                aic, bic = self.results[_col].fit_stats(dist, ['aic', 'bic'])
                gof_string = f"AIC: {aic:.3f}\nBIC: {bic:.3f}"
                extra_legend_entry = Rectangle((0, 0), 1, 1, fc='white',
                                               fill=False, edgecolor='none',
                                               linewidth=0)

                # Set legend
                ax[j].legend([ecdf_plot, fitted_plot, extra_legend_entry],
                             ("ECDF", f"Fitted {dist} distribution",
                              gof_string))

                # Set grid
                ax[j].grid()
    

    # Extreme Value Analysis
    
    # TODO: let the user select different periods for different columns
    def create_ev(self, period: str) -> pd.DataFrame:
        """Creates a dataframe with blockwise extreme values from a given
        resampling period.
        
        Parameters
        ----------
        period : str
            The period to use for the blocks. Supported: [W]eekly, [A]nnual,
            [D]aily
            
        Returns
        -------
        pandas.core.frame.DataFrame
            Dataframe with the extreme values calculated over the given period
        """

        if self._has_timecol is False:
            raise KeyError('No datetime column found.')
        
        self.extremes = self.dataframe.resample(period[0].upper(),
                                                on=self._time_col)\
                                                .max().dropna()\
                                                .reset_index(drop=True)
        return self.extremes

        
    def fit_ev(self) -> Tuple[float, ...]:
        """Fits a Generalized Extreme Value (GEV) distribution to the calculated 
        extreme values.
        
        Returns
        -------
        tuple[float, ...] 
            Tuple containing the shape parameters of the GEV.
        """

        if self.extremes is None:
            raise Exception("No extreme values computed yet! Use create_ev \
                first.")

        # Get SciPy extreme distribution
        dist = self.scipy_dist('Extreme')

        # Iterate over the selected columns
        for _col in self._cols:

            # Fit GEV to the calculated extreme values
            fit_pars = dist.fit(self.extremes[_col])
            
            # Calculate GoF statistics 
            aic, bic = self.aic_bic(dist.pdf(self.extremes[_col], *fit_pars),
                                    len(fit_pars), len(self.extremes[_col]))
            
            # Add parameters and GoF parameters to the results object
            self.results[_col].add_distribution('Extreme', fit_pars, aic, bic)
            
        return fit_pars

            
    def plot_ev(self, **kwargs):
        """Plot the fitted GEV's.
        
        Returns
        -------
        matplotlib.figure.Figure
            Figure object.
        numpy.ndarray[Sequence[matplotlib.axes.Axes]]
            Numpy array with subplot axes.
        """
        
        #if 'Extreme' not in self.results[self._cols[0]].distributions_fitted:
        #    raise Exception("No extreme value distribution fitted yet!")

        fig, ax = plt.subplots(self._ncols, 1, figsize=(10, 10*self._ncols))

        for i, _col in enumerate(self._cols):
            x, f = self.ecdf(self.extremes[_col])    
            fit_pars = self.results[_col].distribution_parameters('Extreme')
            fit_cdf = self.scipy_dist('Extreme').cdf(x, *fit_pars)
            ax[i].plot(x, f, label="Empricial distribution", **kwargs)
            ax[i].plot(x, fit_cdf, label="Fitted extreme distribution",
                       **kwargs)
            ax[i].set_xlabel(self._col_labels["long"][i])
            ax[i].set_ylabel("F(X)")
            ax[i].legend()
            ax[i].grid()

            
        return fig, ax


    def QQ_plot(self, **kwargs):
        """Plot QQ-plots of the fitted distributions.

        Returns
        -------
        matplotlib.figure.Figure
            Figure object.
        numpy.ndarray[matplotlib.figure.SubFigure]
            Numpy array containing subfigures.
        """
        
        # Number of fitted distributions
        ndists = len(self.results[self._cols[0]].distributions_fitted())

        fig = plt.figure(figsize=(5*ndists, 5*self._ncols), constrained_layout=True)
        subfigs = fig.subfigures(self._ncols, 1)
        
        # Iterate over the selected variables
        for i, _col in enumerate(self._cols):

            # Add subplot for every fitted dist to subfigure
            ax = subfigs[i].subplots(1, ndists)
            subfigs[i].suptitle(self._col_labels["long"][i])

            # Calculate ECDF for the current variable
            x, f = self.ecdf(self.dataframe[_col])

            # Iterate over the fitted distributions
            for j, dist in enumerate(
                self.results[_col].distributions_fitted()):

                # Calculate the fitted cdf from parameters
                fit_pars = self.results[_col].distribution_parameters(dist)

                # Plot the quantiles
                sp_dist = self.scipy_dist(dist)
                st.probplot(self.dataframe[_col], fit_pars, sp_dist,
                            plot = ax[j])

                # Set labels
                ax[j].set_xlabel("Value")
                ax[j].set_ylabel("F(X)")

                # Set title of subplot to the distribution
                ax[j].set_title(dist)

                # Set grid
                ax[j].grid()
                
        return fig, subfigs

    
    # Bivariate fit
    
    
    def bivar_fit(self, vars: Optional[List[str]] = None,
                  N: Optional[int] = None) -> pd.DataFrame:
        """Fits a bivariate normal distribution, including covariance, on two
        variable from a dataframe, and draw samples from this fit.

        Parameters
        ----------
        vars : list[str, str], default None
            List containing the names of the columns to fit the distribution to.
        N : int, default None
            Number of samples to draw. If None, equal to the length of the 
            dataset.
            
        Returns
        -------
        pandas.core.DataFrame
            Dataframe containing the drawn samples.
        """

        # Checking the number of variables to compare
        if self._ncols == 1:
            raise ValueError('Only one variable selected, cannot perform \
                bivariate analysis.')
        elif self._ncols > 2 and vars is None:
            raise ValueError('More than two variables selected, please \
                provide the variables to compare through the vars argument.')
        elif vars is not None and len(vars) > 2:
            raise ValueError('Cannot compare more than two variables.')
        elif vars is None and self._ncols == 2:
            self._bivariate_vars = self._cols
        else:
            self._bivariate_vars = vars 
        
        data = self.dataframe[self._bivariate_vars]
        
        self._bivariate_labels = {"long":[], "short":[]}
        
        for i, col in enumerate(self._cols):
            if col in self._bivariate_vars:
                self._bivariate_labels["long"].append(
                    self._col_labels["long"][i])
                self._bivariate_labels["short"].append(
                    self._col_labels["short"][i])

        mean = np.mean(data, axis=0)
        cov = np.cov(data, rowvar=0)
    
        if not N:
            N = self.dataframe.shape[0]

        r_norm = st.multivariate_normal.rvs(mean, cov, N)
        self._bivar_r_norm = pd.DataFrame(r_norm, columns=vars)

        return self._bivar_r_norm


    def bivar_plot(self) -> None:
        """Creates several plots of the columns selected in `bivar_fit`.
        See 'Notes' section of Seaborn documentation page for `kdeplot()` for 
        guidance about smoothing with the Gaussian kernel (`bw_adjust` is a 
        multiplicative factor, increasing --> smoother).
        """

        if self._bivariate_vars == None:
            raise ValueError('No bivariate distribution fitted yet, please\
 use bivar_fit() first.')

        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        
        ax.plot(self.dataframe[self._bivariate_vars[0]], 
                self.dataframe[self._bivariate_vars[1]], 'o',
                markersize=4,
                markeredgecolor='k',
                markeredgewidth=0.25)

        ax.set_xlabel(self._bivariate_labels["long"][0])
        ax.set_ylabel(self._bivariate_labels["long"][1])

        ax.grid(True)

        h = sns.jointplot(data=self.dataframe, x=self._bivariate_vars[0],
                          y=self._bivariate_vars[1],
                          s=16,
                          joint_kws=dict(edgecolor='k', linewidth=0.25),
                          marginal_kws=dict(edgecolor='k', linewidth=1.0))
        h.set_axis_labels(xlabel=self._bivariate_labels["long"][0],
                          ylabel=self._bivariate_labels["long"][1])
        plt.gcf().tight_layout()

        g = sns.displot(data=self.dataframe, x=self._bivariate_vars[0],
                        y=self._bivariate_vars[1], kind='kde', bw_adjust=2.0)
        # g.set_axis_labels(xlabel=self._bivariate_labels["long"][0],
        #                   ylabel=self._bivariate_labels["long"][1])
        # does not work, using plt instead
        plt.xlabel(xlabel=self._bivariate_labels["long"][0])
        plt.ylabel(self._bivariate_labels["long"][1])
        plt.gcf().tight_layout()


    def cov_cor(self) -> Tuple[np.ndarray, float]:
        """Calculates the covariance and Pearson's correlation coefficient of
        the columns selected in `bivar_fit`.
        
        Returns
        -------
        numpy.ndarray
            Covariance matrix.
        float
            Pearson's r coefficient.
        """
        self._cov = np.cov(self.dataframe[self._bivariate_vars[0]],
                           self.dataframe[self._bivariate_vars[1]])
        self._cor, _ = st.pearsonr(self.dataframe[self._bivariate_vars[0]], 
                                   self.dataframe[self._bivariate_vars[1]])
        return self._cov, self._cor
    
    
    def plot_and_or_probabilities(self, quantiles: List[float],
                                  plot: Optional[bool] = False)\
                                      -> Tuple[float, float]:
        """Computes probabilities of one or both given variables in a dataframe
        exceeding a given quantile, and optionally creating a plot of this.

        Parameters
        ----------
        quantiles : list[float, float]
            Quantiles to calculate.
        plot : bool, default False
            Whether to plot the fitted distribution.
            
        Returns
        -------
        tuple[float, float]
            The probabilities of the AND and OR scenarios, respectively.
        """
        df_quantiles = [self.dataframe[self._bivariate_vars[0]]\
                            .quantile(quantiles[0]),
                        self.dataframe[self._bivariate_vars[1]]\
                            .quantile(quantiles[1])]
        and_sc = self.dataframe[(self.dataframe[self._bivariate_vars[0]]\
                                    >= df_quantiles[0]) &
                                (self.dataframe[self._bivariate_vars[1]]\
                                    >= df_quantiles[1])]
        or_sc = self.dataframe[(self.dataframe[self._bivariate_vars[0]]\
                                    >= df_quantiles[0]) |
                               (self.dataframe[self._bivariate_vars[1]]\
                                    >= df_quantiles[1])]
                           
        p_and = len(and_sc)/len(self.dataframe) 
        p_or = len(or_sc)/len(self.dataframe)

        if plot:
            fig, ax = plt.subplots(1, 2, sharex=True, sharey=True,
                                   figsize=(20, 5))
            ax[0].scatter(self.dataframe[self._bivariate_vars[0]],
                          self.dataframe[self._bivariate_vars[1]],
                          marker='o',
                          s=16,
                          c='cyan',
                          edgecolors='k',
                          linewidths=0.25)
            ax[0].scatter(and_sc[self._bivariate_vars[0]],
                          and_sc[self._bivariate_vars[1]],
                          marker='o',
                          s=16,
                          c='red',
                          edgecolors='k',
                          linewidths=0.25)
            ax[0].axvline(df_quantiles[0], color='k')
            ax[0].axhline(df_quantiles[1], color='k')
            ax[0].set_title(f'AND scenario, probability {p_and:.2f}')
            ax[0].set_xlabel(self._bivariate_labels["long"][0])
            ax[0].set_ylabel(self._bivariate_labels["long"][1])
            ax[0].grid()

            ax[1].scatter(self.dataframe[self._bivariate_vars[0]],
                          self.dataframe[self._bivariate_vars[1]],
                          marker='o',
                          s=16,
                          c='cyan',
                          edgecolors='k',
                          linewidths=0.25)
            ax[1].scatter(or_sc[self._bivariate_vars[0]],
                          or_sc[self._bivariate_vars[1]],
                          marker='o',
                          s=16,
                          c='red',
                          edgecolors='k',
                          linewidths=0.25)
            ax[1].axvline(df_quantiles[0], color='k')
            ax[1].axhline(df_quantiles[1], color='k')
            ax[1].set_title(f'OR scenario, probability {p_or:.2f}')
            ax[1].set_xlabel(self._bivariate_vars[0])
            ax[1].set_ylabel(self._bivariate_vars[1])
            ax[1].grid()
    
        return p_and, p_or


    # Static methods

    
    @staticmethod
    def find_datetime_col(dataframe):
        for col in dataframe:
            if pd.api.types.is_datetime64_any_dtype(dataframe[col]):
                return col
                
    
    @staticmethod
    def aic_bic(pdf, k, n):
        logL = np.sum(np.log10(pdf))
        aic = 2*k - 2*logL
        bic = k*np.log(n) - 2*logL
        return aic, bic
        
    
    @staticmethod
    def ecdf(var):
        """
        Compute the empirical cumulative distribution of a variable.

        Args:
            var (array): array containing the values of the variable.

        Returns:
            x (array): array containing the sorted values of the variable.
            f (array): array containing the probabilities.
        """
        x = np.sort(var)
        n = x.size
        f = np.arange(1, n+1)/n
        return x, f
    

    @staticmethod
    def scipy_dist(distribution):
        """
        Turn the name of a distribution into the scipy class
        """
        if distribution.lower()[:4] == "norm":
            dist = st.norm
        elif distribution.lower()[:3] == "exp":
            dist = st.expon
        elif distribution.lower()[:4] == "logn":
            dist = st.lognorm
        elif distribution.lower()[:4] == "logi":
            dist = st.logistic
        elif distribution.lower()[:4] == "extr":
            dist = st.genextreme
        else:
            raise Exception("Distribtution not found!")
        return dist
    

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
    
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st
import warnings

from IPython.display import display
from matplotlib.patches import Rectangle

from class_fitresults import FitResults

# TODO: add docstrings everywhere

class Dataset():

    distributions = ["Normal", "Exponential", "Lognormal", "Logistic"]
    
    # Constructors

    
    def __init__(self, dataframe, cols=None, col_labels=None):
        # Main dataframe
        self.dataframe = dataframe
        
        # Columns to use
        self._time_col = self.find_datetime_col(self.dataframe)
        self._cols = list(dataframe.drop(columns=self._time_col).columns)
        print(self._cols)

        # Descriptive columns labels
        self._col_labels = self._cols.copy()
        if col_labels:
            if len(col_labels) == len(self._cols):
                self._col_labels = col_labels
            else:
                warnings.warn("No. of col_labels does not match no. of cols,\
                    using defaults from dataframe", UserWarning)

        # Helpers
        self._ncols = len(self._cols)

        # To be computed
        self.extremes = None

        # Empty FitResults object for every variable
        self.results = {}
        for _col in self._cols:
            self.results[_col] = FitResults()

    @classmethod
    def import_from_filename(cls, filename, var_time, cols=None,
                             col_labels=None):
        dataframe = pd.read_csv(filename, parse_dates=[var_time])
        if cols:
            cols_used = [var_time] + cols
            dataframe = dataframe[cols_used]
        return cls(dataframe, cols, col_labels)


    @classmethod
    def import_from_surfdrive_path(cls, link, path, var_time, cols=None,
                                   col_labels=None):
        link += r'/download?path=%2F'
        path_lst = path.split('/')
        for s in path_lst[:-1]:
            link += s + "%2F"
        link = link[:-3]
        link += "&files=" + path_lst[-1]
        return cls.import_from_filename(link, var_time, cols, col_labels)


    @classmethod
    def import_from_surfdrive_file(cls, link, var_time, cols=None,
                                   col_labels=None):
        link += "/download"
        return cls.import_from_filename(link, var_time, cols, col_labels)


    # Cleaning dataset

    # TODO: see issue #5 on GitLab  
    def clean_dataset(self, z_score_threshold=5):
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


    def data_summary(self):
        display(self.dataframe[self._cols].describe())
        

    def time_plot(self, together=False, zoom=None, **kwargs):
        figsize = (10, 10) if together else (10, 5*len(self._cols))

        ax = self.dataframe.plot(x=self._time_col,
                          y=self._cols,
                          xlim=zoom,
                          subplots=not together,
                          sharex=True,
                          figsize=figsize,
                          marker='o',
                          ls='none',
                          grid=True,
                          ylabel=self._col_labels,
                          **kwargs)
        
        return plt.gcf(), ax


    def hist_plot(self, together=False, **kwargs):
        figsize = (10, 10) if together else (10, 5*len(self._cols))

        ax = self.dataframe.plot(y=self._cols,
                          kind='hist',
                          subplots=not together,
                          figsize=figsize,
                          title=self._col_labels if not together else None,
                          legend=together,
                          grid=True,
                          **kwargs)

        return plt.gcf(), ax


    # Univariate fit
        

    def plot_ecdf(self, **kwargs):

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
            subfigs[i].suptitle(self._col_labels[i])
            
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
        
        return plt.gcf(), ax


    def fit_distribution(self, distribution='all'):

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
        
        # Number of fitted distributions
        ndists = len(self.results[self._cols[0]].distributions_fitted())
        
        # Create figure and subfigures
        fig = plt.figure(figsize=(5*ndists, 5*self._ncols))
        subfigs = fig.subfigures(self._ncols, 1)
        
        # Iterate over the selected variables
        for i, _col in enumerate(self._cols):

            # Add subplot for every fitted dist to subfigure
            ax = subfigs[i].subplots(1, ndists)
            subfigs[i].suptitle(self._cols[i])

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
    def create_ev(self, period):
        self.extremes = self.dataframe.resample(period[0].upper(),
                                                on=self._time_col)\
                                                .max().reset_index(drop=True)
        return self.extremes

        
    def fit_ev(self):

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
            ax[i].set_xlabel("Value")
            ax[i].set_ylabel("F(X)")
            ax[i].legend()
            ax[i].grid()

            
        return fig, ax

    def QQ_plot(self, **kwargs):
        
        # Number of fitted distributions
        ndists = len(self.results[self._cols[0]].distributions_fitted())

        fig = plt.figure(figsize=(5*ndists, 5*self._ncols), constrained_layout=True)
        subfigs = fig.subfigures(self._ncols, 1)
        
        # Iterate over the selected variables
        for i, _col in enumerate(self._cols):

            # Add subplot for every fitted dist to subfigure
            ax = subfigs[i].subplots(1, ndists)
            subfigs[i].suptitle(self._cols[i])

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
                
                # Plot the ECDF 
                ax[j].plot([0, 1], [0, 1], '--', color='k')

                # Set labels
                ax[j].set_xlabel("Value")
                ax[j].set_ylabel("F(X)")

                # Set title of subplot to the distribution
                ax[j].set_title(dist)

                # Set grid
                ax[j].grid()
                
        return fig

    
    # Bivariate fit
    
    
    def bivar_fit(self, vars, plot=True, labels=None, N=None):
        data = self.dataframe[vars]

        mean = np.mean(data, axis=0)
        cov = np.cov(data, rowvar=0)
    
        if not N:
            N = self.dataframe.shape[0]

        r_norm = st.multivariate_normal.rvs(mean, cov, N)
        df_r_norm = pd.DataFrame(r_norm, columns=vars)

        if plot:
            self.bivar_plot(vars, labels=labels)

        return df_r_norm


    def bivar_plot(self, vars, labels=None):
        plt.figure(figsize=(6, 6))
        plt.plot(self.dataframe[vars[0]], self.dataframe[vars[1]], ".")
        plt.grid()
        if labels:
            plt.xlabel(labels[0])
            plt.ylabel(labels[1])
        plt.show()

        h = sns.jointplot(data=self.dataframe, x=vars[0], y=vars[1])
        if labels:
            h.set_axis_labels(labels[0], labels[1])
        plt.gcf().tight_layout()

        g = sns.displot(data=self.dataframe, x=vars[0], y=vars[1], kind='kde')
        if labels:
            g.set_axis_labels(labels[0], labels[1])
        plt.gcf().tight_layout()

        plt.show()


    def cov_cor(self, vars):
        cov = np.cov(self.dataframe[vars[0]], self.dataframe[vars[1]])
        corr, _ = st.pearsonr(self.dataframe[vars[0]], self.dataframe[vars[1]])
        return cov, corr
    
    
    def and_or_probabilities(self, vars, quantiles, plot=True, labels=None):
        df_quantiles = [self.dataframe[vars[0]].quantile(quantiles[0]),
                        self.dataframe[vars[1]].quantile(quantiles[1])]
        and_sc = self.dataframe[(self.dataframe[vars[0]] >= df_quantiles[0]) &
                                (self.dataframe[vars[1]] >= df_quantiles[1])]
        or_sc = self.dataframe[(self.dataframe[vars[0]] >= df_quantiles[0]) |
                               (self.dataframe[vars[1]] >= df_quantiles[1])]
                           
        p_and = len(and_sc)/len(self.dataframe) 
        p_or = len(or_sc)/len(self.dataframe)

        if plot:
            fig, ax = plt.subplots(1,
                                   2,
                                   sharex=True,
                                   sharey=True,
                                   figsize=(20, 5))
            ax[0].scatter(self.dataframe[vars[0]], self.dataframe[vars[1]])
            ax[0].scatter(and_sc[vars[0]], and_sc[vars[1]])
            ax[0].axvline(df_quantiles[0], color='k')
            ax[0].axhline(df_quantiles[1], color='k')
            ax[0].set_title(f'AND scenario, probability {p_and:.2f}')
            if labels:
                ax[0].set_xlabel(labels[0])
                ax[0].set_ylabel(labels[1])
            ax[0].grid()

            ax[1].scatter(self.dataframe[vars[0]], self.dataframe[vars[1]])
            ax[1].scatter(or_sc[vars[0]], or_sc[vars[1]])
            ax[1].axvline(df_quantiles[0], color='k')
            ax[1].axhline(df_quantiles[1], color='k')
            ax[1].set_title(f'OR scenario, probability {p_or:.2f}')
            if labels:
                ax[1].set_xlabel(labels[0])
                ax[1].set_ylabel(labels[1])
            ax[1].grid()
    

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
    def set_TUDstyle():
        TUcolor = {"cyan": "#00A6D6",
                   "darkgreen": "#009B77",
                   "purple": "#6F1D77",
                   "darkred": "#A50034",
                   "darkblue": "#0C2340",
                   "orange": "#EC6842",
                   "green": "#6CC24A",
                   "lightcyan": "#00B8C8",
                   "red": "#E03C31",
                   "pink": "#EF60A3",
                   "yellow": "#FFB81C",
                   "blue": "#0076C2"}
        plt.rcParams.update({'axes.prop_cycle': plt.cycler(color=TUcolor.values()),
                             'font.size': 16, "lines.linewidth": 4})
        return TUcolor
    
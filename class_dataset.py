import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st

import helpers

from IPython.display import display


class Dataset():

    # Constructors

    
    def __init__(self, dataframe, cols=None):
        self.dataframe = dataframe
        self._time_col = self.find_datetime_col(self.dataframe)
        self._cols = list(dataframe.drop(columns=self._time_col).columns)


    @classmethod
    def import_from_filename(cls, filename, var_time, cols=None):
        dataframe = pd.read_csv(filename, parse_dates=[var_time])
        if cols:
            cols_used = [var_time] + cols
            dataframe = dataframe[cols_used]
        return cls(dataframe, cols)


    @classmethod
    def import_from_surfdrive_path(cls, link, path, var_time, cols=None):
        link += r'/download?path=%2F'
        path_lst = path.split('/')
        for s in path_lst[:-1]:
            link += s + "%2F"
        link = link[:-3]
        link += "&files=" + path_lst[-1]
        return cls.import_from_filename(link, var_time, cols)


    @classmethod
    def import_from_surfdrive_file(cls, link, var_time, cols=None):
        link += "/download"
        return cls.import_from_filename(link, var_time, cols)


    # Cleaning dataset

        
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
                          **kwargs)
        
        return plt.gcf(), ax


    def hist_plot(self, together=False, **kwargs):
        figsize = (10, 10) if together else (10, 5*len(self._cols))

        ax = self.dataframe.plot(y=self._cols,
                          kind='hist',
                          subplots=not together,
                          figsize=figsize,
                          title=self._cols if not together else None,
                          legend=together,
                          grid=True,
                          **kwargs)

        return plt.gcf(), ax


    # Univariate fit
        

    def plot_ecdf(self, var, label=None, **kwargs):
        x, f = self.ecdf(self.dataframe[var])
        
        fig, ax = plt.subplots(1, 4, sharex=True, figsize=(24, 5))
        if label:
            plt.suptitle(f'ECDF of {label}', y=0.99)

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
        plt.show()
        
        return plt.gcf(), ax


    def fit_distribution(self, var, distribution, plot=True, label=None, 
                         **kwargs):
        x, f = self.ecdf(self.dataframe[var])

        dist = self.scipy_dist(distribution)

        fit_pars = dist.fit(self.dataframe[var])
        fit_cdf = dist.cdf(x, *fit_pars)

        if plot:
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.plot(x, f, label="Empirical Distribution", **kwargs)
            ax.plot(x, fit_cdf, label=f"Fitted {distribution} distribution",
                    **kwargs)
            ax.set_xlabel("Value")
            ax.set_ylabel("F(X)")
            if label:
                plt.suptitle(f'CDF of {label}')
            ax.legend()
            ax.grid()
            plt.show()

        return fit_pars, fit_cdf # Should this also return the plot?
        

    def plot_distributions(self, var, together=False, label=None, **kwargs):
        """Plots fitted distributions on a given variable in a single figure
        Currently uses Normal, Exponential, Lognormal, Logistic distributions
        Arguments:
            var (series): the variable
            separate (bool): whether to plot the distributions in seperate plots
            label (str): optional label to put in the title of the plot
        """
        x, f_emp = self.ecdf(self.dataframe[var])
        f_norm = self.fit_distribution(var, "Normal", plot=False)[1]
        f_exp = self.fit_distribution(var, "Exponential", plot=False)[1]
        f_lognorm = self.fit_distribution(var, "Lognormal", plot=False)[1]
        f_logit = self.fit_distribution(var, "Logistic", plot=False)[1]

        fitted = [f_norm, f_exp, f_lognorm, f_logit]
        dist_names = ["Normal", "Exponential", "Lognormal", "Logistic"]

        if not together:
            fig, ax = plt.subplots(1, 4, sharex=True, sharey=True, figsize=(20, 6))

            for i in range(len(fitted)):
                ax[i].plot(x, f_emp, label="Empirical Distribution", **kwargs)
                ax[i].plot(x, fitted[i],
                           label=f"Fitted {dist_names[i]} distribution", **kwargs)
                ax[i].set_xlabel("Value")
                ax[i].grid()
                ax[i].legend(bbox_to_anchor=(0.5, -0.1), loc='upper center')
            ax[0].set_ylabel("F(X)")
            # plt.tight_layout()

        else:
            fig, ax = plt.subplots(sharex=True, sharey=True, figsize=(20, 10))
            ax.plot(x, f_emp, label="Empirical Distribution", **kwargs)
            for i in range(len(fitted)):
                ax.plot(x, fitted[i], label=f"Fitted {dist_names[i]} distribution",
                        **kwargs)
            ax.set_xlabel("Value")
            ax.set_ylabel("F(X)")
            ax.legend(bbox_to_anchor=(0.5, -0.1), ncol=len(fitted) + 1, 
                      loc='upper center')
            plt.grid()

        if label:
            plt.suptitle(f"CDF's of {label}")

        plt.show()
    

    # Extreme Value Analysis
    

    def create_ev(self):
        pass


    def fit_ev(self):
        pass

    
    def QQ_plot(self):
        pass

    
    # Bivariate fit
    

    def bivar_plot(self):
        pass


    def cov_cor(self):
        pass
    
    
    def bivar_fit(self):
        pass
    
    
    def and_or_probabilities(self):
        pass
    

    # Static methods

    
    @staticmethod
    def find_datetime_col(dataframe):
        for col in dataframe:
            if pd.api.types.is_datetime64_any_dtype(dataframe[col]):
                return col
    
    
    @staticmethod
    def aic_bic():
        pass
        
    
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
        "Turn the name of a distribution into the scipy class"
        if distribution.lower()[:4] == "norm":
            dist = st.norm
        if distribution.lower()[:3] == "exp":
            dist = st.expon
        if distribution.lower()[:4] == "logn":
            dist = st.lognorm
        if distribution.lower()[:4] == "logi":
            dist = st.logistic
        if distribution.lower()[:4] == "extr":
            dist = st.genextreme
        else:
            Exception("Distribtution not found!")
        return dist
    
    @staticmethod
    def set_TUDstyle():
        TUcolor = {"cyan": "#00A6D6", "darkgreen": "#009B77", "purple": "#6F1D77",
                   "darkred": "#A50034", "darkblue": "#0C2340",
                   "orange": "#EC6842", "green": "#6CC24A",
                   "lightcyan": "#00B8C8", "red": "#E03C31", "pink": "#EF60A3",
                   "yellow": "#FFB81C", "blue": "#0076C2"}
        plt.rcParams.update({'axes.prop_cycle': plt.cycler(color=TUcolor.values()),
                             'font.size': 16, "lines.linewidth": 4})
        return TUcolor
    
    
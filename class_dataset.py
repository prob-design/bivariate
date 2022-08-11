import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st
import warnings

from IPython.display import display

# TODO: add docstrings everywhere

class Dataset():

    # Constructors

    
    def __init__(self, dataframe, cols=None, col_labels=None):
        self.dataframe = dataframe
        self._time_col = self.find_datetime_col(self.dataframe)
        self._cols = list(dataframe.drop(columns=self._time_col).columns)
        self._col_labels = self._cols.copy()
        if col_labels:
            if len(col_labels) == len(self._cols):
                self._col_labels = col_labels
            else:
                warnings.warn("No. of col_labels does not match no. of cols, using defaults from dataframe", UserWarning)
        self.extremes = None

        self.summary = dict.fromkeys(self._col_labels,
                                     dict(nans_removed=[],
                                          distributions_fitted=[],
                                          fit_parameters={},
                                          goodness_of_fit={}))


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
        

    def plot_ecdf(self, var, label=None, **kwargs):
        # TODO: use all variables, then use col_labels
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
        # TODO: use all variables (dictionary for dists?), then use col_labels
        x, f = self.ecdf(self.dataframe[var])

        dist = self.scipy_dist(distribution)

        fit_pars = dist.fit(self.dataframe[var])
        fit_cdf = dist.cdf(x, *fit_pars)

        aic, bic = self.aic_bic(pdf=dist.pdf(self.dataframe[var], *fit_pars),
                                k=len(fit_pars),
                                n=len(self.dataframe[var]))

        if plot:
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.plot(x, f, label="Empirical Distribution", **kwargs)
            ax.plot(x, fit_cdf, label=f"Fitted {distribution} distribution",
                    **kwargs)
            ax.set_xlabel("Value")
            ax.set_ylabel("F(X)")
            if label:
                plt.suptitle(f'CDF of {label}')
            gof_string = f'AIC: {aic:.3f}\nBIC: {bic:.3f}'
            ax.text(0.9, 0.1, gof_string, transform=ax.transAxes)
            ax.legend()
            ax.grid()
            plt.show()
            
        self.summary[var]['distributions_fitted'].append(distribution)
        self.summary[var]['fit_parameters'][distribution] = fit_pars

        return fit_pars, fit_cdf # Should this also return the plot?
        

    def plot_distributions(self, var, together=False, label=None, **kwargs):
        # TODO: use all variables, then use col_labels
        """
        Plots fitted distributions on a given variable in a single figure
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
    

    def create_ev(self, period):
        self.extremes = self.dataframe.resample(period[0].upper(),
                                                on=self._time_col)\
                                                .max().reset_index(drop=True)
        return self.extremes


    # TODO: this is a lot of duplicate code. We should find a way to make this 
    # and the other fit methods more general and compact.
    def fit_ev(self, var, plot=True, label=None, **kwargs):        

        if self.extremes is None:
            raise Exception("No extreme values computed yet!")
        
        x, f = self.ecdf(self.extremes[var])
        
        dist = self.scipy_dist('Extreme')
        
        fit_pars = dist.fit(self.extremes[var])
        fit_cdf = dist.cdf(x, *fit_pars)

        aic, bic = self.aic_bic(pdf=dist.pdf(self.extremes[var], *fit_pars),
                                k=len(fit_pars),
                                n=len(self.extremes[var]))

        if plot:
            fig, ax = plt.subplots(figsize=(10, 10))
            ax.plot(x, f, label="Empirical Distribution", **kwargs)
            ax.plot(x, fit_cdf, label=f"Fitted extreme distribution",
                    **kwargs)
            ax.set_xlabel("Value")
            ax.set_ylabel("F(X)")
            if label:
                plt.suptitle(f'CDF of {label}')
            gof_string = f'AIC: {aic:.3f}\nBIC: {bic:.3f}'
            ax.text(0.9, 0.05, gof_string, transform=ax.transAxes,
                    horizontalalignment='center', bbox=dict(facecolor='white'))
            ax.legend()
            ax.grid()
            plt.show()

        self.summary[var]['distributions_fitted'].append('Extreme')
        self.summary[var]['fit_parameters']['Extreme'] = fit_pars

        return fit_pars, fit_cdf

    
    def QQ_plot(self, var, distribution, **kwargs):
        pars, cdf = self.fit_distribution(var, distribution, plot=False)
        dist = self.scipy_dist(distribution)
        n = len(self.dataframe[var])
        var_sorted = np.sort(self.dataframe[var])

        ecdf_Q = np.linspace(1, n, n)/(n + 1)
        f_Q = dist.cdf(var_sorted, *pars)

        fig, ax = plt.subplots(figsize=(14, 8))
        ax.plot([0, 1], [0, 1], '--', color='k')
        ax.plot(ecdf_Q, f_Q, **kwargs)
        ax.set_xlabel('Empirical quantiles')
        ax.set_ylabel('Theoretical quantiles')
        ax.set_title(f"QQ-plot of fitted {distribution} distribution")
        ax.grid()
        plt.show()

    
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
    
    
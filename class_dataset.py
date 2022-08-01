import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st

import helpers

from IPython.display import display


class Dataset():
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


    def data_summary(self):
        display(self.dataframe[self._cols].describe())


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
        

    def plot_ecdf(self):
        pass


    def fit_distribution(self):
        pass
        

    def plot_distributions(self):
        pass
    

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
    def ecdf():
        pass
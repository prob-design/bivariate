import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import helpers

from IPython.display import display


class BivariateDataset():
    def __init__(self, df, cols=None):
        self.df = df
        self._time_col = helpers.find_datetime_col(self.df)
        self._cols = list(df.drop(columns=self._time_col).columns)


    @classmethod
    def from_filename(cls, filename, var_time, cols=None):
        df = pd.read_csv(filename, parse_dates=[var_time])
        if cols:
            cols_used = [var_time] + cols
            df = df[cols_used]
        return cls(df, cols)


    @classmethod
    def from_surfdrive_path(cls, link, path, var_time, cols=None):
        link += r'/download?path=%2F'
        path_lst = path.split('/')
        for s in path_lst[:-1]:
            link += s + "%2F"
        link = link[:-3]
        link += "&files=" + path_lst[-1]
        return cls.from_filename(link, var_time, cols)


    @classmethod
    def from_surfdrive_file(cls, link, var_time, cols=None):
        link += "/download"
        return cls.from_filename(link, var_time, cols)

        
    def clean_dataset(self, threshold=3):
        df = self.df.dropna().reset_index(drop=True)

        for col_name in self._cols:
            col = df[col_name]
            col_mean = col.mean()
            col_std = col.std()
            z = (col - col_mean) / col_std
            col_out_idx = col[np.abs(z) > threshold].index.values.tolist()
            df = df.drop(index=col_out_idx).reset_index(drop=True)
        
        self.df = df

        
    def time_plot(self, together=False, zoom=None, **kwargs):
        figsize = (10, 10) if together else (10, 5*len(self._cols))

        ax = self.df.plot(x=self._time_col,
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
        display(self.df[self._cols].describe())


    def hist_plot(self, together=False, **kwargs):
        figsize = (10, 10) if together else (10, 5*len(self._cols))

        ax = self.df.plot(y=self._cols,
                          kind='hist',
                          subplots=not together,
                          figsize=figsize,
                          title=self._cols if not together else None,
                          legend=together,
                          grid=True,
                          **kwargs)

        return plt.gcf(), ax
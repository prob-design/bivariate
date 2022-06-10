# -*- coding: utf-8 -*-
"""
Created on Mon May  2 12:50:00 2022

@author: ievdv
"""
import matplotlib.pyplot as plt
import pandas as pd

def data_exploration(var1, data, zoom, label1, label2, xlim1, xlim2, bins):
    label = [label1, label2]

    for i in range(0, len(var1)):
        fig, ax = plt.subplots(figsize=(10, 10), sharex=True)
        ax.plot('time', var1[i], data=data, linestyle='None',
                marker='o', markersize=1, color='royalblue')
        ax.set_title(var1[i])
        ax.grid(True)
        ax.xaxis.set_major_locator(plt.MaxNLocator(7))
        ax.set(xlabel='Date', ylabel=label[i])
        fig.autofmt_xdate()
        plt.show()

    for i in range(0, len(var1)):
        fig, ax = plt.subplots(figsize=(10, 10), sharex=True)
        ax.plot('time', var1[i], data=zoom, linestyle='None',
                marker='o', markersize=1, color='royalblue')
        ax.set_title(var1[i])
        ax.grid(True)
        ax.xaxis.set_major_locator(plt.MaxNLocator(7))
        ax.set(xlabel='Date', ylabel=label[i])
        fig.autofmt_xdate()
        plt.show()
    
    xlimit=[xlim1, xlim2]  # Define size of x-axis for each variable

    for i in range (0, len(var1)):
        fig, ax1= plt.subplots(figsize=(10, 10))
        ax1.hist(data[var1[i]].values, bins=bins, density=False)
        ax1.set_title(var1[i])
        ax1.grid(True)
        ax1.xaxis.set_major_locator(plt.MaxNLocator(10))
        ax1.set(xlabel=var1[i], ylabel='Frequency')
        plt.xlim(0, xlimit[i])
        plt.show()
        
    return
#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 12:07:48 2017

@author: iangarrett
"""

import numpy as np
from numpy.random import randn
import pandas as pd

from scipy import stats

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns


dataset = randn(25)

sns.rugplot(dataset)

plt.hist(dataset, alpha=0.3)

sns.rugplot(dataset)


x_min = dataset.min() - 2
x_max = dataset.max() + 2

x_axis = np.linspace(x_min,x_max,100)

#If Gaussian basis functions are used to approximate univariate data, 
#and the underlying density being estimated is Gaussian, 
#the optimal choice for h (that is, the bandwidth that minimises 
#the mean integrated squared error) is[20]

bandwidth = ((4*dataset.std()**5)/ (3*len(dataset))) ** 0.2

kernel_list = []

for data_point in dataset:
    #create a kernel for each point and append it to the kernel_list
    
    kernel = stats.norm(data_point,bandwidth).pdf(x_axis)
    kernel_list.append(kernel)
    
    #scale for plotting
    kernel = kernel/ kernel.max()
    kernel = kernel * 0.4
    
    plt.plot(x_axis, kernel, color = 'grey', alpha = .5)

plt.ylim(0,1)


sum_of_kde = np.sum(kernel_list,axis=0)

fig = plt.plot(x_axis, sum_of_kde, color='indianred')

sns.rugplot(dataset)

plt.yticks([])

plt.suptitle("sum of the basis functions")


#one Step

sns.kdeplot(dataset)

sns.rugplot(dataset, color='black')

for bw in np.arange(.5,2,.25):
    sns.kdeplot(dataset, bw=bw, lw=1.8, label=bw)
    
#Cumulative Density Function     
sns.kdeplot(dataset,cumulative = True)


#multivariate density estimation

mean = [0,0]

cov = [[1,0],[0,100]]

dataset2 = np.random.multivariate_normal(mean, cov, 1000)

dframe = pd.DataFrame(dataset2, columns=['x','y'])

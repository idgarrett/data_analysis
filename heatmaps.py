#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 13:35:57 2017

@author: iangarrett
"""

import numpy as np
from numpy.random import randn
import pandas as pd

from scipy import stats

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

flights_dframe = sns.load_dataset('flights')

flights_dframe = flights_dframe.pivot('month','year','passengers')

sns.heatmap(flights_dframe)

#w annotation

sns.heatmap(flights_dframe, annot=True, fmt='d')

f, (axis1,axis2) = plt.subplots(2,1)

yearly_flights = flights_dframe.sum()

years = pd.Series(yearly_flights.index.values)
years = pd.DataFrame(years)

flights = pd.Series(yearly_flights.values)
flights = pd.DataFrame(flights)

year_dframe = pd.concat((years,flights), axis=1)
year_dframe.columns = ['Years','Flights']

sns.barplot('Years', y='Flights', data=year_dframe)

sns.heatmap(flights_dframe, cmap='Blues', cbar_kws={'orientation': 'horizontal'})
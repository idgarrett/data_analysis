#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  1 17:05:24 2017

@author: iangarrett
"""

import numpy as np
import pandas as pd
from pandas import Series, DataFrame

from numpy.random import randn

from scipy import stats

import matplotlib as mpl
import matplotlib.pyplot as plt 
import seaborn as sns

dataset1 = randn(100)

plt.hist(dataset1)

dataset2 = randn(80)

plt.hist(dataset2, color='indianred', alpha=0.5)
plt.hist(dataset1, alpha=0.5)


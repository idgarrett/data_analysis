#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 11:59:39 2017

@author: iangarrett
"""

import pandas as pd
import numpy as np

import cufflinks as cf

from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

init_notebook_mode(connected=True)

cf.go_offline()

df = pd.DataFrame(np.random.randn(100,4), columns = 'A B C D'.split())

df2 = pd.DataFrame({'Category':['A','B','C'], 'Values':[32,43,50]})

df.plot()
df.iplot(kind='scatter', x='A', y='B', mode = 'markers')

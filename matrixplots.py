#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 11:51:02 2017

@author: iangarrett
"""

import seaborn as sns

tips = sns.load_dataset('tips')
flights = sns.load_dataset('flights')


#correlation matrix
tc = tips.corr()
#Heatmap
sns.heatmap(tc, annot=True, cmap='coolwarm')

fp = flights.pivot_table(index='month', columns='year', values='passengers')

sns.heatmap(fp, cmap='magma', linecolor='white', linewidth=1)

#clustermap

sns.clustermap(fp, cmap='coolwarm', standard_scale=1)


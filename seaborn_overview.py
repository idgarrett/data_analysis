#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 13 10:49:52 2017

@author: iangarrett
"""

import seaborn as sns

tips = sns.load_dataset('tips')

tips.head()

sns.distplot(tips['total_bill'], kde=False, bins = 30)

sns.jointplot(x='total_bill', y='tip', data=tips, kind='hex')

#add regression line
sns.jointplot(x='total_bill', y='tip', data=tips, kind='regression')

#compare all numerical data types
sns.pairplot(data=tips)

#add hue - categorical variable

sns.pairplot(tips, hue='sex')

#rugplots

sns.rugplot(tips['total_bill'])

sns.barplot(x='sex',y='total_bill', data=tips)

sns.countplot(x='sex', data=tips)

#box and violin plots - shows the distribution of data

sns.boxplot(x='day', y='total_bill', data=tips)

sns.boxplot(x='day', y='total_bill', data=tips, hue='smoker')

#violin plot

sns.violinplot(x='day', y='total_bill', data=tips)

sns.violinplot(x='day', y='total_bill', data=tips, hue='sex', split=True)

#swarmplot

sns.swarmplot(x='day', y='total_bill', data=tips)






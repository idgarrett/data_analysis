#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 10:29:15 2017

@author: iangarrett
"""

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('/Users/iangarrett/Python-Data-Science-and-Machine-Learning-Bootcamp/Machine Learning Sections/Linear-Regression/USA_Housing.csv')

sns.heatmap(df.corr(),annot=True)

X = df[['Avg. Area Income', 'Avg. Area House Age',
       'Avg. Area Number of Rooms', 'Avg. Area Number of Bedrooms',
       'Area Population']]

y = df['Price']

from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)

#random_state = number of random splits in data set

from sklearn.linear_model import LinearRegression

lm = LinearRegression()

lm.fit(X_train, y_train)

print(lm.intercept_)

cdf = pd.DataFrame(lm.coef_,X.columns, columns=['Coeff'])

#Predictions

predictions = lm.predict(X_test)

plt.scatter(y_test, predictions)

sns.distplot((y_test-predictions))
#should see normally distributed data 

#accuracy
from sklearn import metrics

metrics.mean_absolute_error(y_test, predictions)
metrics.mean_squared_error(y_test, predictions)
#root mean squared error
np.sqrt(metrics.mean_squared_error(y_test, predictions))













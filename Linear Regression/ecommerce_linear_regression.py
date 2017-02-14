#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 10:59:25 2017

@author: iangarrett
"""

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

customers = pd.read_csv('/Users/iangarrett/Python-Data-Science-and-Machine-Learning-Bootcamp/Machine Learning Sections/Linear-Regression/Ecommerce Customers')

#Exploratory Analysis

sns.jointplot(data=customers, x='Time on Website', y='Yearly Amount Spent')

sns.jointplot(data=customers, x='Time on App', y='Yearly Amount Spent')

sns.jointplot(data=customers, x='Time on App', y='Length of Membership', kind='hex')

sns.pairplot(customers)

sns.lmplot(x='Length of Membership', y='Yearly Amount Spent', data = customers)

customers.columns

y = customers['Yearly Amount Spent']

X = customers[['Avg. Session Length', 'Time on App','Time on Website', 'Length of Membership']]

from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

from sklearn.linear_model import LinearRegression

lm = LinearRegression()

lm.fit(X_train, y_train)

#print out coefficients of the model

lm.coef_

#predicting from test data


predictions = lm.predict(X_test)

plt.scatter(y_test, predictions)

#should show normal distribution of errors
sns.distplot((y_test-predictions))


# calculate these accuracy metrics
from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

coeffecients = pd.DataFrame(lm.coef_,X.columns)
coeffecients.columns = ['Coeffecient']
coeffecients

"""
                     Coeffecient
Avg. Session Length     25.981550
Time on App             38.590159
Time on Website          0.190405
Length of Membership    61.279097


Interpreting the coefficients:
Holding all other features fixed, a 1 unit increase in Avg. Session Length is associated with an increase of 25.98 total dollars spent.
Holding all other features fixed, a 1 unit increase in Time on App is associated with an increase of 38.59 total dollars spent.
Holding all other features fixed, a 1 unit increase in Time on Website is associated with an increase of 0.19 total dollars spent.
Holding all other features fixed, a 1 unit increase in Length of Membership is associated with an increase of 61.27 total dollars spent.

"""
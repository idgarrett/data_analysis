#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 12:20:17 2017

@author: iangarrett
"""

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

ad_data = pd.read_csv('/Users/iangarrett/Python-Data-Science-and-Machine-Learning-Bootcamp/Machine Learning Sections/Logistic-Regression/advertising.csv')


from sklearn.cross_validation import train_test_split

X= ad_data[['Daily Time Spent on Site', 'Age', 'Area Income', 'Daily Internet Usage']]
y= ad_data['Clicked on Ad']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression()

logmodel.fit(X_train, y_train)

predictions = logmodel.predict(X_test)

from sklearn.metrics import classification_report

print(classification_report(y_test, predictions))

from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, predictions)
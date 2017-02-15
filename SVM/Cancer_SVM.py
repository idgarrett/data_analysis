#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 11:00:39 2017

@author: iangarrett
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()

df_feat = pd.DataFrame(cancer['data'], columns=cancer['feature_names'])

cancer['target_names']

from sklearn.cross_validation import train_test_split

X = df_feat
y = cancer['target']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)

from sklearn.svm import SVC

model = SVC()

model.fit(X_train, y_train)
#Predict and Evaluation
predictions = model.predict(X_test)


from sklearn.metrics import classification_report,confusion_matrix

print(classification_report(y_test,predictions))
print('/n')
print(confusion_matrix(y_test,predictions))

from sklearn.grid_search import GridSearchCV

param_grid = {'C': [.1,1,10,100,1000], 'gamma':[1,.1,.01,.001,.0001]}

grid = GridSearchCV(SVC(), param_grid, verbose=3)

grid.fit(X_train, y_train)

grid.best_params_
grid.best_estimator_

grid_predictions = grid.predict(X_test)

print(classification_report(y_test,grid_predictions))
print('/n')
print(confusion_matrix(y_test,grid_predictions))








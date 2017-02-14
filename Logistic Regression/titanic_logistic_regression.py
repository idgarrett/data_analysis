#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 11:38:56 2017

@author: iangarrett
"""


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

train = pd.read_csv('/Users/iangarrett/Python-Data-Science-and-Machine-Learning-Bootcamp/Machine Learning Sections/Logistic-Regression/titanic_train.csv')

sns.heatmap(train.isnull(),yticklabels=False, cbar=False, cmap='viridis')

#looks like we can fill in age data, but not cabin data

sns.set_style('whitegrid')

sns.countplot(x='Survived', data=train)

#look at survivor by sex
sns.countplot(x='Survived', data=train, hue='Sex', palette='RdBu_r')

#look at survivor by class
sns.countplot(x='Survived', data=train, hue='Pclass', palette='RdBu_r')

#look at age distribution
sns.distplot(train['Age'].dropna(),kde=False, bins=30)

sns.countplot(x='SibSp', data=train)

train['Fare'].hist(bins=40)

cf.go_offline()

train['Fare'].iplot(kind='hist', bins=40)

import cufflinks as cf
import plotly as py
fig = train['Fare'].iplot(kind='scatter', asFigure=True)
py.offline.plot(fig)


#Cleaning data

sns.boxplot(x='Pclass', y='Age', data=train)

def impute_age(cols):
    Age = cols[0]
    Pclass = cols[1]
    
    if pd.isnull(Age):
        
        if Pclass == 1:
            return 37
        elif Pclass==2:
            return 29
        else:
            return 24
    else:
        return Age
    
train['Age']= train[['Age','Pclass']].apply(impute_age, axis=1)

#check heatmap

sns.heatmap(train.isnull(),yticklabels=False, cbar=False, cmap='viridis')

train.drop('Cabin', inplace=True, axis=1)

train.dropna(inplace=True)

sns.heatmap(train.isnull(),yticklabels=False, cbar=False, cmap='viridis')


#Create dummy variables & drop first to avoid colinearity 

sex = pd.get_dummies(train['Sex'], drop_first=True)

embark = pd.get_dummies(train['Embarked'], drop_first=True)

train = pd.concat([train, sex, embark], axis=1)

#drop unnecessary columns
train.drop(['Sex', 'Embarked', 'Name', 'Ticket'], axis=1, inplace=True)
train.drop(['PassengerId'], axis=1, inplace=True)


#create model, test splits
X = train.drop('Survived', axis=1)
y = train['Survived']

from sklearn.cross_validation import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression()

logmodel.fit(X_train, y_train)

predictions = logmodel.predict(X_test)

from sklearn.metrics import classification_report

print(classification_report(y_test, predictions))

from sklearn.metrics import confusion_matrix

confusion_matrix(y_test, predictions)







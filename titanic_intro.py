#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 14:19:17 2017

@author: iangarrett
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import Series, DataFrame

import os

os.chdir('/Users/iangarrett/data_analysis')


titanic_df = pd.read_csv('train.csv')

#who were the passengers on the titanic

sns.factorplot(x='Sex',data=titanic_df, hue='Pclass', kind='count')

sns.factorplot(x='Pclass',data=titanic_df, hue='Sex', kind='count')

def male_female_child(passenger):
    age,sex = passenger
    
    if age < 16:
        return 'child'
    else:
        return sex
        
        
titanic_df['person'] = titanic_df[['Age','Sex']].apply(male_female_child, axis=1)

sns.factorplot(x='Pclass',data=titanic_df, hue='person', kind='count')

titanic_df['Age'].mean()

titanic_df['person'].value_counts()


fig = sns.FacetGrid(titanic_df, hue='Sex', aspect=4)
fig.map(sns.kdeplot,'Age', shade=True)

oldest = titanic_df['Age'].max()

fig.set(xlim=(0,oldest))
fig.add_legend()


fig = sns.FacetGrid(titanic_df, hue='person', aspect=4)
fig.map(sns.kdeplot,'Age', shade=True)

oldest = titanic_df['Age'].max()

fig.set(xlim=(0,oldest))
fig.add_legend()

fig = sns.FacetGrid(titanic_df, hue='Pclass', aspect=4)
fig.map(sns.kdeplot,'Age', shade=True)

oldest = titanic_df['Age'].max()

fig.set(xlim=(0,oldest))
fig.add_legend()

deck = titanic_df['Cabin'].dropna()

levels = []

for level in deck:
    levels.append(level[0])

cabin_df = DataFrame(levels)
cabin_df.columns=['Cabin']

sns.factorplot(x='Cabin',data=cabin_df, kind='count', palette='winter_d')

#drop the t cabin

cabin_df = cabin_df[cabin_df.Cabin != 'T']

#who was alone and who was with family?

titanic_df['Alone'] = titanic_df.SibSp + titanic_df.Parch

#replace values in 'Alone' 
titanic_df['Alone'].loc[titanic_df['Alone']>0] = 'With Family'
titanic_df['Alone'].loc[titanic_df['Alone'] == 0] = 'Alone'

sns.factorplot(x='Alone',data=titanic_df, kind='count', palette='winter_d')

titanic_df['Survivor'] = titanic_df.Survived.map({0: 'no', 1: 'yes'})

sns.factorplot(x='Survivor',data=titanic_df, kind='count', palette='winter_d')

sns.factorplot(x='Pclass',hue='Survived',data=titanic_df, kind='count', palette='winter_d')

#survival reates by class
sns.factorplot('Pclass','Survived',data=titanic_df)

#check for gender
sns.factorplot('Pclass','Survived', hue='person', data=titanic_df)


sns.lmplot('Age', 'Survived', hue='Pclass', data=titanic_df)

generations = [10,20,40,60,80]

sns.lmplot('Cabin', 'Survived', hue='Pclass', data=titanic_df, x_bins=generations)

#survival reates by deck
sns.factorplot('Cabin','Survived',data=cabin_df)


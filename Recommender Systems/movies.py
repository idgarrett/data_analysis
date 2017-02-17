#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 12:34:49 2017

@author: iangarrett
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

column_names = ['user_id', 'item_id', 'rating', 'timestamp']
df = pd.read_csv('/Users/iangarrett/Python-Data-Science-and-Machine-Learning-Bootcamp/Machine Learning Sections/Recommender-Systems/u.data', sep='\t', names=column_names)

movie_titles = pd.read_csv('/Users/iangarrett/Python-Data-Science-and-Machine-Learning-Bootcamp/Machine Learning Sections/Recommender-Systems/Movie_Id_Titles')

df = pd.merge(df,movie_titles,on='item_id')

df.groupby('title')['rating'].mean().sort_values(ascending=False).head()

ratings = pd.DataFrame(df.groupby('title')['rating'].mean())

ratings['num of ratings'] = pd.DataFrame(df.groupby('title')['rating'].count())

sns.jointplot(x='rating', y='num of ratings', data=ratings)

#Build recommender system

moviemat = df.pivot_table(index='user_id',columns='title',values='rating')
moviemat.head()


starwars_user_ratings = moviemat['Star Wars (1977)']
liarliar_user_ratings = moviemat['Liar Liar (1997)']
starwars_user_ratings.head()


similar_to_starwars = moviemat.corrwith(starwars_user_ratings)
similar_to_liarliar = moviemat.corrwith(liarliar_user_ratings)

corr_starwars = pd.DataFrame(similar_to_starwars,columns=['Correlation'])
corr_starwars.dropna(inplace=True)
corr_starwars.head()

corr_starwars.sort_values('Correlation',ascending=False).head(10)

corr_starwars = corr_starwars.join(ratings['num of ratings'])
corr_starwars.head()

corr_starwars[corr_starwars['num of ratings']>100].sort_values('Correlation',ascending=False).head()


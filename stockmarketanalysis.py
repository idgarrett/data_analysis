#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  3 15:39:50 2017

@author: iangarrett
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
from pandas import Series, DataFrame
from pandas_datareader import data, wb
import os

from __future__ import division

from datetime import datetime

tech_list = ['AAPL','GOOG','MSFT','AMZN']

end = datetime.now()

start = datetime(end.year-1,end.month, end.day)

for stock in tech_list:
    globals()[stock] = data.DataReader(stock,'yahoo',start,end)
    
AAPL['Adj Close'].plot(legend=True)

AAPL['Volume'].plot(legend=True)

ma_day = [10,20,50]

for ma in ma_day:
    column_name = "MA for %s days" %(str(ma))
    
    AAPL[column_name] = AAPL['Adj Close'].rolling(window=ma,center=False).mean()
    
AAPL[['Adj Close', 'MA for 10 days', 'MA for 20 days', 'MA for 50 days']].plot(subplots=False)

AAPL['Daily Return'] = AAPL['Adj Close'].pct_change()

AAPL['Daily Return'].plot(subplots=False)

sns.distplot(AAPL['Daily Return'].dropna(), bins=100, color='purple')

closing_df = data.DataReader(tech_list, 'yahoo', start, end)['Adj Close']

tech_rets = closing_df.pct_change()

sns.jointplot('GOOG', 'GOOG', tech_rets, kind = 'scatter', color='seagreen')

sns.jointplot('GOOG', 'MSFT', tech_rets, kind = 'scatter', color='seagreen')

returns_fig = sns.PairGrid(tech_rets.dropna())

returns_fig.map_upper(plt.scatter, color='purple')

returns_fig.map_lower(sns.kdeplot,cmap='cool_d')

returns_fig.map_diag(plt.hist, bins=30)

# Let's start by defining a new DataFrame as a clenaed version of the oriignal tech_rets DataFrame
rets = tech_rets.dropna()

area = np.pi*20

plt.scatter(rets.mean(), rets.std(),alpha = 0.5,s =area)

# Set the x and y limits of the plot (optional, remove this if you don't see anything in your plot)
plt.ylim([0.01,0.025])
plt.xlim([-0.003,0.004])

#Set the plot axis titles
plt.xlabel('Expected returns')
plt.ylabel('Risk')

# Label the scatter plots, for more info on how this is done, chekc out the link below
# http://matplotlib.org/users/annotations_guide.html
for label, x, y in zip(rets.columns, rets.mean(), rets.std()):
    plt.annotate(
        label, 
        xy = (x, y), xytext = (50, 50),
        textcoords = 'offset points', ha = 'right', va = 'bottom',
        arrowprops = dict(arrowstyle = '-', connectionstyle = 'arc3,rad=-0.3'))

days = 365

dt = 1/days

mu = rets.mean()['GOOG']

sigma = rets.std()['GOOG']

def stock_monte_carlo(start_price, days, mu, sigma):
    
    price = np.zeros(days)
    price[0] = start_price

    shock = np.zeros(days)
    drift = np.zeros(days)
    
    for x in xrange(1,days):
        shock[x] = np.random.normal(loc=mu*dt, scale=sigma*np.sqrt(dt))
        
        drift[x] = mu * dt

        price[x] = price[x-1] + (price[x-1] * (drift[x] + shock[x]))
        
    return price
    
start_price = 770.21

for run in xrange(100):
    plt.plot(stock_monte_carlo(start_price,days,mu,sigma))   
    
plt.xlabel('days')
plt.ylabel('Price')
plt.title('Monte Carlo Fnalysis For Google')

runs = 10000

simulations = np.zeros(runs)

for run in xrange(runs):
    simulations[run] = stock_monte_carlo(start_price,days,mu, sigma)[days-1]


#99% of results should be within our histogram
q = np.percentile(simulations,1)

plt.hist(simulations,bins=200)

#starting price
plt.figtext(.6,.8, s="start price: $%.2f" %start_price)

#mean ending price
plt.figtext(.6,.7, "Mean final price: $%.2f" %simulations.mean())

#Variance of the price (within 99% confidence interval)
plt.figtext(.6,.6, "VaR(.99): $%.2f" %(start_price - q))

#display 1% quantile
plt.figtext(.15,.6, "q(.99): $%.2f" % q )

#plot a line at the 1% quantile result
plt.axvline(x=q, linewidth=4, color='r')

plt.title(u"Final price distribution for Google Stock after %days" % days, weight='bold');


#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 11:30:34 2017

@author: iangarrett
"""

from math import log
import numpy as np
import matplotlib.pyplot as plt

plt.style.use('bmh')

#Set up runtime comparisons
n = np.linspace(1,10,1000)
labels = ['Constant', 'Logarithmic', 'Linear', 'Log Linear', 'Quadratic', 'Cubic', 'Exponential']

big_o = [np.ones(n.shape), np.log(n), n, n*np.log(n), n**2, n**3, 2**n]

#plot Setup

plt.figure(figsize=(6,5))
plt.ylim(0,50)

for i in range(len(big_o)):
    plt.plot(n,big_o[i], label = labels[i])
    
plt.legend(loc=0)
plt.ylabel('Relative Runtime')
plt.xlabel('n')

#O(n) Constant
def func_constant(values):
    print values[0]
    
lst = [1,2,3]

func_constant(lst)

#O(n) Linear

def func_lin(lst):
    
    for val in lst:
        print val
        
func_lin(lst)

def func_quad(lst):
    '''
    Prints pairs for every item in list.
    '''
    for item_1 in lst:
        for item_2 in lst:
            print item_1,item_2
            
lst = [0, 1, 2, 3]

func_quad(lst)

#For Data Structures

def method1():
    l = []
    for n in xrange(10000):
        l = l + [n]

def method2():
    l = []
    for n in xrange(10000):
        l.append(n)

def method3():
    l = [n for n in xrange(10000)]

def method4():
    l = range(10000) # Python 3: list(range(10000))
    
%timeit method1()
%timeit method2()
%timeit method3()
%timeit method4()

   
    






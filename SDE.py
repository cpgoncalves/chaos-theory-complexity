# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 10:01:08 2024

Stochastic Runge-Kutta Method based on:

Roberts A.J. (2012) Modify the Improved Euler scheme to
integrate stochastic differential equations https://arxiv.org/abs/1210.0933

@author: 
    Carlos Pedro Gonçalves
@institution:
    Lusophone University 
@department:
    Department of Management on Civil Aviation and Airports 
    School of Economic Sciences and Organizations
"""

import numpy as np


def K1(h, # time step
      a, # drift
      b, # volatility
      S, # S term
      dW # noise term
       ):
    
    return h * a + b*(dW - np.sqrt(h) * S)


def K2(h, # time step
      a, # drift
      b, # volatility
      S, # S term
      dW # noise term
       ):
    
    return h * a + b*(dW + np.sqrt(h) * S)



def dX(k1, k2):
    
    return 0.5 * (k1 + k2)



    
    


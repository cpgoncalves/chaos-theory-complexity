"""
Example 2: Logistic map's bifurcation and LLE plots using, in the last case,
the Jacobian for numerical estimation.

The current example is part of classroom material for the Mathematics I
curricular unit from Lusophone University's Bachelor 
in Aeronautical Management at the Lisbon University Center's 
School of Economic Sciences and Organizations

Copyright (c) January 2026 Carlos Pedro Gonçalves

"""

import DynMap
import numpy as np
from matplotlib import pyplot as plt

# Parameter r, initial condition, number of transient steps and number of
# simulation steps
params=params=np.arange(2.5,4,0.001)
x0=np.pi/10
transient=1000
T=1000+transient


# User-defined map (in this case it is the logistic map)
def f(x, r):
    return r * x * (1-x)


# Jacobian
def jac(x, r):
    return r * (1 - 2*x)

# Dynamical Map object to be used for simulation and main analytics
x = DynMap.dmap(x0,f)

# Bifurcation plot
x.bifurcation(param_name='r', 
              param_values=params, 
              steps_trans=transient, steps_plot=100, coord=0, x0=x0,s=0.005)


# LLEs estimation and plotting
LLEs=[]

for r in params:
    LLEs.append(x.lyapunov_jac(steps=8000, jac=jac, discard=5000, r=r))
    
    
plt.plot(params,LLEs,c='k',lw=0.5)
plt.plot(params,np.zeros(len(LLEs)),'r--',)
plt.xlabel('r')
plt.ylabel('Largest Lyapunov Exponent')
plt.show()
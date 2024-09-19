# -*- coding: utf-8 -*-
"""
3D  Brownian Motion Simulation

@author: 
    Carlos Pedro Gonçalves
@institution:
    Lusophone University 
@department:
    Department of Management on Civil Aviation and Airports 
    School of Economic Sciences and Organizations

"""

import SDE
import numpy as np
from scipy.stats import norm, bernoulli
from matplotlib import pyplot as plt

h=0.001
n_steps=10000
X=np.array([0,0,0])
a=np.array([0,0,0])
b=1
increments=[]
trajectory=[X]


for t in range(0,n_steps):
    dW = np.sqrt(h)*norm.rvs(size=3)
    S=2*bernoulli.rvs(p=0.5,size=3)-1
    k1 = SDE.K1(h=h,a=a,b=b,S=S,dW=dW)
    k2 = SDE.K2(h=h,a=a,b=b,S=S,dW=dW)
    dX=SDE.dX(k1=k1, k2=k2)        
    X=X+dX
    increments.append(list(dX))
    trajectory.append(list(X))

trajectory=np.array(trajectory)


ax = plt.figure().add_subplot(projection='3d')
ax.plot(trajectory[:,0],trajectory[:,1],trajectory[:,2], lw=0.5)
plt.show()


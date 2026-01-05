"""
Example 4: Simulation of deterministic and stochastic Henon map, for a
stochastic chaos example simulation, allowing the user to control and see the
impact of noise on the map's dynamics in phase space.

The current simulator is part of classroom material for the Mathematics I
curricular unit from Lusophone University's Bachelor 
in Aeronautical Management at the Lisbon University Center's 
School of Economic Sciences and Organizations.

Copyright (c) January 2026 Carlos Pedro Gonçalves

"""

import DynMap
import numpy as np
from matplotlib import pyplot as plt

transient=1000
T=10000+transient
x0=[0.012234,0.233]
transient=500
a=1.4
b=0.3
sigma=0.05


def henon(x, a, b,sigma):
    u=np.random.rand(1, 2).tolist()[0]
    return np.array([(1-sigma)*(1 - a*x[0]**2 + x[1]) + sigma*u[0], (1-sigma)*(b*x[0]) + sigma*u[1]])


x = DynMap.dmap(x0,henon)

trajectory_det=x.run(steps=T, table=False, a=a,b=b,sigma=0)
trajectory_stoch=x.run(steps=T, table=False, a=a,b=b,sigma=sigma)

trajectory_det=trajectory_det[transient:]
trajectory_stoch=trajectory_stoch[transient:]

plt.scatter(trajectory_det[:,0],trajectory_det[:,1],s=0.01)
plt.title("Deterministic Henon Map a= "+str(a)+" b="+str(b))
plt.show()

plt.scatter(trajectory_stoch[:,0],trajectory_stoch[:,1],s=0.01)
plt.title("Stochastic Henon Map a= "+str(a)+" b="+str(b)+" sigma="+str(sigma))
plt.show()


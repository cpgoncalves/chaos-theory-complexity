"""
Example 5: Simulator of CML with quadratic map.

The current simulator is part of classroom material for the Mathematics I
curricular unit from Lusophone University's Bachelor 
in Aeronautical Management at the Lisbon University Center's 
School of Economic Sciences and Organizations.

Copyright (c) January 2026 Carlos Pedro Gonçalves
"""

import CML
import numpy as np

r=1.45
threshold=0.05
coupling=0.2
N=50
transient=10000
T=200
x0=2*np.random.rand(N)-1
lw=0.3


def quadratic_map(x, r=r):
    return 1 - r*x**2


cml = CML.CoupledMapLattice(state=x0,
                            size=N,f=quadratic_map)


history, desync_history = cml.run(steps=T+transient,coupling=coupling,update=True)

history=history[transient:]
desync_history=desync_history[transient:]


CML.do_plot(history,desync_history,threshold=threshold,cmap='plasma',orientation='H',
            lw=lw,return_h=False)


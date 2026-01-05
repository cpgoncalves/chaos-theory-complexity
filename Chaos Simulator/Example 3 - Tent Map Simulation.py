"""
Example 3: Tent map simulator with calculation of iterations
producing the time series plot in both continuous line and scatterplot format,
power spectrum for the simulated series, the colored recurrence plot and 
perform a recurrence stats calculation from the distance matrix.

The current simulator is part of classroom material for the Mathematics I
curricular unit from Lusophone University's Bachelor 
in Aeronautical Management at the Lisbon University Center's 
School of Economic Sciences and Organizations

Copyright (c) January 2026 Carlos Pedro Gonçalves

"""

import DynMap
import numpy as np

# Parameter r, initial condition, number of transient steps and number of
# simulation steps
r=1.9
x0=np.pi/10
transient=1000
T=1000+transient

# Title to be used in plotting
title="Tent Map r= "+str(r)


# User-defined map (in this case it is the tent map)
def f(x, r):
    return np.where(x < 0.5, r * x, r * (1 - x))

# Dynamical Map object to be used for simulation and main analytics
x = DynMap.dmap(x0,f)


# Run the map for T steps and extract the simulated trajectory
trajectory=x.run(steps=T,table=False,r=r)

# Plot the simulated trajectory
x.plot_trajectory1D(trajectory[transient:].T[0],title=title,lw=1.0,s=1.0)

# Plot the power spectrum
x.pspectrum(series=trajectory[transient:].T[0], title=title)

# Plot the colored recurrence plot, if the user supplies a radius the
# black and white recurrence plot is supplied instead
S=x.recurrence_matrix(series=trajectory[transient:].T[0], # series of values
                    radius=None, # radius used for plotting
                    one_dim=True, # series is one dimensional or n-dimensional
                    plot=True # if the recurrence plot is to be obtained
                      )

# Get the recurrence analysis stats from the distance matrix
x.recurrence_analysis(S,radius=0.01)
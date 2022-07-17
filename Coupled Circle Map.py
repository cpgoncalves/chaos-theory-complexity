# -*- coding: utf-8 -*-
"""
Created on Sun Jul 17 12:30:11 2022


@author: Carlos Pedro Gonçalves
@title: Associate Professor
@institution: Lusophone University of Humanities and Technologies
@department:Department of Management on Civil Aviation and Airports.
@school: School of Economic Sciences and Organizations 

Coupled map lattice simulator exemplifying a coupled circle map, with local
coupling and periodic boundary conditions.

The main Python functions for the map are given, along with the space and time
plot using matshow functionality.

Two examples are given of Wolfram's class 4 (edge of chaos) and class 3.
    
The example is provided in the context of the Research Project: 
Chaos Theory and Complexity Sciences

https://sites.google.com/view/chaos-complexity

The project also contains classroom materials used by the author
for his Mathematics classes in at the Bachelor in Aeronautical Management 
(https://www.ulusofona.pt/en/undergraduate/aeronautical-management) 
at Lusophone University of Humanities and Technologies.


The present simulator is a part of the author's talk at the 2022 
International Online Conference:
    
- Management and Governance In Times of Crisis

https://sites.google.com/view/management-governance-crisis 

To cite the software use the following reference:

Gonçalves CP. Python Coupled Circle Map. Chaos Theory and Complexity
Sciences Research Project. July, 2022.


This work is licensed under the BSD 2-Clause License:

Copyright (c) 2022, Carlos Pedro Gonçalves

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


"""

# =============================================================================
# Main Functions
# =============================================================================

import numpy as np
from matplotlib import pyplot as plt

# circle map function
def circle_map(x,K,Omega):
    
    return np.mod(x + (K / (2*np.pi)) * np.sin(2*np.pi*x) + Omega,1)

# coupled circle map for an N element network
def coupled_map(x,N,K,Omega,epsilon):    
    
    
    # apply the circle map
    F = []
    for i in range(0,N):
        F.append(circle_map(x[i],K,Omega))
    
    # use the local nearest neighbors coupling with periodic 
    # boundary conditions
    
    
    x[0] = (1 - epsilon)*F[0]+epsilon*0.5*(F[-1]+F[1])
    x[N-1] = (1 - epsilon)*F[0]+epsilon*0.5*(F[-2]+F[0])
    
    
    for i in range(1,N-1):
        x[i] = (1 - epsilon)*F[i]+epsilon*0.5*(F[i-1]+F[i+1])
    
    # return the updated network state
    return x


# iterate map function
def iterate_map(N,K,Omega,epsilon,transient=100,T=200):
    
    # select a random initial condition for the N elements
    x = np.random.rand(N)
    
    # iterate the map
    trajectory = []
    
    for t in range(0,transient+T):
        x = coupled_map(x,N,K,Omega,epsilon)
        # keep the results after the transient
        if t >= transient:
            trajectory.append(list(x))
    
    return np.array(trajectory)

# =============================================================================
# Simulation Examples
# =============================================================================

# Two examples of Wolfram's class 4 (edge of chaos) and class 3
# presented at the online conference: 
#
# Management and Governance In Times of Crisis
# https://sites.google.com/view/management-governance-crisis 

# Class 4
plt.matshow(iterate_map(N=100,K=1.21,Omega=0.3,epsilon=0.35,
                        transient=100,T=200),
            cmap=plt.cm.inferno_r)
plt.show()


# Class 3
plt.matshow(iterate_map(N=100,K=1.21,Omega=0.25,epsilon=0.2,
                        transient=100,T=200),
            cmap=plt.cm.inferno_r)
plt.show()

"""
Dynamical Maps Simulator

Nonlinear Maps Simulator was developed as a classroom material for the author's 
Mathematics I' classes of the Aeronautical Management Bachelor lectured by
the author at Lusophone University:
https://www.ulusofona.pt/en/lisboa/bachelor/aeronautical-management

The software is part of the international R&D project: 
https://sites.google.com/view/chaos-complexity/

The software code was partially developed using ChatGPT in a human-machine
collaboration for software development.

The software is based on Object-Oriented Programming (OOP).

The class dmap contains the main attributes and methods for a dynamical map
simulation.

Copyright (c) January 2026 Carlos Pedro Gonçalves

"""

from matplotlib import pyplot as plt
from numpy.matlib import repmat
import numpy as np
import pandas as pd

from IPython.display import display

class dmap:
    """ Dynamical map (1D or nD) main class
    
    The attributes are the initial state and the map's function f
    
    The methods are:
        - step: single iteration step used as auxiliary method in the run method.
        - run: runs the simulation returning the trajectory and printing
               an iteration table if asked for.
        - run_mult: runs the map for multiple initial conditions and plots the
                    results it also returns the trajectories if asked for.
        - plot_trajectory1D: plots the trajectory for a 1D map in both a normal
                             time series plot and as a scatterplot.
        - bifurcation: calculates and plots the bifurcation diagram for 
                       different values of a parameter.
        - _num_jacobian: auxiliary method for numerical estimation of the 
                         Jacobian.
        - lyapunov_jac: calculation of largest Lyapunov exponent from either
                        a theoretical or a numerical Jacobian.
        - recurrence_matrix: used for recurrence plots and recurrence analysis.
        - recurrence_analysis: recurrence analysis metrics producing the
          Shannon entropy for the recurrence probability, the average recurrence
          strength and the probability of 100% recurrence given that the 
          diagonal has recurrence points.
    
    NOTE: See the example files for the full implementation of these methods
          for different nonlinear maps.
    
    """

    def __init__(self, state, f):
        # The state attribute can be 1D or nD.
        self.state = np.array(state, dtype=float)
        # The map is user defined as in the example files.
        self.f = f
        
        
    # ===== Simulation Methods =====
    
    # Single iteration step method used to iterate the map.
    def step(self, **kwargs):
        self.state = np.array(self.f(self.state, **kwargs), dtype=float)
    
    # Run the map for a number of iterations equal to the number of steps
    # if table is set to True a table for the iterations is produced using
    # a Pandas' DataFrame, otherwise the trajectory is returned.
    def run(self, steps, table=False, **kwargs):
        trajectory = np.zeros((steps, np.size(self.state)))
        trajectory[0] = self.state
        for t in range(1, steps):
            self.step(**kwargs)
            trajectory[t] = self.state
        if table:
            df = pd.DataFrame({
                'Iteration': np.arange(steps),
                'Trajectory': trajectory if trajectory.shape[1] > 1 else trajectory.flatten()
            }).set_index('Iteration')
            with pd.option_context('display.max_rows', None, 'display.max_columns', None):
                display(df)
        return trajectory
    
    # Run the map for different initial conditions and plot them, this is used
    # to illustrate the sensitive dependence to initial conditions for a 1D
    # map, the number of steps, list of initial conditions, title of the plot
    # line width are provided by the user.
    def run_mult(self, steps, x_list, title, lw, return_trajectories=False,
                 legend=True, **kwargs):
        original_state = np.copy(self.state)
        trajectories = []
        for x0 in x_list:
            self.state = np.array(x0, dtype=float)
            trajectories.append(self.run(steps, **kwargs))
        self.state = original_state
        trajectories = np.array(trajectories)
        plt.plot(trajectories.transpose(1, 0, 2)[:, :, 0] if trajectories.ndim == 3 else trajectories.T,
                 label=[str(x0) for x0 in x_list], lw=lw)
        plt.title(title)
        if legend==True:
            plt.legend(title="x(0)", bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.show()
        if return_trajectories == True:
            return trajectories.transpose(1, 0, 2) if trajectories.ndim == 3 else trajectories.T
    
    
    # Plot a 1D trajectory, used specifically for one-dimensional maps.
    def plot_trajectory1D(self, trajectory, title, lw, s):
        iteration = np.arange(len(trajectory))
        trajectory = np.array(trajectory).flatten()

        plt.plot(iteration, trajectory, c='k', lw=lw)
        plt.xlabel('Iteration')
        plt.ylabel('x(t)')
        plt.title(title)
        plt.show()

        plt.scatter(iteration, trajectory, c='k', s=s)
        plt.xlabel('Iteration')
        plt.ylabel('x(t)')
        plt.title(title)
        plt.show()

        plt.scatter(trajectory[:-1], trajectory[1:], s=s, c='k')
        plt.xlabel('x(t-1)')
        plt.ylabel('x(t)')
        plt.title(title)
        plt.show()
    
    # Bifurcation diagram is produced for one of the coordinates with varying
    # of a parameter, the user must supply the parameter name, 
    # the parameter values, the transient steps to be dropped, and the
    # number of steps to be supplied in the plot
    def bifurcation(self, param_name, param_values, steps_trans, steps_plot, 
                    coord=0, 
                    x0=None,s=0.1, **kwargs):
        original_state = np.copy(self.state)
        xs, ps = [], []

        for p in param_values:
            self.state = np.copy(x0) if x0 is not None else np.copy(original_state)
            self.run(steps_trans, **{param_name: p}, **kwargs)
            traj = self.run(steps_plot, **{param_name: p}, **kwargs)
            traj = np.atleast_2d(traj)
            xs.extend(traj[:, coord])
            ps.extend([p] * steps_plot)

        self.state = original_state
        plt.scatter(ps, xs, s=s, c='k')
        plt.xlabel(param_name)
        plt.title("Bifurcation Diagram")
        plt.show()

    # Estimate the largest Lyapunov exponent using a numerically estimated
    # Jacobian used in the lyapunov_jac method
    def _num_jacobian(self, x, eps_scale=1e-6, **kwargs):
        x = np.atleast_1d(np.array(x, dtype=float))
        d = x.size
        J = np.zeros((d, d))
        h = eps_scale * (1.0 + np.abs(x))
        for i in range(d):
            ei = np.zeros(d)
            ei[i] = h[i]
            f1 = np.atleast_1d(np.array(self.f(x + ei, **kwargs), dtype=float))
            f2 = np.atleast_1d(np.array(self.f(x - ei, **kwargs), dtype=float))
            J[:, i] = (f1 - f2) / (2.0 * h[i])
        return J

    # Estimate the largest Lyapunov exponent from a supplied Jacobian or
    # alternatively from a numerically estimated Jacobian
    def lyapunov_jac(self, steps, jac=None, discard=100, eps_scale=1e-6, **kwargs):
        original_state = np.copy(self.state)
        self.run(max(0, int(discard)), **kwargs)
        x = np.copy(self.state)
        d = np.size(x)
        v = np.random.normal(size=d)
        v /= np.linalg.norm(v)

        sum_log = 0.0
        for _ in range(int(steps)):
            J = jac(x, **kwargs) if jac is not None else self._num_jacobian(x, eps_scale=eps_scale, **kwargs)
            J = np.array([[J]] if np.isscalar(J) else J, dtype=float)
            v = J @ v
            norm_v = np.linalg.norm(v)
            if norm_v == 0 or not np.isfinite(norm_v):
                v = np.random.normal(size=d)
                v /= np.linalg.norm(v)
            else:
                sum_log += np.log(norm_v)
                v /= norm_v
            x = np.array(self.f(x, **kwargs), dtype=float)

        self.state = original_state
        return sum_log / steps

    # Perform spectral analysis for a dynamical variable
    def pspectrum(self,series, title):
    
        # Power spectrum
        ps = np.abs(np.fft.fft(series))**2
        freqs = np.fft.fftfreq(len(series), 1)  # dt = 1
        

        # Only keep positive frequencies
        mask = freqs > 0
        freqs = freqs[mask]
        ps = ps[mask]

        # Plot
        plt.loglog(freqs, ps, 'k', lw=0.8)
        plt.xlabel("Frequency")
        plt.ylabel("Power")
        plt.title(title)
        plt.show()
        
    
    # Get the recurrence matrix for recurrence analysis
    def recurrence_matrix(self,
                          series, # series of values
                          radius=None, # radius used for plotting
                          one_dim=True, # series is one dimensional or n-dimensional
                          plot=True # if the recurrence plot is to be obtained
                          ):
               
        # if the series of observations is a 1D signal...
        if one_dim==True:
            
            # get the series length
            N = len(series)
            
            # convert series to a column vector
            series=np.matrix(series).T
            
            # initialize the recurrence matrix to an N by N zeros matrix
            S=np.matrix(np.zeros(shape=(N,N)))
            
            # for each observation:
            for i in range(0,N):
                # the i-th column of S is comprised of the distance
                # between the i-th observation and each other observation
                # in the series using the absolute value of the difference
                S[:,i]=abs(repmat(series[i],1,N).T-series)
       
        # otherwise: the series of observations is a sequence embedded in 
        # d-dimensional phase space
        else:
            
            # convert into matrix format
            series=np.matrix(series)
            
            # get the number of lines (number of observations)
            N = np.size(series,0)
            # get the number of columns (number of dimensions)
            dim = np.size(series,1)
            
            
            # initialize the distance matrix as above
            S=np.matrix(np.zeros(shape=(N,N)))
            
            # for each dimension:
            for d in range(0,dim):
                # extract the corresponding column
                series_d=series[:,d]
                # the i-th column of S is comprised of the sum of the squares
                # of the difference between each observation in the series and
                # the corresponding d-th column
                for i in range(0,N):
                    S[:,i]+=np.power(repmat(series_d[i][0],1,N).T-series[:,d],2)
            # take the square root to get the Euclidean distances
            S = np.sqrt(S)
        
        
        # if thee radius is provided get the recurrence matrix
        if radius != None:
            B = S <= radius
            B = 1*B
            B=np.asarray(B)
        
        
        if plot == True:
            # if the radius is provided plot the black and white recurrence plot
            if radius != None:
                plt.matshow(B, cmap=plt.cm.gray_r)
                plt.show()
               
                
            # if the radius is not provided...
            else:
                # plot the colored recurrence plot
                plt.matshow(S,cmap=plt.cm.inferno)
                plt.colorbar()
                plt.show()
        
        # return the distance matrix
        if radius != None:
            return B
        elif radius == None:
            return S


    # Perform recurrence analysis function applied to a distance matrix
    def recurrence_analysis(self,
                            S, # distance matrix
                            radius, # radius to test
                            printout_lines=False, # printout lines with 100% recurrence
                            printout_results=True, # return the diagonals with 100% recurrence
                            return_stats=False # return stats
                            ):
           
        # get the recurrence matrix for the given radius
        B = S <= radius
        B = 1*B
        
        # initialize the number of diagonals
        num_diagonals = 0
        
        # list for diagonals with 100% recurrence
        diagonals_full = []
        
        # initialize the number of diagonals with recurrence
        recurrence_total = 0
            
        # initialize the recurrence strength (will have the sum of fills)
        recurrence_strength = 0
        
        # while the size of B is different from 1...
        while B.size != 1:
            # add one more to the number of diagonals
            num_diagonals += 1
            # delete the last row and column to get the 
            # new main diagonal (the first parallel to the previous main diagonal)
            B = np.delete(B,(-1),1)
            B = np.delete(B,(0),0)
            # get the trace to get how many recurrence points are in the line
            recurrence = np.trace(B)
            # get the size of the diagonal
            size_diagonal = np.size(B,0)
            
            # if there are recurrence points in the line...
            if recurrence > 0:
                # add to the total number of lines with recurrence
                recurrence_total += 1
                # if all points in the diagonal are recurrence points...
                if recurrence == size_diagonal:
                    # add the diagonal rank to the diagonals list
                    diagonals_full.append(num_diagonals)
                # get the recurrence proportion
                # (proportion of the line that has recurrence points)
                recurrence_strength += recurrence/size_diagonal
        
        
        # get the number of diagonals with 100% recurrence
        num_recurrence_100 = len(diagonals_full)
        
        # if there are lines with 100% recurrence, calculate the crosstabs for
        # the distances between diagonals with 100% recurrence
        if num_recurrence_100 != 0:
            if printout_results == True:
                print("Number recurrence 100:", num_recurrence_100)
            distances=[] # distances between the lines with 100% recurrence
            if printout_lines == True:
                print("\nLines with 100% recurrence:\n")    
            for i in range(0,num_recurrence_100):
                if printout_lines == True:
                    # print each line with 100% recurrence
                    print("\nLine", diagonals_full[i])
                if i > 0:
                    distances.append(diagonals_full[i]-diagonals_full[i-1])
            if len(distances) > 0:
                distances = pd.DataFrame(distances,columns=["Distances"])
                if printout_results == True:
                    print(pd.crosstab(index=distances["Distances"],columns="count"))
        
        ARS = recurrence_strength / recurrence_total
        P100 = num_recurrence_100/recurrence_total
        
        # print other stats
        if printout_results == True:
            print("\nNumber of lines with 100% recurrence:", num_recurrence_100)
            print("\nNumber of lines with recurrence:", recurrence_total)
            print("\nTotal number of diagonals:", num_diagonals)
            P_rec = recurrence_total/num_diagonals
            print("\nProportion of diagonals with recurrence:",P_rec)
            # if there are recurrence points print the additional stats
            if recurrence_total > 0:
                if P_rec != 0 and P_rec != 1:
                    Shannon = -P_rec * np.log2(P_rec) - (1 - P_rec) * np.log2(1 - P_rec)
                else:
                    Shannon = 0
                print("\nShannon entropy:", Shannon)
                print("\nAverage recurrence strength:", ARS)
                print("\nP[100% recurrence|recurrence]", P100)
        
        # if one wishes to extract the statistics
        if return_stats:
            return ARS, P100

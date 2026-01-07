"""
Coupled Map Lattice's Simulator was developed as a classroom material 
for the author's  Mathematics I' classes of the Aeronautical Management 
Bachelor lectured by the author at Lusophone University:
https://www.ulusofona.pt/en/lisboa/bachelor/aeronautical-management

The software is part of the international R&D project: 
https://sites.google.com/view/chaos-complexity/

The software code was partially developed using ChatGPT in a human-machine
collaboration for software development.

The software is based on Object-Oriented Programming (OOP) along with
functional programming.

The class dmap contains the main attributes and methods for a dynamical map
simulation.

Copyright (c) January 2026 Carlos Pedro Gonçalves



"""
import numpy as np
import matplotlib.pyplot as plt

def rand_init(N):
    """Uniform random initialization in the unit interval for the initial 
    lattice state and setup of the variable that will store the full 
    trajectory"""
    return np.random.rand(N)


class CoupledMapLattice:
    """ 
    Coupled 1D Map Lattice class and methods.
         
    The attributes are the state the lattice size and the map.
    
    The methods are:
        - step: used for the iteration of the lattice using the left and right
                neighbors' connection and periodic conditions at the boundary
                (ring lattice).
        - run: runs the lattice with either the update with the map f
               before applying the local coupling or using the previous
               state or alternatively using the previous mean value 
               of the left and right neighbors for the local
               nearest neighbors' coupling formally:
               
               a) Rule with update: 
               x(i,t+1) = (1-coupling)f(x(i,t))+coupling*0.5*(f(x(i-1,t))+f(x(i+1,t)))
               
               b) Rule without update:
               x(i,t+1) = (1-coupling)f(x(i,t))+coupling*0.5*(x(i-1,t)+x(i+1,t))
    
    
    """
    
    def __init__(self, state, size, f):
        self.state = state # initial state
        self.N = size # lattize size
        self.f = f # map
        
    # Single iteration step method
    def step(self,coupling,update=True,**kwargs):
        f_vals = self.f(self.state,**kwargs) # apply the map
        new_state = np.zeros_like(self.state) # initialize the new state
        if update==True:
            # if the update rule is true get the result from applying the map
            # for the left and right neighbors...
            left = np.roll(f_vals,shift=1)
            right = np.roll(f_vals,shift=-1)
        else:
            # ...otherwise use the previous state of the left and right 
            # neighbors
            left = np.roll(self.state,shift=1)
            right = np.roll(self.state,shift=-1)
        
        # update the new state applying the local nearest neighbors' coupling
        new_state = (1-coupling)*f_vals + coupling * 0.5 * (left + right)
        self.state = new_state
       
        
    # Lattice map simulation
    def run(self, steps,coupling,update=True,**kwargs):
        history = np.zeros((steps, self.N)) # initialize the history as a numpy array
        
        history[0]=self.state # first line is the initial state
        
        for t in range(1,steps):
            # iterate the lattice
            self.step(coupling=coupling,update=update,**kwargs)
            # update the history
            history[t] = self.state
            
        # Extract the desynchronization history for analysis 
        # using the absolute deviation between each site's state
        # and the mean value of the two nearest neighbors, the higher the value
        # the greater the local deviation (more desynchronized)
        left_value=np.roll(history,shift=1,axis=1)
        right_value=np.roll(history,shift=-1,axis=1)
        mean_value=(left_value+right_value)/2
        desync_history = np.abs(history-mean_value)
        
        return history, desync_history # return the history and desync_history



"""

Functions used for analysis:
    
    a) pspectrum: power spectrum for mean field analysis.
    b)  do_plot: used for plotting the CML's dynamics


"""


def pspectrum(series,title,y_min,y_max,time_step = 1,lw=0.1):
    
    # perform the spectrum analysis
    ps = np.abs(np.fft.fft(series))**2
    freqs = np.fft.fftfreq(len(series), time_step)
    idx = np.argsort(freqs)
    
    # plot the power spectrum in a log-log scale
    fig, ax = plt.subplots(1)
    plt.title(title)
    ax.loglog(freqs[idx],ps[idx],c='k',lw=lw)
    ax.set_xlabel('Frequency')
    ax.set_ylabel('Power')
    plt.show()


def do_plot(history,desync_history,threshold=0.01,
            cmap='plasma',cmap2='gray',orientation='V',lw=0.4,s=0.1,return_h=False):
    
    """ 
    Do the plot for the history, the mean field plot and first returns plot
    along with the power spectrum.
    
    """
    
    
    h=np.mean(history,axis=1) # calculate the mean field
    desync=np.mean(desync_history,axis=1) # calculate the mean desynchronization
    
    if orientation == 'V': # plots show lattice sites vertically and time horizontally
        
        # Plot the lattice state evolution using the chosen colormap
        plt.imshow(history.T, aspect='auto', cmap=cmap, origin='lower')
        plt.ylabel('Lattice Site')
        plt.xlabel('Time Step')
        plt.title('Coupled Map Lattice Evolution')
        plt.colorbar(label='State Value')
        plt.show()
        
        # Plot the desynchronization evolution using the chosen colormap
        plt.imshow(desync_history.T, aspect='auto', cmap=cmap, origin='lower')
        plt.ylabel('Lattice Site')
        plt.xlabel('Time Step')
        plt.title('Desynchronization Evolution')
        plt.colorbar(label='Sync Level')
        plt.show()
        
        # Plot a synchronization event defined in terms of low desynchronization value
        plt.imshow((desync_history.T < threshold)*1.0, aspect='auto', cmap=cmap2, 
                   origin='lower')
        plt.ylabel('Lattice Site')
        plt.xlabel('Time Step')
        plt.title('Synchronization Event for Threshold: '+str(threshold))
        plt.colorbar(label='Sync Level')
        plt.show()
        
        
        # Plot a half-point graph 1 if it is higher than the middle point
        # 0 otherwise
        max_history=np.max(history)
        min_history=np.min(history)
        middle=(max_history+min_history)/2
        bin_history=(history > middle)*1.0
        
        plt.imshow(bin_history.T, aspect='auto', cmap=cmap2, 
                   origin='lower')
        plt.ylabel('Lattice Site')
        plt.xlabel('Time Step')
        plt.title('Half-Point: '+str(middle))
        plt.colorbar(label='x(i,t) > '+str(middle))
        plt.show()
        
        
    elif orientation == 'H': # plots show lattice sites horizontally and time vertically
        
        plt.imshow(history, aspect='auto', cmap=cmap)
        plt.xlabel('Lattice Site')
        plt.ylabel('Time Step')
        plt.title('Coupled Map Lattice Evolution')
        plt.colorbar(label='State Value')
        plt.show()
        
        
        plt.imshow(desync_history, aspect='auto', cmap=cmap)
        plt.xlabel('Lattice Site')
        plt.ylabel('Time Step')
        plt.title('Desynchronization Evolution')
        plt.colorbar(label='Desynchronization Level')
        plt.show()
        
        
        plt.imshow((desync_history < threshold)*1.0, aspect='auto', cmap='gray')
        plt.xlabel('Lattice Site')
        plt.ylabel('Time Step')
        plt.title('Synchronization Evolution for Threshold: '+str(threshold))
        plt.colorbar(label='Synchronization event = 1')
        plt.show()
        
                
        max_history=np.max(history)
        min_history=np.min(history)
        middle=(max_history+min_history)/2
        bin_history=(history > middle)*1.0
        
        plt.imshow(bin_history, aspect='auto', cmap='gray')
        plt.xlabel('Lattice Site')
        plt.ylabel('Time Step')
        plt.title('Half-Point: '+str(middle))
        plt.colorbar(label='x(i,t) > '+str(middle))
        plt.show()
    
    # Plot the mean field dynamics
    plt.plot(h,c='k',lw=lw)
    plt.title("Mean Field Dynamics")
    plt.show()
    
    plt.scatter(np.arange(len(h)),h,c='k',s=s)
    plt.title("Mean Field Dynamics")
    plt.show()
    
    # Plot the first returns for the mean field
    plt.scatter(h[:-1],h[1:],c='k',s=s)
    plt.xlabel("h(t-1)")
    plt.ylabel("h(t)")
    plt.show()
    
    # Plot the desynchronization dynamics
    plt.plot(desync,c='k',lw=lw)
    plt.title("Mean Desynchronization Dynamics")
    plt.show()
    
    
    # Plot the mean field versus the desynchronization
    plt.scatter(h,desync,c='k',s=s)
    plt.title("Mean Field vs Mean Desynchronization")
    plt.xlabel("h(t)")
    plt.ylabel("Mean Desynchronization(t)")
    plt.show()
    
    # Plot the power spectral density for the mean field
    pspectrum(series=h,title="Mean Field Power Spectrum", 
              y_min=min(h),
              y_max=max(h), 
              time_step = 1,
              lw=lw)
    
    if return_h:
        return h # return the mean field if asked for

    
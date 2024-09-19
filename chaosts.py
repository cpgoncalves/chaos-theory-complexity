# -*- coding: utf-8 -*-
"""

Carlos Pedro Goncalves, September, 2024
Lusophone University

Chaos time series analyzer with incorporated smart topological data analysis
methods including embedding dimension selection based on best performance
of adaptive AI predictor and k-nearest neighbors' graph analysis

The software is provided for both research into chaos theory in time series
analysis and used as classroom material for the author's Statistics and
Mathematics' classes of the Aeronautical Management Bachelor lectured by
the author at Lusophone University:
https://www.ulusofona.pt/en/lisboa/bachelor/aeronautical-management

The software is part of the international R&D project: 
https://sites.google.com/view/chaos-complexity/

Version 1:
Copyright (c) October 2023 Carlos Pedro Gonçalves
Version 2 (present version):
Copyright (c) September 2024 Carlos Pedro Gonçalves

This work is licensed under the BSD 3-Clause License

@author: cpdsg
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from numpy.matlib import repmat
from scipy import stats
from scipy.stats import linregress
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import explained_variance_score as e_variance
from sklearn.metrics import r2_score
from sklearn.neighbors import kneighbors_graph
from sklearn.feature_selection import mutual_info_regression
import networkx as nx
from scipy.signal import detrend
from statsmodels.tsa.tsatools import detrend as poly_detrend
import nolds # requires installation of nolds

# =============================================================================
# Signal Analysis Methods
# =============================================================================


# Detrending a series using either linear of polynomial
def ts_detrend(signal,order):
    if order == 1:
        detrended_signal = detrend(signal)
    else:
        detrended_signal = poly_detrend(signal,order)
    
    trend = signal - detrended_signal
    
    print("\nDetrending Results:")
    print("R2:", r2_score(signal, trend))
    
    
    return detrended_signal, trend



# Spectral analysis function with option for fitting the slope and 
# extracting the Hurst exponent in the case of a power law spectrum
# used for Self-Organized Criticality markers' identification (SOC)
def pspectrum(series,title,time_step = 1,
              low_cut=None,high_cut=None,fit_slope=False):
    
    # perform the spectrum analysis
    ps = np.abs(np.fft.fft(series))**2
    freqs = np.fft.fftfreq(len(series), time_step)
    idx = np.argsort(freqs)
    
    # plot the power spectrum in a log-log scale
    fig, ax = plt.subplots(1)
    plt.title(title)
    ax.loglog(freqs[idx],ps[idx],c='k',lw=0.60)
    ax.set_xlabel('Frequency')
    ax.set_ylabel('Power')
    plt.show()
    
    
    
    # if slope fit for power law is used estimate the slope and produce
    # the regression results plus the Hurst exponent
    if fit_slope == True:
        
        n=int(1+len(freqs[idx])/2)
        f_log=np.log(freqs[idx][n:])
        ps_log=np.log(ps[idx][n:])
        
                
        slope, intercept, r, p, se = linregress(f_log[low_cut:high_cut],
                                                ps_log[low_cut:high_cut])
        
        print("\nSlope:", slope)
        print("\nIntercept:", intercept)
        print("\nR2", r ** 2)
        print("\np-value:", p)
        beta = -slope
        print("\nHurst:", (beta - 1)/2)
        fig2, ax2 = plt.subplots(1)
        plt.plot(f_log,ps_log,c='k',lw=0.60)
        plt.plot(f_log, intercept + slope * f_log, 'k--', lw=0.8)
        ax2.set_xlabel('Frequency')
        ax2.set_ylabel('Power')
        plt.show()

# Calculation of log-log histogram for the case of SOC markers' identification
def loglog_hist(series,bins):
    
    # build the histogram using the bins and get the relative frequencies
    hist, bin_edges = np.histogram(series,bins=bins)
    hist = np.asarray(hist)/np.sum(hist)
    
    # calculate the class centers as the middle points calculated from
    # the bin edges
    bin_edges = np.asarray(bin_edges)
    bin_centers = (bin_edges[1:]+bin_edges[:-1])/2
    
    # build a Pandas dataframe with the class centers
    d = {'c': bin_centers, 'f': hist}
    df=pd.DataFrame(data=d)

    # exclude the classes with zero frequency
    df=df[df.f!=0]
    
    # return the logarithms for the class centers and frequencies
    log_c = np.log(df.c)
    log_f = np.log(df.f)
    return log_c, log_f


# Plot the Histogram in log-log scale used for SOC markers' identification
def plot_hist(log_c,log_f,fit_data=False,start=None,stop=None):
    
    # if a linear fit is to be applied (in case of a power law fit)
    # the fitting is done on a chosen range where power law decay has been
    # identified
    if fit_data == True:
        
        slope, intercept, r, p, se = linregress(log_c[start:stop],log_f[start:stop])
        
        print("\nSlope:", slope)
        print("\nIntercept:", intercept)
        print("\nR2", r ** 2)
        print("\np-value:", p)
        
    # get the scatterplot for the logarithm of the class centers versus
    # the logarithm of the frequencies
    plt.scatter(log_c,log_f,c='k',s=2)
    # if the power law fit was applied plot the fitted line
    if fit_data == True:
        plt.plot(log_c, intercept + slope * log_c, 'k--', lw=0.2, label='fitted line')
    plt.xlabel('log(c)')
    plt.ylabel('log(freq)')
    plt.show()
    

# Perform R/S Analysis
def rs_analysis(data):
    H = nolds.hurst_rs(data, debug_plot=True)
    print("\nHurst (R/S) Analysis:", H)
    print("Beta (R/S) Analysis:", 2*H+1)


# =============================================================================
# Delay Embedding Methods
# =============================================================================

# Autocorrelation vs partial autocorrelation delay setting
def lag_select(series,max_lag,pacf=True,plot=True):
    
    # if the partial autocorrelation, estimate the PACF function
    # if the autocorrelation is used, estimate the ACF function
    if pacf == True:
        corr = sm.tsa.pacf(series, nlags = max_lag)
    else:
        corr = sm.tsa.acf(series, nlags = max_lag)
    
    
    # search for the first zero crossing of the PACF or ACF function
    lag = -1
    
    for i in range(1,len(corr)):
        if corr[i-1] > 0 and corr[i] <= 0:
            lag = i
            break
    
    if lag == -1:
        print("\nNo lag was found!")
    else:
        print("\nLag:", lag)
    
    # plot the results if requested
    if plot==True:
        log_lag = np.log(np.array(list(range(1,lag))))
        log_corr=np.log(corr[1:lag])
        
        slope, intercept, r, p, se = linregress(log_lag,log_corr)
        print("\nSlope:", slope)
        print("\nIntercept:", intercept)
        print("\nR2", r ** 2)
        print("\np-value:", p)
        
        plt.scatter(log_lag,log_corr,c='k',s=2)
        plt.plot(log_lag, intercept + slope * log_lag, 'k--', lw=0.2, label='fitted line')
        plt.xlabel('log(lag)')
        plt.ylabel('log(PACF)')
        plt.show() 
    
    # return the lag for delay embedding
    return lag


# First minimum mutual information lag selection
def mi_lag(series, max_lag):
    
    
    # search for the first minimum of the mutual information
    lag = -1
    
    
    for i in range(2,max_lag):
       
        mi_1 = np.float(mutual_info_regression(series[:-(i-1)].reshape(-1,1), series[i-1:],
                                      discrete_features=[False]))
        mi_2 = np.float(mutual_info_regression(series[:-i].reshape(-1,1), series[i:],
                                      discrete_features=[False]))
        mi_3 = np.float(mutual_info_regression(series[:-(i+1)].reshape(-1,1), series[i+1:],
                                      discrete_features=[False]))
        
                
        if mi_2 < mi_1 and mi_2 < mi_3:
            
             lag = i
             break
        
        
    if lag == -1:
        print("\nNo lag was found")
    else:
        print("\nLag:", i)
    
    return i
    




# Shift series function used for delay embedding
def shift_series(df, # Pandas dataframe
                 target_name, # target variable name
                 lag # lag used
                 ):
    
    predictors = [] # name list of lagged predictors
    
    # add new lagged variable for the chosen lag to the dataframe
    new_variable = target_name+' -'+str(lag)
    predictors.append(new_variable)
    df[new_variable] = df[target_name].shift(lag)
    

# Delay embedding of a series
def embed_series(df, # Pandas dataframe
                  series, # series name
                  dE, # embedding dimension
                  tau # time lag
                  ):
    
    names = [] # names of variables list
    
    # for each dimension
    for i in range(1,dE):
        name = ['lag '+str(i)] # get the lagged variable name
        df[name[0]] = df[series].shift(i * tau) # shift using the lag = i * tau
        names = name + names # add the name to the list of names
       
    names.append(series) # append the series name to the list of names
    
    # get the new dataframe with the list of names for each column
    E=df[names] 
    E=E.dropna()
    
    # return the dataframe for the delay embedding with each entry given
    # by (x(t-(dE-1)*tau),...,x(t-3*tau),x(t-2*tau),x(t-tau),x(t))
    return E



# =============================================================================
# Topological Analysis Methods and Box-Counting Dimension
# =============================================================================

# Function to extract the distance matrix and the recurrence plot
# either for a signal or for an n-dimensional sequence of points
def recurrence_matrix(series, # series of values
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
            plt.matshow(S,cmap=plt.cm.inferno_r)
        
        plt.show()
    
    # return the distance matrix
    if radius != None:
        return B
    elif radius == None:
        return S


# Recurrence analysis function applied to a distance matrix
def recurrence_analysis(S, # distance matrix
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

    

# KNN graph analysis
def knn_analysis(E, k, title, node_size=10, loglog=True, return_graph=False):
    
    # calculate the graph and plot it
    knn = kneighbors_graph(E,n_neighbors=k)
    knn = np.matrix(knn.toarray())
    
    G=nx.from_numpy_matrix(knn)
    
    plt.title('KNN Graph '+title)
    pos = nx.spring_layout(G, seed=1039799)
    nx.draw_networkx_nodes(G, pos, node_size=node_size)
    nx.draw_networkx_edges(G, pos, alpha=0.4)
    plt.show()
    
    # estimate the main graph metrics
    num_nodes = G.number_of_nodes()
    num_edges= G.number_of_edges()
    max_edges = (num_nodes * (num_nodes - 1))/2
    print("\nNumber of nodes:", num_nodes)
    print("\nNumber of edges:", num_edges)
    print("\nMaximum number of edges:", max_edges)
    print("\nProportion Completeness:", num_edges/max_edges)
    
    
    # plot the connected components of the graph
    degree_sequence = sorted((d for n, d in G.degree()), reverse=True)
    fig = plt.figure("Degree for Adjacency Matrix", figsize=(8, 8))
    ax1 = fig.add_subplot()
    Gcc = G.subgraph(sorted(nx.connected_components(G), key=len, reverse=True)[0])
    pos = nx.spring_layout(Gcc, seed=10396953)
    nx.draw_networkx_nodes(Gcc, pos, ax=ax1, node_size=node_size)
    nx.draw_networkx_edges(Gcc, pos, ax=ax1, alpha=0.4)
    ax1.set_title("Connected components of G "+title)
    ax1.set_axis_off()
    plt.show()
    
    # get the degree distribution for the graph
    values, counts = np.unique(degree_sequence, return_counts=True)
    
    # convert the absolute frequencies into relative frequencies
    p = np.array(counts)/np.sum(counts)
    
    # calculate the degree distribution entropy
    H_degree = -p[0] * ( np.log(p[0])/np.log(G.number_of_nodes()) )
   
    for i in range(1,len(p)):
        H_degree = H_degree - p[i] * ( np.log(p[i])/np.log(G.number_of_nodes()) )
   
    print("\nDegree Distribution Entropy:", H_degree)
    
    
    # estimate the Kolmogorov-Sinai entropy of the graph
    Adjacency = nx.to_numpy_array(G,weight=None)
    
    L = np.max(np.linalg.eigvals(Adjacency))
    
    #L = np.max(np.linalg.eigvals(knn))
    
    print("\nKolmogorov-Sinai Entropy:", np.log(L)/np.log(2))
    
    # plot the degree distribution
    if loglog == True:
        values, counts = np.unique(degree_sequence, return_counts=True)
        plt.xticks(np.arange(0, len(values), 10000))
        plt.loglog(values,counts, marker='.')       
    else:
        values, counts = np.unique(degree_sequence, return_counts=True)
        plt.plot(values,counts, marker='.')
    
    plt.title("Degree Distribution "+title)
    plt.xlabel("Degree")
    plt.ylabel("Number of Nodes")
    plt.show()
    
    if return_graph == True:
        return G


# Box counting dimension calculation
def calculate_BoxCounting(sequence,max_bins,title,cutoff):
    # Calculate Box Counting dimension for N-dimensional sequence
    # the algorithm is adapted from 
    # https://francescoturci.net/2016/03/31/box-counting-in-numpy/
    
    Bins=list(range(1,max_bins+1)) # number of bins
    Ns=[] # number of boxes
    num_cols = np.size(sequence,1) # number of columns
        
    # for each bin
    for b in Bins:
        # fompute the Histogram
        bins_values = tuple([b]*num_cols)
        H, edges=np.histogramdd(sequence, 
                                bins=bins_values)
        # sum the number of boxes
        Ns.append(np.sum(H>0))
    
    # take the logarithms of the bins and number of boxes
    log_Bins = np.log(Bins)
    log_Ns = np.log(Ns)
    
    # if no cutoff is used, the data sample includes all elements
    if cutoff == None:
        sample_log_Bins = np.array(log_Bins)
        sample_log_Ns = np.array(log_Ns)
    # otherwise the regression is performed on a subsample for the defined
    # cutoff region:
    else:
        sample_log_Bins = np.array(log_Bins)[cutoff[0]:cutoff[-1]]
        sample_log_Ns = np.array(log_Ns)[cutoff[0]:cutoff[-1]]
        
            
    # perform the regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(sample_log_Bins,
                                                                   sample_log_Ns)
    
    # get the predictions
    log_Pred = slope * sample_log_Bins + intercept
    
    # print the results
    print("\nBox Counting Dimension")
    print("\nR^2:", r_value**2)
    print("R:", r_value)
    print("\nIntercept:", intercept)
    print("Dimension:", slope)
    print("p-value of slope:", round(p_value,6))
    
    # plot the results
    plt.plot(sample_log_Bins,sample_log_Ns, '.',c='k', mfc='none')
    plt.plot(sample_log_Bins,log_Pred)
    plt.xlabel('log 1/s')
    plt.ylabel('log Ns')
    plt.title(title)
    plt.show()


# Evaluate Lyapunov Spectrum
def lyap_spectra_eval(data,max_order,matrix_dim,tau):
    
    # get the orders
    orders = np.arange(1,max_order)
    
    # setup the Lyapunov spectra list
    spectra = []
    
    # dE list
    dE_values = []
    
    # for each embedding dimension...
    for order in orders:
        
        # get the embedding dimension
        dE = order * (matrix_dim - 1) + 1
        
        # append the dimensions
        dE_values.append(dE)
        
        # calculate the Lyapunov spectrum
        lyap_spectrum = nolds.lyap_e(data, emb_dim=dE,
                                     matrix_dim=matrix_dim,tau=tau)
        # update the spectra list
        spectra.append(lyap_spectrum)
    
    # plot the Lyapunov spectra
    plt.plot(dE_values, spectra,marker='.')
    plt.xlabel("dE")
    plt.ylabel("Lyapunov Exponent")
    plt.show()
    
    # print the last estimated exponents
    sum_spectrum = 0
    DKY = None
    for i in range(0,len(lyap_spectrum)):
        sum_spectrum_previous = sum_spectrum
        print("\nL"+str(i+1), lyap_spectrum[i])
        sum_spectrum += lyap_spectrum[i]
        if sum_spectrum_previous >= 0 and sum_spectrum < 0:
            DKY = (i - 1) + sum_spectrum_previous / abs(lyap_spectrum[i])
            
        
    print("\nLyapunov Spectrum Sum:", sum_spectrum)
    if DKY != None:
        print("\nKaplan-Yorke Dimension:", DKY)


# Calculate Lyapunov Exponents
def calculate_lyap(data,dE,tau,matrix_dim=4,spectrum=True):
    
    # if one is not evaluating the Lyapunov spectrum...
    if spectrum == False:
        # use Rosenstein et al. method
        L1 = nolds.lyap_r(data, emb_dim=dE, lag=tau)
        print("\nLargest Lyapunov Exponent (Rosenstein et al.):", L1)
    # otherwise use the Eckmann et al. method
    else:
        # Lyapunov spectrum
        lyap_spectrum = nolds.lyap_e(data, emb_dim=dE,
                                     matrix_dim=matrix_dim,tau=tau)
        print("\nLyapunov Spectrum (Eckmann et al.):")
        sum_spectrum = 0
        for i in range(0,len(lyap_spectrum)):
            print("\nL"+str(i+1), lyap_spectrum[i])
            sum_spectrum += lyap_spectrum[i]
        print("\nLyapunov Spectrum Sum:", sum_spectrum)



# =============================================================================
# Prediction Methods
# =============================================================================




# Machine learning-based prediction method
def predict(E, # embedded series
            ml, # machine learning model
            target_name, # target variable name
            window, # sliding window for learning
            horizon=1 # prediction horizon
            ):
    
    X=np.array(E[:-horizon]) # predictors
    y=np.array(E[target_name][horizon:]) # target series
    
    y_pred = [] # list of predictions
    
    # train the AI using the sliding window
    for t in range(window,len(y)-1):
        ml=ml.fit(X[t-window:t,:],y[t-window:t])
        prediction = ml.predict([X[t,:]])[0]
        if prediction < 0:
            prediction = 0
        y_pred.append(prediction)
    
    # return the predictions and the observed values
    return y_pred, y[window+1:]
            

# Get the prediction metrics
def prediction_metrics(y_pred, y_target, title, return_residuals=False):

    plt.title(title)
    plt.plot(y_target,lw=0.5)
    plt.plot(y_pred,lw=0.5)
    plt.show()

    plt.scatter(y_target,y_pred,s=0.5)
    plt.xlabel('observed')
    plt.ylabel('predicted')
    plt.show()
    
    print("\nRho:", np.corrcoef(y_pred,y_target)[0][1])

    print("\nRMSE:", mse(y_target, y_pred,squared=False))
    print("\nRMSE/Amplitude:", mse(y_target, y_pred,squared=False)/(max(y_target)-min(y_target)))
    print("\nExplained Variance:", e_variance(y_target,y_pred))
    print("\nMax Deviation:", max(np.abs(y_target-y_pred)))
    print("\nR2 score", r2_score(y_target,y_pred))
    
    
    if return_residuals == True:
        return y_target-y_pred



# Estimate the prediction dimension retrieving the dimension with the highest
# R2
def prediction_dimension(df,series,dim_list,tau,ml,window,metric,printouts=False):
    
    metric_list=[]
    
    for dE in dim_list:
        E=embed_series(df=df,series=series,dE=dE,tau=tau)
        y_pred, y_target = predict(E=E,ml=ml,target_name=series,window=window,horizon=1)
        if metric == 'R2':
            score = r2_score(y_target,y_pred)
        elif metric == 'error':
            score = mse(y_target, y_pred,squared=False)/(max(y_target)-min(y_target))
        metric_list.append(score)
        if printouts == True:
            print("\nTesting Dimension:", dE)
            print("Metric score:", score)
    if metric == 'R2':
        optimal_score = max(metric_list)
        ind=np.where(np.array(metric_list)==optimal_score)
    elif metric == 'error':
        optimal_score = min(metric_list)
        ind=np.where(np.array(metric_list)==optimal_score)
    
    dE=dim_list[ind][0]
        
    print("\nOptimal Dimension:", dE)
    print("Score:",optimal_score)
    
    plt.plot(dim_list,metric_list,'.',c='k')
    plt.xlabel('Embedding Dimension')
    plt.ylabel('Score')
    plt.show()
    
    return dE, optimal_score
    
# Correlation Sum
def correlation_sum(S, # distance matrix
                    radius # radius
                    ):
    # function to get the correlation integral
    B = S <= radius
    B = 1*B
    num_lines = B.shape[0]
    sum_values = np.sum(B) - num_lines
    return sum_values /(num_lines * (num_lines - 1))
    


# Calculate the Correlation Dimension
def correlation_dimension(S, # distance matrix
                          radii, # radii list
                          plot_result=True, # plot the observed and fitted line
                          return_results=False # return results
                          ):
    log_Cr = [] # list of logarithms of the correlation sums
    log_r = [] # list of logarithms of the radii
    
    # for each radius
    for radius in radii:
        # calculate the correlation sum
        Cr = correlation_sum(S,radius)
        # if there are recurrences
        if Cr != 0:
            # append the logarithm of the radius to the log_r list
            log_r.append(np.log(radius))
            # append the logarithm of the correlation sum to the log_Cr list
            log_Cr.append(np.log(Cr))
    
    
    slope, intercept, r, p, se = linregress(log_r,log_Cr)
    
    if plot_result == True:
        plt.scatter(log_r,log_Cr)
        plt.plot(log_r, intercept + slope * np.array(log_r), 'k--', lw=0.2)
        plt.xlabel("log(r)")
        plt.ylabel("log(Cr)")
        plt.show()
        
    print("\nDimension:", slope)
    print("\nR2", r ** 2)
    print("\np-value:", p)
    
    
    if return_results == True:
        return log_r, log_Cr, slope, intercept
    
# Calculate Correlation Dimensions for Different Embeddings
def correlation_dimensions(df, # Pandas dataframe
                           series, # series name
                           max_dE, # maximum embedding dimension
                           tau, # time delay
                           radii # radii list
                           ):
    
    embedding_dimensions = list(np.arange(2,max_dE+1)) # embedding dimensions
    estimated_dimensions = [] # estimated dimensions list
    
    # for each embedding dimension
    for dE in embedding_dimensions:
        print("\nEmbedding dimension:", dE)
        # embed the series
        E = embed_series(df,series,dE,tau)
        # get the recurrence matrix
        S = recurrence_matrix(E,radius=None,one_dim=False,plot=False)
        # estimate the correlation dimension
        log_r, log_Cr, slope, intercept = correlation_dimension(S,radii,
                                                                plot_result=False,
                                                                return_results=True)
        # plot the fitted lines
        plt.scatter(log_r,log_Cr,s=1)
        plt.plot(log_r, intercept + slope * np.array(log_r), 'k--', lw=0.2)
        
        # store the estimated dimensions
        estimated_dimensions.append(slope)
    
    plt.xlabel("log(r)")
    plt.ylabel("log(Cr)")
    plt.show()
    
    plt.plot(embedding_dimensions, estimated_dimensions,lw=1,marker='.')
    plt.xlabel("Embedding Dimension")
    plt.ylabel('Correlation Dimension')
    plt.show()

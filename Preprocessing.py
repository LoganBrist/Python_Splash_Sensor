# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 22:24:23 2019

@author: Logan
"""

# =============================================================================
# Data Preprocessing
# =============================================================================

import CustomFunctions as cf
import pandas as pd
import numpy as np

#list of data directories for easy swapping
path = []
path.append(r"Data\Continuous Motion/") #path[0]  
path.append(r"Data\Stationary/")        #path[1] 
path.append(r"Data\Baby Drop/")         #2
path.append(r"Data\Basketball/")        #3
path.append(r"Data\Duck/")              #4
path.append(r"Data\Frisbee/")           #5
path.append(r"Data\Greg/")              #6
path.append(r"Data\Lemon/")             #7
path.append(r"Data\Hose Nozzle/")      #8
path.append(r"Data\Pee/")               #9
path.append(r"Data\Pool Jet/")          #10
path.append(r"Data\Small Object/")      #11

data_x = []
data_y = []
data_z = []

for i in range(len(path)):
    fs = 283.81           #sample rate of testing setup
    directory   = path[i] #Which data set to use (see above for list)
    nameroot    = 'DATA'  #Data file name, excluding the number
    fileformat  = '.TXT'  #Data file format
    t0    = 30            #window start time in seconds
    tf    = 60            #Window end time in seconds (absolute, not relative to t0)
    
    #------------------------------------------------------------------------------
    # Format Data
    
    #Load accelerometer trials (ex. x[0] is trial one etc.)
    x,y,z,classification = cf.loaddata(directory,nameroot,fileformat)
    
    #each category is classified
    classification = i 
    
    ''''
    if(i == 1):
        classification = 0
    else:
        classification = 1
    '''
        
    
    #Create time vectors for each trial (for plot time axis)
    t = cf.timevector(z,fs)
    
    #Window each trial in time, from t0 to tf seconds
    z = cf.window(z,fs,t0,tf) 
    y = cf.window(y,fs,t0,tf) 
    x = cf.window(x,fs,t0,tf) 
    t = cf.window(t,fs,t0,tf)
    
    
    for j in range(len(t)):
        #average filter 
        #N = 3
        #z_smooth = np.convolve(z[j], np.ones((N,))/N, mode='valid') 
        #y_smooth = np.convolve(y[j], np.ones((N,))/N, mode='valid') 
        #x_smooth = np.convolve(x[j], np.ones((N,))/N, mode='valid') 
        
        #normalize z axis
        z[j] = z[j] + 1
        
        #convert to list
        z_trial = list(z[j])
        y_trial = list(y[j])
        x_trial = list(x[j])
        
        z_trial.append(classification)
        y_trial.append(classification)
        x_trial.append(classification)
        
        data_z.append(z_trial)
        data_y.append(y_trial)
        data_x.append(x_trial)
        
    print(path[i],' loaded.')  

dfx = pd.DataFrame(data_x)
dfy = pd.DataFrame(data_y)
dfz = pd.DataFrame(data_z)

dfx.to_csv('data_x.csv')
dfy.to_csv('data_y.csv')
dfz.to_csv('data_z.csv')

print('Saved to CSV files. ')

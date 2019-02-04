# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 18:07:21 2019

@author: Logan
"""
# =============================================================================
# Processing for the accelerometer signal
# 
# Group 41
# =============================================================================

#Import support vector machine functions
from sklearn import svm
import os
from matplotlib import pyplot as plt
from scipy import signal
import numpy as np
from scipy.fftpack import rfft

#separateing the data 
x = []    
y = []
z = []
classification = []

#Retrieve data for every trial
file_names = os.listdir()

for file in file_names:
    if 'data' and '.txt' in file:
        pass #execute code to read data
    else: 
        continue #check next file
    
    with open(file,'r') as data:    
        #data = open(file,'r') 
        
        #vectors for current file
        x_temp = []  
        y_temp = []   
        z_temp = []
        
        label = 'No classification'
        
        for line in data:
            #separate characters by spaces
            word = line.split()
            
            #if the row contains numbers, separate them into the vectors
            try:
                x_temp.append(float(word[0]))
                y_temp.append(float(word[1]))
                z_temp.append(float(word[2]))
    
            except:
                pass
            
            #if the row is the classification label, record the classification
            try:
                if word[0] == 'classification:':
                    label = word[1]
            
            except:
                pass
    
        #add data to one row of the entire data set  
        #lists converted into arrays for further fast processing, spectrogram
        # needs an array attribute "size" so it won't work without this conversion
        x.append(np.array(x_temp)) 
        y.append(np.array(y_temp))
        z.append(np.array(z_temp))
        classification.append(label)
        
        data.close()
    

# Plots the data for each trial
for i in range (0,len(x)):
    plt.plot(x[i],label="X")
    plt.plot(y[i],label="Y")
    plt.plot(z[i],label="Z")
    plt.legend()
    plt.title('Acceleration Data ' + str(i) + ' (' + classification[i] + ')')
    plt.figure()

#STFT of z signal
fs = 48
f, t, Zxx = signal.stft(z[1], fs,nperseg = 128)
plt.pcolormesh(t,f,np.abs(Zxx))
plt.title('STFT Magnitude')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()

#Spectrogram of z signal
f, t, Sxx = signal.spectrogram(z[1], fs, window = 'hamming', nperseg = 128)
plt.pcolormesh(t, f, Sxx)
plt.title('Spectrogram Magnitude')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [sec]')
plt.show()

# tbc.......

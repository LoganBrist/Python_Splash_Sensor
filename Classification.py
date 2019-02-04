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
        
    data = open(file,'r') 
    
    #vectors for current file
    x_temp = []  
    y_temp = []   
    z_temp = []
    
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
                classification.append(word[1])
        
        except:
            pass
        
    #add data to one row of the entire data set    
    x.append(x_temp)
    y.append(y_temp)
    z.append(z_temp)
    
    data.close()


# Plots the data for each trial
for i in range (0,len(x)):
    plt.plot(x[i],label="X")
    plt.plot(y[i],label="Y")
    plt.plot(z[i],label="Z")
    plt.legend()
    plt.title('Acceleration Data (' + classification[i] + ')')
    plt.figure()


# tbc.......

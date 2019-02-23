# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 13:47:41 2019

@author: Logan
"""
from sklearn import svm
import os
from matplotlib import pyplot as plt
from scipy import signal
import numpy as np
import scipy.fftpack as fftp
#------------------------------------------------------------------------------

def plot(x,t = None,y_min = -.1, y_max = .1, cla = None): 
    for i in range (len(x)): 
        #with or without time in second   
        if (t == None):   
           plt.plot(x[i])
           plt.xlabel("sample")
        else:
            plt.plot(t[i],x[i])
            plt.xlabel("seconds")
                   
        plt.ylim([y_min,y_max]) 
        
        if cla == None:
            clas = "No classification"
        else:
            clas = cla[i]  
                      
        plt.title('Acceleration Data ' + str(i) + ' (' + clas + ')')
        plt.figure()
#------------------------------------------------------------------------------

def fft(x,width = None,fs = None,f_min = 0,f_max = None,y_min = 0, y_max = 100, cla = None):
    #Default parameter values (that depend on other parameters)
    width = width or len(x)
    fs    = fs    or len(x)
    f_max = f_max or fs/2
    
    fft = []
    for i in range(len(x)):
        t = np.linspace(0,fs,width)
        z = fftp.fft(x[i],width)
        plt.plot(t,abs(z))
        plt.xlim([f_min,f_max])
        plt.ylim([y_min,y_max])
        
        if cla == None:
            clas = "No classification"
        else:
            clas = cla[i]
        
        plt.title('FFT Magnitude ' + str(i) + ' (' + clas + ')')
        plt.ylabel('Magnitude')
        plt.xlabel('Frequency [Hz]')
        plt.show()
        fft.append([t,z])
    return fft
#------------------------------------------------------------------------------
     
def psd(x, width = 128, fs = 1, f_min = 0, f_max = None, p_min = 1e-11,p_max = 1e2,cla=None) :
    psd = []
    for i in range(len(x)):
        f,Pxx = signal.periodogram(x[i],fs,nfft=width)
        plt.semilogy(f, Pxx)
        plt.ylim([p_min,p_max])
        f_mx = f_max or max(f)
        plt.xlim([f_min,f_mx])
        
        if cla == None:
            clas = "No classification"
        else:
            clas = cla[i]
            
        plt.title('Power Spectral Density ' + str(i) + ' (' + clas + ')')
        plt.ylabel('Power')
        plt.xlabel('Frequency [Hz]')
        plt.show()
        psd.append([f,Pxx])
    return psd

#------------------------------------------------------------------------------
def stft(x, width = 128, fs = 1, f_min = 0, f_max = 10,cla=None):
    stft = []
    for i in range(len(x)):
        f, t, Zxx = signal.stft(x[i], fs, nperseg = width)
        plt.pcolormesh(t,f,np.abs(Zxx))
        plt.ylim([f_min,f_max])
        
        if cla == None:
            clas = "No classification"
        else:
            clas = cla[i]
            
        plt.title('STFT Magnitude ' + str(i) + ' (' + clas + ')')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.show()
        stft.append([f, t, Zxx])
    return stft
#------------------------------------------------------------------------------

def spectrogram(x, width = 128, fs = 1, f_min = 0, f_max = 10,cla= None):
    spec = []
    for i in range(len(x)):
        f, t, Sxx = signal.spectrogram(x[i], fs, window = 'hamming', nperseg = width)
        plt.pcolormesh(t, f, Sxx)
        plt.ylim([f_min,f_max])
        
        if cla == None:
            clas = "No classification"
        else:
            clas = cla[i]
            
        plt.title('Spectrogram Magnitude ' + str(i) + ' (' + clas + ')')
        plt.ylabel('Frequency [Hz]')
        plt.xlabel('Time [sec]')
        plt.show()
        spec.append([f, t, Sxx])
    return spec 
#------------------------------------------------------------------------------

def window(x,fs,t0,tf):
    n0  = round(fs * t0)
    n1  = round(fs * tf)
    y = []
    for i in range(len(x)):
        try:
            y_temp = x[i][n0:n1]
        except:
            print("Time is out of range, No filter applied.")
        y.append(y_temp)          
    return y
#------------------------------------------------------------------------------

def timevector(x,fs):
    t = []
    for i in range(len(x)):      
        t_temp = []
        for i in range(len(x[i])):
            t_temp.append(i/fs)
        t.append(t_temp)   
    return t
    
#------------------------------------------------------------------------------

def loaddata(path,nameroot,fileformat):
    #Return lists 
    x = []    
    y = []
    z = []
    classification = []
    
    #Retrieve data for every trial
    file_names = os.listdir(path) 
    
    for file in file_names:
        #Check if file matches the parameter name and format
        if nameroot and fileformat in file:
            pass #execute code to read data
        else: 
            continue #check next file
        
        fullpath = path + file
        with open(fullpath,'r') as data:    
            #temporary vectors for current file
            x_temp = []  
            y_temp = []   
            z_temp = []
            
            #initialize label in case its not overwritten
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
            z.append(np.array(y_temp))
            
            classification.append(label)
            
            data.close()
            
    return x,y,z,classification
#------------------------------------------------------------------------------


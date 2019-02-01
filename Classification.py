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

#import text file with time/acceleration stamps
data = open('exampledata.txt','r') 

#separate data into two lists
t = []    #time vector
a = []    #acceleration vector
y = []    #classification vector

t_temp = []   #vector for current file
a_temp = []   #vector for current file
for line in data:  
    word = line.split()
    
    if line == 0:  
        y.append(word[1])   #issue here, but gtg
    if  line == 1:
        continue
    else:
        t_temp.append(float(word[0]))
        a_temp.append(float(word[1]))
    
#add data to one row of the entire data set    
t.append(t_temp)
a.append(a_temp)

print("time:", t)
print("acc.:", a)
print("classification:",y)

# tbc.......
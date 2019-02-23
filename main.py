# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 18:07:21 2019
@author: Logan
"""
# =============================================================================
# Processing for splash sensor
# Group 41
# Desription: All critical lines of code for processing accelerometer 
# data goes in this program. All functions are found in CustomFunctions.py, 
# which is where most of the code is. This keeps everything clean and keeps the 
# main program straight forward  
# 
# =============================================================================

#------------------------------------------------------------------------------
# Imports and variales

#Import Functions from function file
import CustomFunctions as cf

#list of data directories for easy swapping
path = []
path.append(r"Data\Continuous Motion/") #path[0]  
path.append(r"Data\Small Object/")      #path[1] 
path.append(r"Data\Stationary/")        #path[2] 
path.append(r"Data\Sample Frequency/")  #path[3]

#Parameters
fs = 283.81           #sample rate of testing setup
directory   = path[2] #Which data set to use (see above for list)
nameroot    = 'DATA'  #Data file name, excluding the number
fileformat  = '.TXT'  #Data file format
t0    = 25            #window start time in seconds
tf    = 60            #Window end time in seconds (absolute, not relative to t0)
#------------------------------------------------------------------------------
# Format Data

#Load accelerometer trials (ex. x[0] is trial one etc.)
x,y,z,classification = cf.loaddata(directory,nameroot,fileformat)

#Create time vectors for each trial (for plot time axis)
t = cf.timevector(z,fs)

#Window each trial in time, from t0 to tf seconds
z = cf.window(z,fs,t0,tf) 
t = cf.window(t,fs,t0,tf)

#------------------------------------------------------------------------------
# Plots

# Plot acceleration vs. time for each trial
cf.plot(z,t,y_min = -.2, y_max = .1, cla = classification)
   
# Calculate and Plot FFT for each trial
fft  = cf.fft(z,len(z[0]),fs,f_min = 0,f_max = 10,y_max = 10,cla = classification) 

#Calculate and Plot Power Spectral Density for each trial
psd =  cf.psd(z,len(z[0]), fs, f_min = 0, f_max = 10,cla= classification)

#Calculate and Plot STFT for each trial, return in form [freq bins,time bins,value] 
stft = cf.stft(z,512,fs,0,10,cla = classification)

#Calculate and Plot Spectrogram for each trials, return in form [freq bins,time bins,value]
spectrogram = cf.spectrogram(z,512,fs,0,10,cla = classification)

#------------------------------------------------------------------------------
# Filters

  #averaging filter
  
#------------------------------------------------------------------------------
# Features
  
  #Mean
  #PSD
  #Peak frequencies
  #Peak distance

#------------------------------------------------------------------------------
# Statistical analysis
  
  #Correlation matrix - belongs in Customfunctions
  def correl_matrix(X,cols):
    fig = plt.figure(figsize=(7,7), dpi=100)
    ax1 = fig.add_subplot(111)
    cmap = cm.get_cmap('jet',30)
    cax = ax1.imshow(np.abs(X.corr()),interpolation='nearest',cmap=cmap)
    major_ticks = np.arange(0,len(cols),1)
    ax1.set_xticks(major_ticks)
    ax1.set_yticks(major_ticks)
    ax1.grid(True,which='both',axis='both')
    plt.title('Correlation Matrix')
    labels = cols
    ax1.set_xticklabels(labels,fontsize=9)
    ax1.set_yticklabels(labels,fontsize=12)
    fig.colorbar(cax, ticks=[-0.4,-0.25,-.1,0,0.1,.25,.5,.75,1])
    plt.show()
    return(1)
    
    #Pair plotting - belongs in Customfunctions
    def pairplotting(df):
        sns.set(style='whitegrid', context='notebook')
        cols = df.columns
        sns.pairplot(df[cols],size=2.5)
        plt.show()
#------------------------------------------------------------------------------
# Training - This may or may not belongs in customfunctions
        
# Perceptron
        
from sklearn.linear_model import Perceptron

ppn = Perceptron(max_iter=4, tol=1e-3, eta0=0.001, fit_intercept=True, random_state=0, verbose=True)

# Training
ppn.fit(X_train_std, y_train)

# Prediction
y_pred = ppn.predict(X_test_std)

print('\n\nPerceptron\n')
# Accuracy score
#print('\nMisclassified samples: %d' % (y_test != y_pred).sum())
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))

# Combined Accuracy score
y_combined_pred = ppn.predict(X_combined_std)
#print('\nMisclassified combined samples: %d' % (y_combined != y_combined_pred).sum())
print('Combined Accuracy: %.2f' % accuracy_score(y_combined, y_combined_pred))


# Logistic Regression

print('\n\nLogistic Regression\n')

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(C=1000.0, random_state=0)

# Training
lr.fit(X_train_std, y_train)

# Prediction
y_pred = lr.predict(X_test_std)

# Accuracy score
#print('Misclassified samples: %d' % (y_test != y_pred).sum())
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))

# Combined Accuracy score
y_combined_pred = lr.predict(X_combined_std)
#print('\nMisclassified combined samples: %d' % (y_combined != y_combined_pred).sum())
print('Combined Accuracy: %.2f' % accuracy_score(y_combined, y_combined_pred))



# Support Vector Machine 


print('\n\nSupport Vector Machine\n')

from sklearn.svm import SVC
svm = SVC(kernel='rbf', tol=1e-3, random_state=0, gamma=0.2 , C=10.0, verbose=True)

# Training
svm.fit(X_train_std, y_train)

# Prediction
y_pred = svm.predict(X_test_std)

# Accuracy score
#print('Misclassified samples: %d' % (y_test != y_pred).sum())
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))

# Combined Accuracy score
y_combined_pred = svm.predict(X_combined_std)
#print('\nMisclassified combined samples: %d' % (y_combined != y_combined_pred).sum())
print('Combined Accuracy: %.2f' % accuracy_score(y_combined, y_combined_pred))


# Random Forest Decision Tree

print('\n\nRandom Forest Decision Tree\n')

from sklearn.ensemble import RandomForestClassifier
forest = RandomForestClassifier(criterion='entropy', n_estimators=10 ,random_state=1, n_jobs=2)

# Training
forest.fit(X_train_std,y_train)

# Prediction
y_pred = forest.predict(X_test_std)

# Accuracy score
#print('Misclassified samples: %d' % (y_test != y_pred).sum())
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))

# Combined Accuracy score
y_combined_pred = forest.predict(X_combined_std)
#print('Misclassified combined samples: %d' % (y_combined != y_combined_pred).sum())
print('Combined Accuracy: %.2f' % accuracy_score(y_combined, y_combined_pred))
    

# K-Nearest Neighbor 

print('\n\nK-Nearest Neighbor\n')

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3,p=2,metric='minkowski')

# training
knn.fit(X_train_std,y_train)

# Prediction
y_pred = knn.predict(X_test_std)

# Accuracy score
#print('Misclassified samples: %d' % (y_test != y_pred).sum())
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))

# Combined Accuracy score
y_combined_pred = knn.predict(X_combined_std)
#print('Misclassified combined samples: %d' % (y_combined != y_combined_pred).sum())
print('Combined Accuracy: %.2f' % accuracy_score(y_combined, y_combined_pred))

#------------------------------------------------------------------------------
# Finish? 

#After training, we should have a set of values from whichever algorithm works best
# that can be exported. Coding then needs to be done on the microcontroller to 
# stream accelerometer data through the predictor values.

  
# tbc.......
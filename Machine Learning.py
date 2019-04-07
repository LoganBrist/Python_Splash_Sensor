# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 22:53:03 2019

@author: Logan
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Mar  5 14:05:23 2019

@author: Logan
"""
# =============================================================================
#  Notes
#
#  Changing number of components on PCA analysis doesn't seem effective. Let's extract
#  signal power, peak heights, and peak spread as variables from the time series data.
#  
# =============================================================================
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

splashdata = pd.read_csv('data_z.csv',header = None)

#Test-Train split
endrow,endcol = splashdata.shape
from sklearn.cross_validation import train_test_split
X,y = splashdata.iloc[1:,1:endcol-1].values, splashdata.iloc[1:,endcol-1].values

X_filtered = []
y_filtered = []

for i in range(len(X)):
    #Stationary, pool jet, or sprinkle
    if y[i] == (1 or 10 or 9):
        X_filtered.append(X[i])
        y_filtered.append(0)
        
    #Greg    
    elif y[i] == 6:
        X_filtered.append(X[i])
        y_filtered.append(2)
    
    #Baby, duck
    elif y[i] == (2 or 4): 
        X_filtered.append(X[i])
        y_filtered.append(1)
        
X_train, X_test, y_train, y_test = train_test_split(X_filtered,y_filtered,test_size=0.3,random_state=0)

#Standardize
from sklearn.preprocessing import StandardScaler
stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test) 


##############################################################################
# Principle Component Analysis
##############################################################################

from sklearn.decomposition import PCA
pca = PCA(n_components=3)
X_train_std = pca.fit_transform(X_train_std) #X_train
X_test_std =  pca.transform(X_test_std)      #X_test


# Combined data
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))


##############################################################################
# Perceptron
##############################################################################

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

#confusion matrix
confmat = confusion_matrix(y_true=y_test, y_pred = y_pred)
fig,ax = plt.subplots(figsize=(2.5,2.5))
ax.matshow(confmat, cmap=plt.cm.Blues,alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j,y=i,s=confmat[i,j],va='center',ha='center')
plt.xlabel('predicted label')
plt.ylabel('true label')
plt.show()

y_combined
##############################################################################
# Logistic Regression
##############################################################################

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


#confusion matrix
#confmat = confusion_matrix(y_true=y_test, y_pred = y_pred)
confmat = confusion_matrix(y_true=y_combined, y_pred = y_combined_pred)
fig,ax = plt.subplots(figsize=(2.5,2.5))
ax.matshow(confmat, cmap=plt.cm.Blues,alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j,y=i,s=confmat[i,j],va='center',ha='center')
plt.xlabel('predicted label')
plt.ylabel('true label')
plt.show()

##############################################################################
# Support Vector Machine (pick one version) 
##############################################################################

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

#confusion matrix
#confmat = confusion_matrix(y_true=y_test, y_pred = y_pred)
confmat = confusion_matrix(y_true=y_combined, y_pred = y_combined_pred)
fig,ax = plt.subplots(figsize=(2.5,2.5))
ax.matshow(confmat, cmap=plt.cm.Blues,alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j,y=i,s=confmat[i,j],va='center',ha='center')
plt.xlabel('predicted label')
plt.ylabel('true label')
plt.show()

##############################################################################
# Random Forest Decision Tree
##############################################################################

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

#confusion matrix
confmat = confusion_matrix(y_true=y_test, y_pred = y_pred)
fig,ax = plt.subplots(figsize=(2.5,2.5))
ax.matshow(confmat, cmap=plt.cm.Blues,alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j,y=i,s=confmat[i,j],va='center',ha='center')
plt.xlabel('predicted label')
plt.ylabel('true label')
plt.show()    

##############################################################################
# K-Nearest Neighbor (find the best value of K) 
##############################################################################

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

#confusion matrix
confmat = confusion_matrix(y_true=y_test, y_pred = y_pred)
fig,ax = plt.subplots(figsize=(2.5,2.5))
ax.matshow(confmat, cmap=plt.cm.Blues,alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j,y=i,s=confmat[i,j],va='center',ha='center')
plt.xlabel('predicted label')
plt.ylabel('true label')
plt.show()

##############################################################################
# Clustering
##############################################################################

print('\n\nK means\n')

from sklearn.cluster import KMeans
km = KMeans(n_clusters=2,init='k-means++',n_init=10,max_iter=300,tol=1e-4,random_state=0)

#Training and Prediction
y_pred = km.fit_predict(X_test_std)

# Accuracy score
#print('Misclassified samples: %d' % (y_test != y_pred).sum())
print('Accuracy: %.2f' % accuracy_score(y_test, y_pred))

# Combined Accuracy score
y_combined_pred = knn.predict(X_combined_std)
#print('Misclassified combined samples: %d' % (y_combined != y_combined_pred).sum())
print('Combined Accuracy: %.2f' % accuracy_score(y_combined, y_combined_pred))

#confusion matrix
confmat = confusion_matrix(y_true=y_test, y_pred = y_pred)
fig,ax = plt.subplots(figsize=(2.5,2.5))
ax.matshow(confmat, cmap=plt.cm.Blues,alpha=0.3)
for i in range(confmat.shape[0]):
    for j in range(confmat.shape[1]):
        ax.text(x=j,y=i,s=confmat[i,j],va='center',ha='center')
plt.xlabel('predicted label')
plt.ylabel('true label')
plt.show()



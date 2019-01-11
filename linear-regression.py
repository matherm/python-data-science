# -*- coding: utf-8 -*-
"""
Spyder Editor

This temporary script file is located here:
C:\Users\Matthias\.spyder2\.temp.py
"""
#%%
import numpy as np
import pandas as pd
from scipy import stats
from scipy import linalg
from sklearn.svm import  *
import matplotlib.pyplot as plt

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data"
cols = ["CRIM","ZN","INDUS","CHAS","NOX","RM","AGE","DIS","RAD","TAX","PTRATIO","B","LSTAT","TGT"]
boston = pd.read_csv (url , sep=" ", skipinitialspace=True , header=None ,names=cols , index_col=False )

X = boston.TGT.values
Y = boston.RM.values

Z = np.polyfit(X, Y, 2)
print(Z)

xe = np.arange(0, 90, 1)
#X, Y = np.meshgrid(x, y)
F = Z[0]*xe + Z[1]*xe +Z[2]
 
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.plot_surface(X, Y, F)
#plt.show()


plt.plot(X,Y, "ro")
plt.plot(xe, F)
R = stats.pearsonr(X,Y)
print(R[0])
A = np.array([ [1,2,3], [4,5,6], [7,8,9] ])
for data in xrange(0,3):
    print(data)

print(linalg.pinv(A))
print(linalg.inv(A))

list = ["abc","def","ghi"]
print(len(list))
print()
print(A.shape)


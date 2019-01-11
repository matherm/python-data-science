# -*- coding: utf-8 -*-
"""
Spyder Editor

Dies ist eine temporäre Skriptdatei.
"""

import numpy as np
import numpy.random as random
import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline

data=random.uniform(0,1,1000)
hist, bin_edges = np.histogram(data, normed=True, bins = 100)
cdf = np.cumsum(hist)

data=random.normal(0,1,1000)
hist, bin_edges = np.histogram(data, normed=True, bins = 100)
cdf2 = np.cumsum(hist)

plt.figure(figsize=(12,8),facecolor='1.0') 
plt.plot(cdf,cdf2,"o")



norm=random.normal(0,2,len(data))
norm.sort()
plt.figure(figsize=(12,8),facecolor='1.0') 

plt.plot(norm,data,"o")

#generate a trend line as in http://widu.tumblr.com/post/43624347354/matplotlib-trendline
z = np.polyfit(norm,data, 1)
p = np.poly1d(z)
plt.plot(norm,p(norm),"k--", linewidth=2)
plt.title("Normal Q-Q plot", size=28)
plt.xlabel("Theoretical quantiles", size=24)
plt.ylabel("Expreimental quantiles", size=24)
plt.tick_params(labelsize=16)
plt.show()
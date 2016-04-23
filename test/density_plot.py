# -*- coding: utf-8 -*-
"""
This is a file to show plot density picture.

@author: York(xy_xie@nuaa.edu.cn)
"""

import numpy as np  
import matplotlib.pyplot as plt  
from scipy.stats import gaussian_kde

x = np.loadtxt('points.txt')
x = x.reshape((3466, 5, 2))

left_eye = x[:, 0, :]
right_eye = x[:, 1, :]
nose = x[:, 2, :]
left_mouse = x[:, 3, :]
right_mouse = x[:, 4, :]

data = [left_eye, right_eye, nose, left_mouse, right_mouse]

for dat in data:
    # Calculate the point density
    xy = np.vstack([dat[:, 0],dat[:, 1]])
    z = gaussian_kde(xy)(xy)

    # Sort the points by density, so that the densest points are plotted last
    idx = z.argsort()
    x, y, z = dat[idx,0], dat[idx,1], z[idx]

    plt.scatter(x, y, c=z, s=30, edgecolor='')
    
plt.savefig('density.png')
#plt.show()
    
plt.figure()

plt.scatter(left_eye[:,0], left_eye[:,1], marker = '>', color = 'm', label='L_E', s = 15)  

plt.scatter(right_eye[:,0], right_eye[:,1], marker = '<', color = 'c', label='R_E', s = 15)  

plt.scatter(nose[:,0], nose[:,1], marker = 'o', color = 'r', label='N', s = 15)  

plt.scatter(left_mouse[:,0], left_mouse[:,1], marker = 'x', color = 'g', label='L_M', s = 30)  

plt.scatter(right_mouse[:,0], right_mouse[:,1], marker = '+', color = 'b', label='R_M', s = 30)  

plt.legend(loc = 'upper left',prop={'size':8})  

plt.savefig('scatter.png')
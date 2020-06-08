#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 23 15:27:23 2020

@author: matheus
"""
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

receptorsPositions = np.zeros((4,3))
receptorsPositions[0] = np.array([-0.83433945, -0.47652493, -0.17790941])
receptorsPositions[1] = np.array([0.96742833, -0.02519705,  0.23273241])
receptorsPositions[2] = np.array([0.20181641,  0.19504837, -0.1400522])
receptorsPositions[3] = np.array([-0.02670906,  0.99015801, -0.11934458])

array = receptorsPositions
array_ax = plt.axes(projection='3d')
array_ax.set_xlim([-1.5, 1.5])
array_ax.set_xlim([-1.5, 1.5])
array_ax.set_xlim([-1.5, 1.5])
array_ax.scatter3D(array[:,0], array[:,1], array[:,2], c='darkblue', depthshade=False , s=40)
for i in range(array.shape[0]):
    array_ax.text(array[i,0], array[i,1], array[i,2]+0.1, f"mic {str(i)}", zorder=1)
plt.title("Disposição espacial dos microfones", fontsize=12)
plt.show()



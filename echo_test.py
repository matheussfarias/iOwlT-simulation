#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr  3 22:23:00 2020

@author: davi
"""

import numpy as np
import matplotlib.pyplot as plt
from multilateration_algorithms import *
import scipy.signal as scc
import scipy.fftpack as ff


sampleRate = 32000

# defining position of source
sourcePosition = np.array([-5.53, 8.74, 1.35]) 

numberOfReceptors = 5
numberOfDimentions = 3

receptorsPositions = np.zeros((numberOfReceptors, numberOfDimentions)) # ndarray with receptors positions

# defining position of receivers
receptorsPositions[0] = np.array([1, 1, 0])
receptorsPositions[1] = np.array([-1, 1, 0])
receptorsPositions[2] = np.array([1, -1, 0])
receptorsPositions[3] = np.array([-1, -1, 0])
receptorsPositions[4] = np.array([0, 0, 1])



# generate function that simulates the shot emmited
samples = int(sampleRate/2)
rand = (2*np.random.rand(samples) - 1) # Random vector between 1 and -1

# example of non-delayed shot plot
t = np.linspace(0, 0.5, samples)
plt.plot(t, create_shot(t, samples, rand,1))
plt.show()

# example of delayed shot plot
s = signal(rand, 0.1, sampleRate, 1, 0.05)
plt.plot(t, s)
plt.show()

# example of echo removal of the example above
alpha, N = find_echo_coef(s)
s_without_echo = echo_remove(s, alpha, N)
plt.plot(s_without_echo)
plt.show()

# noise control
m = np.array([1, 1, 1, 1, 1])
e = np.array([0, 0, 0, 0, 0])


# results
delays, sounds = delayEstimator (rand,sampleRate, sourcePosition, receptorsPositions, m, e, typeComb = "")
results = MLE(receptorsPositions, delays, sampleRate)
result = MLE_HLS(receptorsPositions, delays, sampleRate)

# HLE(delays, sampleRate, receptorsPositions)
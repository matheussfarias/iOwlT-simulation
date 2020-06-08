#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  6 21:05:38 2020

@author: davi
"""

import numpy as np
import matplotlib.pyplot as plt

def us(x, L=1, fa =1, fc=1):    
    ta = 1/(fa * L)
    alpha = ta/(ta + (1 / (2 * np.pi * fc)))
    
    y_us = np.zeros(x.size * L)
        
    for i in range(1, y_us.size):
        x_us = x[int(i/L)] if i%L == 0 else 0
        y_us[i] = alpha*x_us + (1-alpha)*y_us[i-1]
    return y_us
        
L=10
fc = 5000

fs = 4000 
fa = 16000

t = np.linspace(0, 1, fa)
x = np.sin(2 * np.pi * fs * t)

x_f = abs(np.fft.fft(x))
f = np.fft.fftfreq(t.size, d=(1/fa))

x_us = us(x, L, fa, fc)
#x_us2 = us(x_us, 1, fa*L, fc)

x_us_f = abs(np.fft.fft(x_us))
f_us = np.fft.fftfreq(x_us_f.size, 1/(L*fa))

plt.plot(t, x)
plt.show()

plt.plot(f, x_f)
plt.show()

plt.plot(f_us, x_us_f)
plt.show()
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 15 19:35:02 2022

@author: vikas
"""

import numpy as np
import matplotlib.pyplot as plt

b = np.random.binomial(1, 0.3, size =100)
plt.hist(b)


b = np.random.randint(0,2, size=100)
plt.hist(b)


b = np.random.normal(0,2, size=100)
plt.hist(b)


b = np.random.multinomial(5, [0.1,0.3,0.2,0.3,0.1], size=100)
b

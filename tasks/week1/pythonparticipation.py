#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib
import matplotlib.pyplot as plt
#a function to give yx23x8
def yx23x8(x):
    return x**2 + 3*x + 8


# In[2]:


#defining a function to plot our earlier function of x called yx23x8 against x
def plotyx23x8(x):
    #plots x and y where y is defined in a previous functino to be y= x^2 + 3x + 8
    plt.plot(x,yx23x8(x), color='deeppink')
    #labels the x axis x
    plt.xlabel('x')
    #labels the y axis y = x^2 + 3x + 8
    plt.ylabel('$yx23x8$')
    #shows the plot
    plt.show()
    


# In[ ]:





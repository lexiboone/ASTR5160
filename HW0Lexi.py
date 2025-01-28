#!/usr/bin/env python
# coding: utf-8

# In[27]:


import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import sys

# LB makes a function that takes m, the slope, and b, the intercept of a given line and generates x values at random and the corresponding y values using the formula y = mx + b. Then the y values will be scattered according to gaussians centered on the y values with na standard deviation of 0.5
def straightlinescatter(m,b):
    
    # LB generate random x floating points
    x = np.random.uniform(0,10,10)
    
    # LB recover appropriate y values
    y = m*x + b
    
    # LB scatters points by a random offset drawn from a Gaussian of standard deviation 0.5 
    sigma = 0.5
    error = np.empty(len(x))
    
    # LB fills error with sigma values given
    error.fill(sigma)
    
    #resaves y values according to scattrer
    y = np.random.normal(y,sigma,10)
    
    # LB returns x, y after scattering, and yerror
    return x, y, error

# LB assigning x , y, and y error for a line with a slope and intercept both equal to 1
x,y,yerr = straightlinescatter(1,1)

#defining a function to return a straight line for a given slope m, intercept b, and x values
def straightline(x,m,b):
    
    return m*x + b
    
popt, pcov = curve_fit(straightline,x,y,sigma=yerr)

# LB making figure big
plt.figure(figsize=(17,10))

# LB Plotting data
plt.errorbar(x,y,yerr=yerr, fmt='o', elinewidth=2, color = '#275DAD', label = 'Data')

# LB Plotting original line
plt.plot(x, 1*x + 1, color = '#C490D1', linewidth=2, label='Original Line')

# LB Plotting best fit line
plt.plot(x , popt[0]*x + popt[1],linestyle='dotted',color='#B8336A' ,linewidth=2,label='Best Fitting Line')

# LB adding legend
plt.legend()

# LB making labels
plt.xlabel('x')
plt.ylabel('y')

#LB saving figure
plt.savefig('Techniques2HW0.png')

# LB showing plot
plt.show()

# LB making a function that combines all the previous parts
def totalfunc(m,b):
    #  LB calling previous function to make some scattered data
    x1,y1,yerr1 = straightlinescatter(m,b)

    # LB using curve fit to fit a line to that data
    popt, pcov = curve_fit(straightline,x1,y1,sigma=yerr)
    
    # LB making figure big
    plt.figure(figsize=(17,10))
    
    # LB Plotting data
    plt.errorbar(x1,y1,yerr=yerr, fmt='o', elinewidth=2, color = '#275DAD', label = 'Data')
    
    # LB Plotting original line
    plt.plot(x1, m*x1 + b, color = '#C490D1', linewidth=2, label=f'Original Line (y = {m}x+{b})')
    
    # LB Plotting best fit line
    plt.plot(x1 , popt[0]*x1 + popt[1],linestyle='dotted',color='#B8336A' ,linewidth=2,label=f'Best Fitting Line (y = {popt[0]:.2e}x+{popt[1]:.2e})')
    
    # LB adding legend
    plt.legend()
    
    # LB making labels
    plt.xlabel('x')
    plt.ylabel('y')
    
    #LB saving figure
    plt.savefig('Techniques2HW0.png')
    
    # LB showing plot
    plt.show()
    

if __name__ == "__main__":
    # LB prompt at CL to input slope and intercept and read in as a float
    slope = float(input("m: "))    

    intercept = float(input("b: "))
    
    
    # LB run total function with those inputs
    totalfunc(slope,intercept)



# In[ ]:





# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[45]:


from astropy.table import Table
from astropy.table import QTable
import astropy.units as u
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import sys

# LB reading in struc.fits file
objs = Table.read("struc.fits")

# LB defining RA and Dec to be columns in the file
RA = objs["RA"]
dec = objs["DEC"]

# LB plotting RA and Dec and adding labels
plt.figure(figsize=(17,10))
plt.plot(objs["RA"],objs["DEC"],"bx",label='Objects',markersize=10)
plt.xlabel('Right Ascension')
plt.ylabel('Declination')

# LB defining extinction to be the first column of the 5-array titled extinction
extinction = objs['EXTINCTION'][:,0]

# LB defining a subset of the extinction to find only where extinction is greater than .22
x=np.where(extinction>.22)

# LB creating a subset of RA and Dec where it will only include values wehre extinction is greater than .22
RAsubset = RA[x]
decsubset = dec[x]

#plotting the subset where extinction is greater than .22, adding legend, and showing plot
plt.scatter(RAsubset,decsubset,s=100,marker='x',color='deeppink', label='Extinction greater than .22',zorder=2)
plt.legend()
plt.show()

# LB generating three sets of 100 random numbers from 1-100
x1 = np.random.randint(1,1000,100,dtype=int)
x2 = np.random.randint(1,1000,100,dtype=int)
x3 = np.random.randint(1,1000,100,dtype=int)

# LB stacking together to make 1 3D array
randomnum = np.column_stack((x1,x2,x3))

# LB creating a rec array in table form
t = QTable([RA, dec, randomnum],

           names=('RA','Dec','randomnum'),

           meta={'name': 'first table'})

# LB Writing to a fits file
t.write('week2Lexi.fits',overwrite=True)

# test = Table.read("week2Lexi.fits")
# print(x1)
# print(test['randomnum'][:,0])


# In[ ]:





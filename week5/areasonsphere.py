#!/usr/bin/env python
# coding: utf-8

# In[47]:


from astropy.table import Table
from astropy.table import QTable
import astropy.units as u
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.time import Time
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import sys
import dustmaps
from dustmaps.config import config
from numpy.random import random
import healpy as hp
import os
import math

# LB generating a set of 1000000 random points of ra, dec in degrees that populate the sphere equally in area
ra = 360. * (random(1000000))
dec = (180/np.pi) * np.arcsin(1. - random(1000000)*2.)
nside = 1
pixels = hp.ang2pix(nside, ra, dec, lonlat = True)
print(f'The area is {hp.nside2pixarea(1)} square radians')

# LB finding the positions of which the pixels are equal to some value (in our case, 2, 5, 8)
ii = pixels == 2
it = pixels == 5
iv = pixels == 8

# LB using np.unique with return_counts= True to find the number of points within each HEALPixel
numpoints = np.unique(pixels, return_counts = True)
print(numpoints[1])
print('The areas are close but not exactly equal area')

# LB Plotting the points of ra and dec in pixels 2,5,8 over the total ra and dec
plt.figure(figsize=(16,9))
plt.scatter(ra,dec, marker='*', color='#883689', label ='pixels')
plt.scatter(ra[ii], dec[ii], marker='*', color='#bee3b6', label ='pixels = 2')
plt.scatter(ra[it], dec[it], marker='*', color='#8fd4cb', label ='pixels = 5')
plt.scatter(ra[iv], dec[iv], marker='*', color='pink', label ='pixels = 8')
plt.ylim(min(dec)-10,max(dec)+35)
plt.xlabel('RA (degrees)')
plt.ylabel('Declination (degrees)')
plt.legend(loc=1)
plt.savefig('areasonsphere.png')
plt.show()

# LB using ang2pix with nside = 2 to map the ra,dec to the next level of the hierarchy
nside2 = 2
pixels2 = hp.ang2pix(nside2, ra, dec, lonlat = True)

# LB finding which nside=2 pixels lie inside pixel 5 at nside=1 and only showing the unique ones
# (So it wont show the same pixel twice)
pixinside = pixels2[it]
unique_pixins = np.unique(pixinside) #, return_counts=True)
print(f'The nside=2 pixels lie inside pixel 5 at nside = 1 are {unique_pixins}')


# In[ ]:





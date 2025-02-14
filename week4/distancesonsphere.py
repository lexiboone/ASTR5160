#!/usr/bin/env python
# coding: utf-8

# In[53]:


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
import os
import os
import math

# LB reading in coordinates and representing them as cartesian
ra1, dec1 = 263.75*u.degree, -17.9*u.degree
c1 = SkyCoord(ra1, dec1, frame='icrs')
c2 = SkyCoord(ra = '20 24 59.9', dec = '10 6 0', unit=(u.hourangle,u.deg))
c1.separation(c2)
c1.representation_type = "cartesian"
c2.representation_type = "cartesian"

# LB Manually finding the angle between them using dot product and checking if SkyCoords agrees
mag_c1 = (c1.x**2 + c1.y**2 + c1.z**2)**.5
mag_c2 = (c2.x**2 + c2.y**2 + c2.z**2)**.5
z = math.acos((c2.x * c1.x + c2.y * c1.y + c2.z * c1.z)/(mag_c1 * mag_c2))
print(f'The angle is {z * (180/np.pi)} deg')
print(c1.separation(c2).degree)

# LB mkaing arrays of random values of RA between 2 and 3 and Dec between -2 and 2. Eliza showed me this method
minra = 2
maxra = 3
mindec = -2
maxdec = 2
ra_1 = np.random.random(100) + 2
ra_2 = np.random.random(100) + 2
dec_1 = np.random.random(100) * (maxdec - mindec) - 2
dec_2 = np.random.random(100) * (maxdec - mindec) - 2

# LB Plotting both in different colors/markers
plt.figure(figsize=(16,9))
plt.scatter(ra_1, dec_1, color='deeppink', label='coordinates 1')
plt.scatter(ra_2, dec_2, marker='x', color='forestgreen', label='coordinates 2')
plt.xlabel('RA (hourangle)')
plt.ylabel('Dec (degrees)')
plt.legend()
plt.show()

# LB finding where the points are less than 10' of each other and converting to degrees
c_1 = SkyCoord(ra = ra_1, dec = dec_1, unit=(u.hourangle,u.deg))
c_2 = SkyCoord(ra = ra_2, dec = dec_2, unit=(u.hourangle,u.deg))
id1, id2, d2, d3 = c_2.search_around_sky(c_1, (10/60)*u.deg)

# LB plotting on same plot as the RA and Dec
plt.figure(figsize=(16,9))
plt.scatter(ra_1, dec_1, color='deeppink', label='coordinates 1')
plt.scatter(ra_2, dec_2, marker='x', color='forestgreen', label='coordinates 2')
plt.scatter(ra_1[id1],dec_1[id1],color='blue', label="points within 10'")
plt.scatter(ra_2[id2],dec_2[id2],color='blue')
plt.xlabel('RA (hourangle)')
plt.ylabel('Dec (degrees)')
plt.legend()
plt.show()

# LB combining both RA and Dec into one array
raboth = np.concatenate([ra_1, ra_2])
decboth = np.concatenate([dec_1, dec_2])

# LB Plotting both sets on one plot
plt.figure(figsize=(16,9))
plt.scatter(raboth,decboth,color='dodgerblue',label='Both sets')
plt.xlabel('RA (hourangle)')
plt.ylabel('Dec (degrees)')
plt.legend()
plt.show()

# LB  finding all of the points that will fall on your plate and plot this subset of points
center = SkyCoord(ra = '2 20 5', dec = '-0 6 12', unit=(u.hourangle,u.deg))
cboth = SkyCoord(ra = raboth*u.hourangle, dec = decboth*u.degree)
ii = center.separation(cboth) < 1.8*u.degree
plt.figure(figsize=(16,9))
plt.scatter(raboth,decboth,color='dodgerblue',label='Both sets')
plt.scatter(raboth[ii],decboth[ii],color='orangered',label='Coordinates in Plate')

plt.xlabel('RA (hourangle)')
plt.ylabel('Dec (degrees)')
plt.legend()
plt.show()


# In[ ]:





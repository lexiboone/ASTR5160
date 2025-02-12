#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[41]:


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

# LB defining Dust enivronment
os.environ['DUSTDIR']= '/d/scratch/ASTR5160/data/dust/v0_1/maps'

# LB Generating a random set of 10,000 points of RA and DEC in radians
ra = 2 * np.pi * (random(10000) - 0.5)
dec = np.arcsin(1 - random(10000) * 2.)

#LB Plotting them on a standard x, y grid
plt.figure(figsize=(16, 9))
plt.scatter(ra, dec, s=3, color='firebrick')
plt.ylabel('Dec (Radians)')
plt.xlabel('RA (Radians)')
plt.show()

# LB Plotting in Aitoff projection, adding a grid, and labeling x-axis in degrees
fig = plt.figure(figsize=(16, 9))
ax = fig.add_subplot(111, projection="aitoff")
ax.scatter(ra, dec, marker='o', color='magenta', s=0.7, alpha=0.5)
xlab = ['14h', '16h', '18h', '20h', '22h', '0h', '2h', '4h', '6h', '8h', '10h']
ax.set_xticklabels(xlab, weight=800)
ax.grid(color='b', linestyle='dashed', linewidth=2)

# LB Plotting in Lambert projection and changing x-axis from hours to degrees
fig = plt.figure(figsize=(16, 9))
ax = fig.add_subplot(111, projection="lambert")
ax.scatter(ra, dec, marker='o', color='magenta', s=0.7, alpha=0.5)
xlab = ['14h', '16h', '18h', '20h', '22h', '0h', '2h', '4h', '6h', '8h', '10h']
ax.set_xticklabels(xlab, weight=800)
ax.grid(color='b', linestyle='dashed', linewidth=2)

# LB Reading in dust directory configuration
dustdir = os.getenv("DUSTDIR")
if dustdir is None:
    print("Set the DUSTDIR environment variable to point to the main " +
    "DUSTDIR data directory (e.g. /d/scratch/ASTR5160/data/dust/v0_1/maps)")
config["data_dir"] = dustdir #"/d/scratch/ASTR5160/data/dust/v0_1/maps"
from dustmaps.sfd import SFDQuery
sfd = SFDQuery()

# LB defining RA and Dec in Degrees from 0 to 360 and -90 to 90, respectively
ra_dust = np.arange(0.5, 359.5, 1)
dec_dust = np.arange(-89.5, 89.5, 1)

# LB meshing RA and Dec together
x1, y1 = np.meshgrid(ra_dust, dec_dust)

# LB reading coordinate points in to SkyCoords
c = SkyCoord(ra = x1 * u.deg, dec = y1 * u.deg)#.galactic

# LB finding reddening at each point
ebv = sfd(c)

# LB converting ra and dec to x,y in Aitoff
from astropy import wcs
w = wcs.WCS(naxis=2)
w.wcs.ctype = ["RA---AIT", "DEC--AIT"]
x, y = w.wcs_world2pix(x1, y1, 1)

# LB Plotting Ra and Dec in Aitoff projection
fig = plt.figure(figsize=(16, 9))
ax = fig.add_subplot(111, projection="aitoff")
plt.scatter(x1,y1,s=0.7)
xlab = ['14h', '16h', '18h', '20h', '22h', '0h', '2h', '4h', '6h', '8h', '10h']
ax.set_xticklabels(xlab, weight=800)
ax.grid(color='b', linestyle='dashed', linewidth=2)
plt.show()

# LB Plotting x,y in Aitoff projection
fig = plt.figure(figsize=(16, 9))
ax = fig.add_subplot(111, projection="aitoff")
plt.scatter(x,y,s=0.7)
xlab = ['14h', '16h', '18h', '20h', '22h', '0h', '2h', '4h', '6h', '8h', '10h']
ax.set_xticklabels(xlab, weight=800)
ax.grid(color='b', linestyle='dashed', linewidth=2)
plt.show()

# LB making contour plot of dust reddening
fig = plt.figure(figsize=(17,10))
ax = fig.add_subplot(111, projection="aitoff")
plt.contourf(x1, y1, ebv, levels=100, cmap='magma')
plt.show()



# In[ ]:





# In[ ]:





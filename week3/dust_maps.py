#!/usr/bin/env python
# coding: utf-8

# In[16]:


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

# LB testing dustmaps
ugriz = np.array([4.239 ,3.303 , 2.285 , 1.698 , 1.263])
ra, dec = '12h42m30s', '+41d12m00s'
c = SkyCoord(ra, dec).galactic
dustdir = 'dustmaps/'
config["data_dir"] = dustdir
from dustmaps.sfd import SFDQuery
sfd = SFDQuery()
sfd(c)

# LB inputting info about Quasar 1
quasar1 = SkyCoord(246.933 * u.deg, 40.795 * u.deg).galactic
g1 = 18.81
r1 = 18.73
i1 = 18.82
gr1 = g1 - r1
ri1 = r1 - i1

#LB calculating extinction
ebv1 = sfd(quasar1)
A1 = ebv1*ugriz
gr1c = A1[1] - A1[2]
ri1c = A1[2] - A1[3]

# LB inputting info about quasar #2, RA, Dec, colors
quasar2 = SkyCoord(236.562 * u.deg, 2.440 * u.deg).galactic
g2 = 19.10
r2 = 18.79
i2 = 18.73
gr2 = g2 - r2
ri2 = r2 - i2

# LB calculating extinction
ebv2 = sfd(quasar2)
A2 = ebv2 * ugriz
gr2c = A2[1] - A2[2]
ri2c = A2[2] - A2[3]

# LB plotting g-r and r-i for the two quasars
plt.figure(figsize=(17,10))
plt.scatter(gr1, ri1, label='Quasar 1', color='deeppink')
plt.scatter(gr2, ri2, label='Quasar 2', color='blue')
plt.xlabel('g - r', fontsize=17)
plt.ylabel('r - i', fontsize=17)
plt.legend(fontsize=14)
plt.show()

# LB plotting the corrected colors in the same plot
plt.figure(figsize=(17,10))
plt.scatter(gr1c, ri1c, label='Corrected Quasar 1', color='deeppink')
plt.scatter(gr2c, ri2c, label='Corrected Quasar 2', color='blue')
plt.xlabel('g - r', fontsize=17)
plt.ylabel('r - i', fontsize=17)
plt.legend(fontsize=14)
plt.show()

#The colors agree better

# LB making arrays of size 100 centered around RA and Dec of Quasar 1 with bins of 0.1
RA1 = np.arange(236.6 - 5, 236.6 + 5, 0.1)
Dec1 = np.arange(2.4 - 5, 2.4 + 5, 0.1)

# LB making arrays of size 100 centered around RA and Dec of Quasar 1 with bins of 0.13 and 0.1, respectively
RA2 = np.arange(246.9 - (50 * 0.13), 246.9 + (50 * 0.13), 0.13)
Dec2 = np.arange(40.8 - (50 * 0.1), 40.8 + (50 * 0.1), 0.1)

# LB making a mesh grid of RA and Dec for both pairs 
x1,y1 = np.meshgrid(RA1,Dec1)
x2,y2 = np.meshgrid(RA2,Dec2)

# LB finding the amount of reddening E(B-V) for each pair of positions
c1 = SkyCoord(ra = x1 * u.deg, dec = y1 * u.deg)#.galactic
c2 = SkyCoord(ra = x2 * u.deg, dec = y2 * u.deg)#.galactic
ebvpair1 = sfd(c1)
ebvpair2 = sfd(c2)

# LB making a contour plot showing reddening at each RA and Dec of Quasar 1 and plotting Quasar
plt.figure(figsize=(17,10))
plt.contourf(RA1, Dec1, ebvpair1)
plt.scatter(236.562, 2.440, label='Quasar 2', color='white')
plt.colorbar()
plt.show()

# LB making a contour plot showing reddening at each RA and Dec of Quasar 2 and plotting the Quasar
plt.figure(figsize=(17,10))
plt.contourf(RA2, Dec2, ebvpair2)
plt.scatter(246.933, 40.795, label='Quasar 1', color='white')
plt.colorbar()
plt.show()


# In[ ]:





# In[ ]:





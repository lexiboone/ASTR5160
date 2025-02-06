#!/usr/bin/env python
# coding: utf-8

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

# LB defining an RA and Dec in hms and (deg,',"), respectively (I picked coordinates of TIC 119178132)
c = SkyCoord(ra = '14 03 06.05', dec = '-1 28 49.02', unit=(u.hourangle,u.deg))

# LB finding RA and Dec and multiplying by pi/180 to tranform to radians
alpha = c.ra.deg*(np.pi/180)
delta = c.dec.deg*(np.pi/180)

# LB using SkyCoord to transform to cartesian coordinates
c.representation_type = "cartesian"
print(c)

# LB using equations to check if they agree with the SkyCoord output
x = np.cos(alpha)*np.cos(delta)
y = np.sin(alpha)*np.cos(delta)
z = np.sin(delta)
print(x,y,z)

# LB defining galactic center using galactic coordinates
galcenter = SkyCoord(0,0, frame='galactic', unit='deg')
galcenter_icrs = galcenter.transform_to('icrs')
print(galcenter_icrs)
print(galcenter_icrs.ra.hms)

# The Galactic Center is in Sagittarius (in Sagittarius A). It is near the edge of the constellation (off the top of the spout of the "teapot" aterism in Sagittarius)

# LB making array of RA that spans the year and defining coordinates with these RA values and a declination of 40 degrees
# LB Dec of 40 because a star at zenith in Laramie will always be at 40 deg dec
ra_array = np.arange(0,360,(360/365.25))
laramie = SkyCoord(ra_array,40, unit='deg')

# LB tranforming to galactic coordinates and defining l and b as the output of these coordinates
laramie_gal = laramie.transform_to('galactic')
l = laramie_gal.l.degree
b = laramie_gal.b.degree

# LB making an array for every day in a year
days = np.arange(0,365.25,1)

# LB plotting (l,b) as a function of days
fig = plt.figure(figsize=(17,10))
ax = fig.add_subplot(projection='3d')
ax.scatter(l,b,days, marker='x',color='deeppink')
ax.xaxis.pane.fill = False
ax.yaxis.pane.fill = False
ax.zaxis.pane.fill = False
ax.set_xlabel('l (degrees)')
ax.set_ylabel('b (degrees)')
ax.set_zlabel('days', rotation=90)
ax.set_box_aspect(None, zoom=0.9)
plt.show()

# LB also plotting 2d (l,b) over the year
plt.figure(figsize=(17,10))
sc = plt.scatter(l, b, c=days, vmin=0, vmax=365.25, cmap='gist_rainbow')
plt.colorbar(sc,label='days')
plt.xlabel('l (degrees)')
plt.ylabel('b (degrees)')
#plt.legend()
plt.show()

# LB defining coordinates of the Sun, the moon, and the planets
sun = SkyCoord(ra = '21 9 52', dec = '-16 20 0', unit = (u.hourangle,u.deg), distance = 0.9859239*u.AU)
moon = SkyCoord(ra = '1 37 22', dec = '12 13 31', unit= (u.hourangle,u.deg), distance = 370369 * u.km)
mercury = SkyCoord(ra = '20 56 6.7', dec = '-19 27 40', unit = (u.hourangle,u.deg), distance = 1.409  *u.AU)
venus = SkyCoord(ra = '23 53 51.5', dec = '1 55 46', unit = (u.hourangle,u.deg), distance = 0.495 * u.AU)
mars = SkyCoord(ra = '7 26 10.4', dec = '26 12 52', unit = (u.hourangle,u.deg), distance = 0.701 * u.AU)
jupiter = SkyCoord(ra = '4 37 41.4', dec = '21 36 8', unit = (u.hourangle,u.deg), distance = 4.598 * u.AU)
saturn = SkyCoord(ra = '23 16 39.1', dec = '-6 44 51', unit = (u.hourangle,u.deg), distance = 10.445 * u.AU)
uranus = SkyCoord(ra = '3 22 20.1', dec = '18 15 58', unit = (u.hourangle,u.deg), distance = 19.403 * u.AU)
neptune = SkyCoord(ra = '23 53 36.1', dec = '-2 4 33', unit = (u.hourangle,u.deg), distance = 30.621 * u.AU)
pluto = SkyCoord(ra = '20 19 24.7', dec = '-22 59 18', unit = (u.hourangle,u.deg), distance = 36.149 * u.AU)

# LB defining an object list and an object name list and colors that correspond
obs_list = [sun,moon,mercury,venus,mars,jupiter,saturn,uranus,neptune,pluto]
obs_names = ['sun','moon','mercury','venus','mars','jupiter','saturn','uranus','neptune','pluto'] #justice for pluto
color_list = ['gold','darkgrey','midnightblue','darkorange','firebrick','sandybrown','burlywood','darkcyan','steelblue','black']

# LB making a figure to put my points
plt.figure(figsize=(17,10))

# LB converting every object in the lost to heliocentrictrueecliptic
for i in range(len(obs_list)):
    helio_coords = obs_list[i].transform_to('heliocentrictrueecliptic')
    lon = helio_coords.lon.degree
    lat = helio_coords.lat.degree
    plt.scatter(lon,lat,color=color_list[i],label=obs_names[i],s=70)
    plt.xlabel('Longitude (degrees)')
    plt.ylabel('Latitude (degrees)')
    plt.legend()
    print(helio_coords)

plt.show()


# In[ ]:





# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[64]:


from astropy.table import Table
from astropy.table import QTable
import astropy.units as u
import astropy
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.time import Time
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import sys

# LB defining an RA and Dec in hms and (deg,',"), respectively (I picked coordinates of TIC 119178132)
c = SkyCoord(ra = '14 03 06.05', dec = '-1 28 49.02', unit=(u.hourangle,u.deg))

#LB converting the RA and Dec to degrees and printing so we can check against our manual calculation
ra_degrees = c.ra.deg
dec_degrees = c.dec.deg

print(f'The astropy converted RA is {ra_degrees} degrees')
print(f'The astropy converted Declination is {dec_degrees} degrees')

# LB manually checking using the given formula and printing the output
ra_manual = 15 * (14 + (3/60)) + (6.05/3600)
dec_manual = -1 * (1 + (28/60)) + (49.02/3600)
print(f'The manually calculated RA is {ra_manual} degrees')
print(f'The manually calculated Declination is {dec_manual} degrees') 

# LB defining t to be the time now and using t.mjd and t.jd to convert the time into mjd and jd
t = Time.now()
MJD = t.mjd
JD = t.jd

# LB checking that MJD and JD are related using the formula we were given (should equal 2400000.5)
check = JD-MJD
print(f'The MJD/JD offset is {check}')

# LB making an array of values ranging from 100 days before and after the MJD today in steps of 1
MJDdates = np.arange(MJD - 100, MJD + 100, 1)
print(f"Some dates near today's MJD are {MJDdates}")

# LB converting our longitude is 105°58'33''W, its latitude is 41°5'49''N into degrees.

long_manual = -1 * (105 + (58/60)) + (33/3600)
lat_manual = 1 * (41 + (5/60)) + (49/3600)

# LB setting the location of WIRO using our calculated longitude and latitude and its altitude.
WIRO = EarthLocation(lon = long_manual * u.deg, lat = lat_manual * u.deg, height=2943 * u.m)

# LB defining a star with equatorial coorindates
star = SkyCoord(ra = '12 0 0', dec = '30 0 0', unit=(u.hourangle,u.deg))

# LB conversion from UTC to Mountain Standard Time
utcoffset = -7 * u.hour  

# LB finding the time right now in MST using the offset we found previously 
Datenow = str(Time.now()-utcoffset)

# LB Changing the time to be 11pm at that date
elevenpmmst = (Datenow[:10]) + ' 23:00:00'

# LB changing time from strong back to astropy Time
elevenpmmstfloat = astropy.time.core.Time(elevenpmmst)


# LB finding alt,az coordinates of star at WIRO tonight at 11 pm
staraltaz = star.transform_to(AltAz(obstime=elevenpmmstfloat, location=WIRO))

# LB converting to airmass
starairmasss_11pmtonight = staraltaz.secz

# LB printing airmass
print(f'The airmass at 11pm tonight is {starairmasss_11pmtonight}')

# LB converted 30 days to hours to calculate offset a month from now
monthoff = (720) * u.hour

# LB adding the month to todays date and subtracting the UTC offset to get MST
nextmonth = Time.now() + monthoff - utcoffset

# LB converting date to string
Datemonthaway = str(nextmonth)

# LB changing the time to be 11pm on that night 
elevenmonthaway = (Datemonthaway[:10]) + ' 23:00:00'

# LB converting back to the astropy time format
elevenmonthawayfloat = astropy.time.core.Time(elevenmonthaway)

# LB Finding the alt,az coordinates a month from now from WIRO
staraltazmonthaway = star.transform_to(AltAz(obstime=elevenmonthawayfloat, location=WIRO))

# LB converting to airmass
starairmasss_11pmmonthaway = staraltazmonthaway.secz

# LB printing airmass
print(f'The airmass at 11pm tonight is {starairmasss_11pmmonthaway}')


# In[ ]:





# In[ ]:





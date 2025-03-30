#!/usr/bin/env python
# coding: utf-8

# In[1]:


from astropy.table import Table, vstack
from astropy.table import QTable
import astropy.units as u
from astropy.coordinates import SkyCoord, EarthLocation, AltAz
from astropy.time import Time
from astropy.coordinates import SkyCoord
import numpy as np
import random
import astropy.units as u
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import sys
import os
import glob
 
# LB using Dr. Myers DESI code and editing for this use
def decode_sweep_name(sweepname, nside=None, inclusive=True, fact=4):
    """Retrieve RA/Dec edges from a full directory path to a sweep file

    Parameters
    ----------
    sweepname : :class:`str`
        Full path to a sweep file, e.g., /a/b/c/sweep-350m005-360p005.fits
    nside : :class:`int`, optional, defaults to None
        (NESTED) HEALPixel nside
    inclusive : :class:`book`, optional, defaults to ``True``
        see documentation for `healpy.query_polygon()`
    fact : :class:`int`, optional defaults to 4
        see documentation for `healpy.query_polygon()`

    Returns
    -------
    :class:`list` (if nside is None)
        A 4-entry list of the edges of the region covered by the sweeps file
        in the form [RAmin, RAmax, DECmin, DECmax]
        For the above example this would be [350., 360., -5., 5.]
    :class:`list` (if nside is not None)
        A list of HEALPixels that touch the  files at the passed `nside`
        For the above example this would be [16, 17, 18, 19]
    """
    # ADM extract just the file part of the name.
    sweepname = os.path.basename(sweepname)

    # ADM the RA/Dec edges.
    ramin, ramax = float(sweepname[6:9]), float(sweepname[14:17])
    decmin, decmax = float(sweepname[10:13]), float(sweepname[18:21])

    # ADM flip the signs on the DECs, if needed.
    if sweepname[9] == 'm':
        decmin *= -1
    if sweepname[17] == 'm':
        decmax *= -1

    if nside is None:
        return [ramin, ramax, decmin, decmax]

    pixnum = hp_in_box(nside, [ramin, ramax, decmin, decmax],
                       inclusive=inclusive, fact=fact)

    return pixnum

def is_in_box(objs, radecbox):
    """Determine which of an array of objects are inside an RA, Dec box.

    Parameters
    ----------
    objs : :class:`~numpy.ndarray`
        An array of objects. Must include at least the columns "RA" and "DEC".
    radecbox : :class:`list`
        4-entry list of coordinates [ramin, ramax, decmin, decmax] forming the
        edges of a box in RA/Dec (degrees).

    Returns
    -------
    :class:`~numpy.ndarray`
        ``True`` for objects in the box, ``False`` for objects outside of the box.

    Notes
    -----
        - Tests the minimum RA/Dec with >= and the maximum with <
    """
    ramin, ramax, decmin, decmax = radecbox

    # ADM check for some common mistakes.
    if decmin < -90. or decmax > 90. or decmax <= decmin or ramax <= ramin:
        msg = "Strange input: [ramin, ramax, decmin, decmax] = {}".format(radecbox)
        log.critical(msg)
        raise ValueError(msg)
        
    ii = ((objs[:,0] >= ramin) & (objs[:,0] <= ramax)
          & (objs[:,1] >= decmin) & (objs[:,1] <= decmax))
    #print(objs[0])

    return ii

# LB seeing which files have any objects fall within it. It will save them into sweepfiles.
#sweepfiles = []
def findsweep(objs,files):
    sweepfiles = []
    for i in files:
        ramin, ramax, decmin, decmax = decode_sweep_name(i)
        radecbox1 = ramin, ramax, decmin, decmax 
        
        if np.any(is_in_box(objs,radecbox1)) == True:
            #print(i)
            #return i
            sweepfiles.append(i)
    return sweepfiles
            
            
# LB setting the directory of the stars/quasars and reading in the files using astropy Table
os.environ['STARDIR']= '/d/scratch/ASTR5160/week10/'
stardir = os.getenv("STARDIR")
file1 = os.path.join(stardir,'stars-ra180-dec30-rad3.fits')
file2 = os.path.join(stardir,'qsos-ra180-dec30-rad3.fits')
#print(file2)
objs1 = Table.read(file1, unit_parse_strict='silent')
objs2 = Table.read(file2, unit_parse_strict='silent')
#print(objs1)
# LB calling the RA and dec of the stars and quasars and making one array with both
RA1 = objs1["RA"]
Dec1 = objs1["DEC"]
#print(RA1)
both1 = np.column_stack((RA1,Dec1))
RA2 = objs2["RA"]
Dec2 = objs2["DEC"]
both2 = np.column_stack((RA2,Dec2))

# LB reading in those coordinates to skycoord
obj1 = SkyCoord(RA1, Dec1, unit=(u.degree, u.degree))
obj2 = SkyCoord(RA2, Dec2, unit=(u.degree, u.degree))

os.environ['LEGACYDIR']= '/d/scratch/ASTR5160/data/legacysurvey/dr9/south/sweep/9.0/'
legacydir = os.getenv("LEGACYDIR")
#print(legacydir)
# print(both1)

#print(both1[:,0])

# LB finding all the sweep file names in the directory
user_dir = '/d/scratch/ASTR5160/data/legacysurvey/dr9/south/sweep/9.0/'
file_list = []
for dirpath,_,filenames in  os.walk(user_dir):
    for name in filenames:
        if name.endswith('.fits'):
            file_list.append(os.path.join(user_dir,dirpath,name))
#print(file_list) 

# LB running function to find which sweep files have the star and quasar coordinates
stars = findsweep(both1, file_list)

quasars = findsweep(both2, file_list)



# In[2]:


# LB stacking all the 4 sweep files (same for both so only used the output for the star) and making them skycoords
totfiles = vstack([Table.read(file, unit_parse_strict = 'silent') for file in stars])

totcoordinates = SkyCoord(totfiles["RA"], totfiles["DEC"], unit=(u.degree, u.degree))

# LB finding objects only .5 arcescs away  
i1 = obj1.search_around_sky(totcoordinates, u.arcsecond * 0.5)[0]
i2 = obj2.search_around_sky(totcoordinates, u.arcsecond * 0.5)[0]

# LB masking these values and finding flux for both the star and the quasar
closestars = totfiles[i1]


G1 = closestars["FLUX_G"]
R1 = closestars["FLUX_R"]
Z1 = closestars["FLUX_Z"]
W11 = closestars["FLUX_W1"]
W21 = closestars["FLUX_W2"]

MWG1 =  closestars["MW_TRANSMISSION_G"]
MWR1 = closestars["MW_TRANSMISSION_R"]
MWZ1 = closestars["MW_TRANSMISSION_Z"]
MWW11 = closestars["MW_TRANSMISSION_W1"]
MWW21 =  closestars["MW_TRANSMISSION_W2"]


closequasars = totfiles[i2]

G2 = closequasars["FLUX_G"]
R2 = closequasars["FLUX_R"]
Z2 = closequasars["FLUX_Z"]
W12 = closequasars["FLUX_W1"]
W22 = closequasars["FLUX_W2"]

MWG2 = closequasars["MW_TRANSMISSION_G"]
MWR2 = closequasars["MW_TRANSMISSION_R"]
MWZ2 = closequasars["MW_TRANSMISSION_Z"]
MWW12 = closequasars["MW_TRANSMISSION_W1"]
MWW22 = closequasars["MW_TRANSMISSION_W2"]


# 


# In[13]:


# LB correcting for dust
corrected_g1 = G1 / MWG1
corrected_r1 = R1 / MWR1
corrected_z1 = Z1 / MWZ1
corrected_W11 = W11 / MWW11
corrected_W21 = W21 / MWW21

corrected_g2 = G2 / MWG2
corrected_r2 = R2 / MWR2
corrected_z2 = Z2 / MWZ2
corrected_W12 = W12 / MWW12
corrected_W22 = W22 / MWW22

print(len(totfiles))


# In[14]:


# LB calculating magnitudes
gmag_star1 = 22.5 - 2.5 * np.log10(corrected_g1)
rmag_star1 = 22.5 - 2.5 * np.log10(corrected_r1)
zmag_star1 = 22.5 - 2.5 * np.log10(corrected_z1)
W1mag_star1 = 22.5 - 2.5 * np.log10(corrected_W11)
W2mag_star1 = 22.5 - 2.5 * np.log10(corrected_W21)

gmag_quasar1 = 22.5 - 2.5 * np.log10(corrected_g2)
rmag_quasar1 = 22.5 - 2.5 * np.log10(corrected_r2)
zmag_quasar1 = 22.5 - 2.5 * np.log10(corrected_z2)
W1mag_quasar1 = 22.5 - 2.5 * np.log10(corrected_W12)
W2mag_quasar1 = 22.5 - 2.5 * np.log10(corrected_W22)


print('Star mags')
print(f'The g mag is {gmag_star1}')
print(f'The r mag is {rmag_star1}')
print(f'The z mag is {zmag_star1}')
print(f'The W1 mag is {W1mag_star1}')
print(f'The W2 mag is {W2mag_star1}')


print('         ')
print('Quasar mags')
print(f'The g mag is {gmag_quasar1}')
print(f'The r mag is {rmag_quasar1}')
print(f'The z mag is {zmag_quasar1}')
print(f'The W1 mag is {W1mag_quasar1}')
print(f'The W2 mag is {W2mag_quasar1}')


# In[16]:


# LB finding the r - W1 and g -z for the stars and the quasars
star_r_W1 = rmag_star1 - W1mag_star1
star_g_z = gmag_star1 - zmag_star1

quasar_r_W1 = rmag_quasar1 - W1mag_quasar1
quasar_g_z = gmag_quasar1 - zmag_quasar1


# In[43]:


# Lb finding approximate values for a line that will divide the quasars and stars. Plotting to visually see if this is a good choice.
b = -1
m = 1
x = np.linspace(-2, 7, 10000)
y = m * x + b
plt.figure(figsize=(16,9))
plt.scatter(star_g_z, star_r_W1, marker='*', c='darkviolet', label='Stars')
plt.scatter(quasar_g_z, quasar_r_W1, marker='d', c='dodgerblue', label='Quasars')
plt.plot(x,y)
plt.xlabel('g - z')
plt.ylabel('r - W1')
plt.legend()
plt.show()
 
# Lb modifying one of Eliza's functions to determine if this is a star or not
def isitstar(g_z, r_W1):
    """
    Parameters:
    ------------
    g_z (np.array) - g - z magnitude of object
    r_W1 (np.array) - r - W1 magnitude of object

    Returns:
    ---------
    (boolean) - True if star, False if quasar

    Note:
    -----
    This is not perfect, there are some outliers on both sides.
    However, this does account for a majority of the stars and quasar
    classifications.
    """

    b = -1
    m = 1
    x = np.linspace(-2, 7, 10000)
    divide = m * x + b

    if r_W1 > divide.all() and g_z < x.all():
        return 'Not a star'
    else: 
        return 'Is a star'
    
# LB testing with random values of out colors
randomind = np.random.randint(len(quasar_r_W1))
print(isitstar(quasar_g_z[randomind], quasar_r_W1[randomind]))
print(isitstar(star_g_z[randomind], star_r_W1[randomind]))
# Lb find out that sometimes it doesn't work :( 


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





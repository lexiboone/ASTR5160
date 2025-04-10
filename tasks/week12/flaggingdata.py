#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# In[1]:


import glob
import numpy as np
import os
from astropy.coordinates import SkyCoord
from astropy.table import Table, vstack
import astropy.units as u

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
        
    ii = ((objs['RA'] >= ramin) & (objs['RA'] <= ramax)
          & (objs['DEC'] >= decmin) & (objs['DEC'] <= decmax))
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
def mag_nanomaggies(flux):
    #LB takes flux in nanomaggy and returns magnitude
    return 22.5 - 2.5 * np.log10(flux)

    
# In[4]:

if __name__ == "__main__":
    # LB reading in sweep files from last week and stacking them into one big file and putting into SkyCoord
    sweepfilenames = ['/d/scratch/ASTR5160/data/legacysurvey/dr9/south/sweep/9.0/sweep-170p025-180p030.fits', '/d/scratch/ASTR5160/data/legacysurvey/dr9/south/sweep/9.0/sweep-170p030-180p035.fits', '/d/scratch/ASTR5160/data/legacysurvey/dr9/south/sweep/9.0/sweep-180p025-190p030.fits', '/d/scratch/ASTR5160/data/legacysurvey/dr9/south/sweep/9.0/sweep-180p030-190p035.fits']
    objs1 = vstack([Table.read(file, unit_parse_strict = 'silent') for file in sweepfilenames])
    totcoordinates = SkyCoord(objs1["RA"], objs1["DEC"], unit=(u.degree, u.degree))
    RA1 = objs1["RA"]
    Dec1 = objs1["DEC"]
    # LB making dictionary with the object given and finding the corresponding sweep file
    location = Table({"RA": [188.53667], "DEC" : [21.04572]})
    totsweepfiles = glob.glob('/d/scratch/ASTR5160/data/legacysurvey/dr9/south/sweep/9.0/*.fits')
    #closefile = '/d/scratch/ASTR5160/data/legacysurvey/dr9/south/sweep/9.0/sweep-180p020-190p025.fits'
    closefile = findsweep(location,totsweepfiles)
    # LB for troubleshooting uncomment
    #print(closefile)
    # LB reading in the file and putting coordinates into SkyCoord
    openfile = Table.read(closefile[0], unit_parse_strict = 'silent')
    objectskycoord = SkyCoord(openfile["RA"], openfile["DEC"], unit=(u.degree, u.degree))
    #location = Table({"RA": [188.53667], "DEC" : [21.04572]})
    # LB finding minimum separation between the object and object in the sweepfile
    givenobject = SkyCoord('188.53667', '21.04572', unit=(u.degree, u.degree))
    #sweep_objects = SkyCoord(RA1, Dec1, unit=(u.degree, u.degree))
    separations = givenobject.separation(objectskycoord)
    ix = separations.argmin()

    # LB indexing the values we want according to the closes object
    objtype = openfile["TYPE"][ix]
    #print(openfile[ix])
    #print(openfile)
    print('Object is exponential')
    bitmaskg = openfile["ALLMASK_G"][ix]
    bitmaskr = openfile["ALLMASK_R"][ix]
    bitmaskz = openfile["ALLMASK_Z"][ix]
    
    print(bitmaskg)
    print(bitmaskr)
    print(bitmaskz)
    print('All of the bands are saturated in all exposures. In the SkyViewer, it does look saturated.')
    #print(len(objs1))
    
    #psfsc = SkyCoord(onlypsf['RA'], onlypsf['DEC'], unit=(u.degree, u.degree))
    # LB reading in the coordinates given in the next task and finding objects within 3 degrees
    c2 = SkyCoord('180.0', '30.0', unit=(u.degree, u.degree))
    ind = (totcoordinates.separation(c2) < 3*u.degree)
    nearbyobjs = objs1[ind]
    # LB finding only objects with r < 20 and PSF type
    rflux = nearbyobjs["FLUX_R"]
    rmag = mag_nanomaggies(rflux)
    
    allind = (nearbyobjs["TYPE"] == 'PSF') & (rmag < 20)
    print(nearbyobjs[allind])
    fitsfile = glob.glob('/d/scratch/ASTR5160/week10/qsos-ra180-dec30-rad3.fits')[0]
    print(f'There are {len(nearbyobjs[allind])} r < 20, TYPE == PSF objects')
    # LB finding objects in the quasar file that match using a .5" matchign radius
    qsosfile = Table.read(fitsfile, unit_parse_strict = 'silent')
    qsoscoordinates = SkyCoord(qsosfile["RA"], qsosfile["DEC"], unit=(u.degree, u.degree))
    
    restrictedobjs = nearbyobjs[allind]
    restricted_sc = SkyCoord(restrictedobjs["RA"], restrictedobjs["DEC"], unit=(u.degree, u.degree))
    id1, id2, d2, d3 = qsoscoordinates.search_around_sky(restricted_sc, 0.5*u.arcsecond)
    print(qsosfile[id2])
    print(f'There are {len(qsosfile[id2])} objects that are quasars')


# In[ ]:





# In[ ]:





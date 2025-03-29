from astropy.table import Table
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


os.environ['LEGACYDIR']= '/d/scratch/ASTR5160/data/legacysurvey/dr9/south/sweep/9.0'
legacydir = os.getenv("LEGACYDIR")
print(legacydir)

# LB finding all the sweep file names in the directory
user_dir = '/d/scratch/ASTR5160/data/legacysurvey/dr9/south/sweep/9.0'
file_list = []
for dirpath,_,filenames in  os.walk(user_dir):
    for name in filenames:
        if name.endswith('.fits'):
            file_list.append(os.path.join(user_dir,dirpath,name))

# LB finding UBVRI to ugriz
B_V =  0.873
V = 15.256
U_B =  0.320
V_R =  0.505
R_I =  0.511

g =  V + 0.60*(B_V) - 0.12
r = V - 0.42*(B_V) + 0.11
uu = g + 1.28*(U_B)   + 1.13
i =  -1 * (0.91*(R_I) - 0.20 -r)
z = -1 * (1.72*(R_I) - 0.41 - r)
print(f'The ugriz is {uu} {g} {r} {i} {z}')
print('This is close to the values in SDSS, but z differs the most')


# LB making dictionary of the coordinates of PG1633_099A
bothh = Table({"RA": [248.85833], "DEC" : [9.79806]})
#print(file_list)
# LB using DESI code and editing for this use
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

    ii = ((objs["RA"] >= ramin) & (objs["RA"] < ramax)
          & (objs["DEC"] >= decmin) & (objs["DEC"] < decmax))

    return ii

# LB seeing which files have any objects fall within it. It will print them and save them into sweepfiles.
sweepfiles = []
def findsweep(objs,files):
    for i in files:
        ramin, ramax, decmin, decmax = decode_sweep_name(i)
        radecbox1 = ramin, ramax, decmin, decmax 
        
        if np.any(is_in_box(objs,radecbox1)) == True:
            print(i)
            sweepfiles.append(i)
     
    
# LB reading in the file 
findsweep(bothh,file_list)
objs1 = Table.read(sweepfiles[0], format='fits', unit_parse_strict='silent')
#print(objs1)
RA1 = objs1["RA"]
Dec1 = objs1["DEC"]

# LB finding the index of the object in the sweep file, Eliza and Josh showed me this
PG1633_099A = SkyCoord('16:35:26', '+09:47:53', unit=(u.hourangle, u.degree))
sweep_objects = SkyCoord(RA1, Dec1, unit=(u.degree, u.degree))
separations = PG1633_099A.separation(sweep_objects)
ix = separations.argmin()

# LB finding the flux of PG1633_099A
G = objs1["FLUX_G"][ix]
R = objs1["FLUX_R"][ix]
Z = objs1["FLUX_Z"][ix]
W1 = objs1["FLUX_W1"][ix]
W2 = objs1["FLUX_W2"][ix]
W3 = objs1["FLUX_W3"][ix]
W4 = obs1["FLUX_W4"][ix]


# LB converting flux to magnitude
print(G)
gm = 22.5 - 2.5 * np.log10(G)
rm = 22.5 - 2.5 * np.log10(R)
zm = 22.5 - 2.5 * np.log10(Z)
w1m = 22.5 - 2.5 * np.log10(W1)
w2m = 22.5 - 2.5 * np.log10(W2)
w3m = 22.5 - 2.5 * np.log10(W3)
w4m = 22.5 - 2.5 * np.log10(W4)

print(f'The g magnitude is {gm}')
print(f'The r magnitude is {rm}')
print(f'The z magnitude is {zm}')
print('They somewhat agree with the Navigate values')
      
print(f'The W1 magnitude is {w1m}')
print(f'The W2 magnitude is {w2m}')
print(f'The W3 magnitude is {w3m}')
print(f'The W4 magnitude is {w4m}')
print('Not detected in W4 band')

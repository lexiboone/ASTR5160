#!/usr/bin/env python
# coding: utf-8

# In[8]:


from sphericalcaps import *
import pymangle

# LB running the functions used in spherical caps to generate two files that can be used in pymangle
writetopolygon12(5, 6, 30, 40, '4caps.ply')
writetopolygon12(11, 12, 60, 70, '2caps.ply')

# LB reading in the mask using pymangle and generating 1000000 points of RA and Dec in sphere 
m = pymangle.Mangle("4caps.ply")
ra_rand = 360. * (random(1000000))
dec_rand = (180 / np.pi) * np.arcsin(1. - random(1000000) * 2.)

# LB reading in the mask using pymangle and generating 1000000 points of RA and Dec in sphere 
m2 = pymangle.Mangle("2caps.ply")
ra_rand2 = 360. * (random(1000000))
dec_rand2 = (180 / np.pi) * np.arcsin(1. - random(1000000) * 2.)

# LB finding the positions of points that fell in mask
good = m.contains(ra_rand, dec_rand)
good1 = m2.contains(ra_rand2, dec_rand2)

# LB plotting the points and the points that fell in mask
plt.figure(figsize=(16,9))
plt.scatter(ra_rand,dec_rand, color='hotpink',label='Random points')
plt.scatter(ra_rand[good],dec_rand[good], color='mediumseagreen',label='Random points in mask 1')
plt.xlabel('RA')
plt.ylabel('Dec')
plt.show()

plt.figure(figsize=(16,9))
plt.scatter(ra_rand2,dec_rand2, color='hotpink',label='Random points')
plt.scatter(ra_rand2[good1],dec_rand2[good1], color='mediumseagreen',label='Random points in mask 2')
plt.xlabel('RA')
plt.ylabel('Dec')
plt.show()

# LB randomly generating 10000 points
ragen, decgen = m.genrand(10000)
ragen1, decgen1 = m2.genrand(10000)

# LB plotting those points
plt.figure(figsize=(16,9))
plt.scatter(ragen,decgen, color='mediumseagreen',label='Random points in mask 1')
plt.xlabel('RA')
plt.ylabel('Dec')
plt.show()

plt.figure(figsize=(16,9))
plt.scatter(ragen1,decgen1, color='mediumseagreen',label='Random points in mask 2')
plt.xlabel('RA')
plt.ylabel('Dec')
plt.show()

# In[ ]:





# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[1]:


import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import chi2
file = glob.glob('/d/scratch/ASTR5160/week13/line.data')[0]

# Reading in file
    
df = pd.read_csv(file, delim_whitespace=True) # For tab-separated data
#print(df)
#print(df[0])
# print(df.columns)

x0 = np.array([df['#x']])
# print(x0)
x1 = np.array([df['0.5']])
x2 = np.array([df['1.5']])
x3 = np.array([df['2.5']])
x4 = np.array([df['3.5']])
x5 = np.array([df['4.5']])
x6 = np.array([df['5.5']])
x7 = np.array([df['6.5']])
x8 = np.array([df['7.5']])
x9 = np.array([df['8.5']])

x_bins = [x0,x1,x2,x3,x4,x5,x6,x7,x8,x9]
x_bins_names = ['x0','x1','x2','x3','x4','x5','x6','x7','x8','x9']
O = []
vara = []
stda =[]

# finding mean and variance and std
for i in range(len(x_bins)):
    mean = np.mean(x_bins[i])
    var = np.var(x_bins[i], ddof = 1)
    std = np.std(x_bins[i], ddof=1)
    O.append(mean)
    vara.append(var)
    stda.append(std)
    print(f'The mean for {x_bins_names[i]} is {mean}')
    print(f'The variance for {x_bins_names[i]} is {var}')

print(x0)
print(len(x0))

# for plotting
x0bin = np.full(20,0.5)
x1bin = np.full(20,1.5)
x2bin = np.full(20,2.5)
x3bin = np.full(20,3.5)
x4bin = np.full(20,4.5)
x5bin = np.full(20,5.5)
x6bin = np.full(20,6.5)
x7bin = np.full(20,7.5)
x8bin = np.full(20,8.5)
x9bin = np.full(20,9.5)

xsmooth = np.arange(0,11,0.1)
m = 3
b = 5
ysmooth = m*xsmooth + b
print(x0bin)
plt.figure(figsize=(16,9))
plt.scatter(x0bin,x0, label='x0')
plt.scatter(x1bin,x1, label='x1')
plt.scatter(x2bin,x2, label='x2')
plt.scatter(x3bin,x3, label='x3')
plt.scatter(x4bin,x4, label='x4')
plt.scatter(x5bin,x5, label='x5')
plt.scatter(x6bin,x6, label='x6')
plt.scatter(x7bin,x7, label='x7')
plt.scatter(x8bin,x8, label='x8')
plt.scatter(x9bin,x9, label='x9')
plt.plot(xsmooth,ysmooth)
plt.show()

mrange = np.arange(3.1,3.9,0.8/100)
print(len(mrange))
brange = np.arange(4,6,2/100)
print(len(brange))
# ms, bs = np.meshgrid(mrange,brange)
# ms = np.ravel(ms)
# bs = np.ravel(bs)
x_bins_values = [0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5]

# xbins_vals = [np.full(10000, i) for i in x_bins_values]

# E = [ms * i + bs for i in xbins_vals]
# #print(E[0])
# #O = [x0,x1,x2,x3,x4,x5,x6,x7,x8,x9]
# O_E_sq = (O - E)**2
# chi_sq = np.sum((O_E_sq)/var)

        
print(O)
x0 = df['#x']
# print(x0)
x1 = df['0.5']
x2 = df['1.5']
x3 = df['2.5']
x4 = df['3.5']
x5 = df['4.5']
x6 = df['5.5']
x7 = df['6.5']
x8 = df['7.5']
x9 = df['8.5']

x_bins = [x0,x1,x2,x3,x4,x5,x6,x7,x8,x9]
#print(x_bins)
print(len(x_bins))
expected_y = []
chisqa = []
mval = []
# Plotting chisq
plt.figure(figsize=(16,9))
for i in range(len(mrange)):
    #mvalues.append(mrange[i])
    for ix in range(len(brange)):
        x_bins_values = np.array([0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5])
        
        E = mrange[i] * x_bins_values + brange[ix]
        #print(E[0])
        #O = [x0,x1,x2,x3,x4,x5,x6,x7,x8,x9]
        O_E_sq = (O - E)**2
        chi_sq = np.sum((O_E_sq)/var)
        plt.scatter(mrange[i],chi_sq, color='indigo')
        plt.scatter(brange[ix],chi_sq, color='deeppink')
        
        chisqa.append(chi_sq)
#         for i in range(len(O)):
#             #print(len(O))
#             #print(len(x0))
#             #print(O)
#             O_E_sq = (O[i] - E[i])**2
#             var = np.var(O[i], ddof=1)
#             chi_sq = np.sum((O_E_sq)/var)
#             #print(chi_sq)
#             chisqa.append(chi_sq)
#             #print(var)
#             #print(E)
print(len(chisqa))
plt.scatter(3.5,50, color='indigo', label='m')
plt.scatter(4.1,50, color='deeppink', label='b')

plt.legend()
plt.show()
#plt.plot(mrange,chisqa)

# LB different way Xander showed me
# Make grid of m and b and ravel so there are related length m and b for indexing
ms, bs = np.meshgrid(mrange,brange)
ms = np.ravel(ms)
bs = np.ravel(bs)
x_bins_values = [0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5]
E = np.array([ms*i+bs for i in x_bins_values]).T
# Resizing
o = O*np.ones((10000,10)) 
v = var*np.ones((10000,10))
#O = [x0,x1,x2,x3,x4,x5,x6,x7,x8,x9]
# Finding chi sq
O_E_sq = (o - E)**2
chi_sq = np.sum((O_E_sq)/v, axis=1)

mci = np.argmin(chi_sq)
#ms, bs = np.meshgrid(mrange,brange)

print(len(ms))
bestm = ms[mci]
bestb = bs[mci]
bestchi = chisqa[mci]
x_bins_values = [0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5,8.5,9.5]



print(f'The minimum chi squared is {bestchi}, corresponding to a line with slope {bestm} and intercept {bestb}')
plt.plot(ms,chi_sq)
plt.plot(bs,chi_sq)
# Finding where chisq confidence is 68 and 95%

i68 = chi2.sf(chi_sq, 2) > .32
i95 = chi2.sf(chi_sq, 2) > .05

m68 = ms[i68]
b68 = bs[i68]

m95 = ms[i95]
b95 = bs[i95]
xsmooth = np.arange(0,11,0.1)
m = 3
b = 5
ysmoothn = bestm*xsmooth + bestb
print(x0bin)
plt.figure(figsize=(16,9))
# IDK why these confidence bands look the same :(
#Plotting the best fit lines and confidence bands
[plt.plot(xsmooth, m68[i]*xsmooth + b68[i],color='pink',alpha=0.3) for i in range(len(m68))]
[plt.plot(xsmooth, m95[i]*xsmooth + b95[i],color='deeppink', alpha=0.3) for i in range(len(m95))]
plt.scatter(x0bin,x0, label='x0')
plt.scatter(x1bin,x1, label='x1')
plt.scatter(x2bin,x2, label='x2')
plt.scatter(x3bin,x3, label='x3')
plt.scatter(x4bin,x4, label='x4')
plt.scatter(x5bin,x5, label='x5')
plt.scatter(x6bin,x6, label='x6')
plt.scatter(x7bin,x7, label='x7')
plt.scatter(x8bin,x8, label='x8')
plt.scatter(x9bin,x9, label='x9')
plt.errorbar(x_bins_values, O, yerr=std, fmt='o', color='black', label='average y')
plt.plot(xsmooth,ysmoothn, color='mediumvioletred', label='Minimum chi squared')


plt.show()


# In[61]:


from numpy import matrix
from numpy import linalg
x0 = np.array([df['#x']])
# print(x0)
x1 = np.array([df['0.5']])
x2 = np.array([df['1.5']])
x3 = np.array([df['2.5']])
x4 = np.array([df['3.5']])
x5 = np.array([df['4.5']])
x6 = np.array([df['5.5']])
x7 = np.array([df['6.5']])
x8 = np.array([df['7.5']])
x9 = np.array([df['8.5']])

# Stacking all bins and making matrix
x_bins = [x0,x1,x2,x3,x4,x5,x6,x7,x8,x9]
y_meas = np.row_stack((x0,x1,x2,x3,x4,x5,x6,x7,x8,x9))
y_meas = matrix(y_meas)

    
tymeas = y_meas #.T
#print(np.cov(tymeas))

# Making covariance matrix
covmatrix = np.cov(tymeas)
print(np.shape(np.cov(tymeas)))
print('Shape is 10x10')
diag = np.diag(covmatrix)
print(diag)
# LB matches the variance we found above
print(covmatrix)
anticorr = np.abs(-1 - covmatrix)
print('Anticorr')
print((anticorr))

print(np.argmin(anticorr))
print(anticorr[7,0])
print('1 & 8 are most anticorrelated')

corr = np.abs(covmatrix-1)
print('Corr')
print((corr))

print(np.argmin(corr))
print('5 & 6 are most correlated')

#print(min(anticorr))
# for i in range(len(covmatrix)):
#     for u in range(len(covmatrix)):
#         anticorr = 


# In[ ]:





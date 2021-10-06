#!/usr/bin/env python
# coding: utf-8

# # Import Packages

# In[1]:


import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm, rcParams, colors
from matplotlib import gridspec as gspec
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import FontProperties
import matplotlib.path as mpat
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
#from colorspacious import cspace_converter
from collections import OrderedDict


# Analysis imports
import numpy as np
import numpy.ma as ma
import csv
import netCDF4 as nc
from netCDF4 import Dataset
import iris
import iris.coord_categorisation
import glob
import warnings
from iris.experimental.equalise_cubes import equalise_attributes

import pandas as pd
from sklearn import linear_model


cmaps = OrderedDict()
cmaps['Diverging'] = [
            'PiYG', 'PRGn', 'BrBG', 'PuOr', 'RdGy', 'RdBu',
            'RdYlBu', 'RdYlGn', 'Spectral', 'coolwarm', 'bwr', 'seismic']

cmaps2 = OrderedDict()
cmaps2['Sequential'] = [
            'YIGn', 'YIOrRd', 'YIOrBr', 'BuGn', 'Greens']


color = ['darkblue', 'dodgerblue', '#80b1d3', 'darkcyan', '#8dd3c7', 'darkseagreen', 'darkgreen', 'olive', 'gold', 
         'orange', 'peachpuff', '#fb8072', 'red', 'hotpink', '#fccde5','#bebada' ]


# # Import data and model outputs

# In[3]:


## Data and model outputs for individual sites

Leaf = pd.read_csv('~/outputs/Leaf_data.csv')
TR = pd.read_csv('~/outputs/TR_slope_data.csv')

leaf_obs_mod = pd.read_csv('~/outputs/Leaf_obs_mod.csv')
TR_slope_IAV_obs_mod = pd.read_csv('~/outputs/TR_slope_IAV.csv')
TR_obs = pd.read_csv('~/outputs/TR_obs.csv')


## Model outputs for global run with photorespiratory and mesophyll effects (available upon request through the NERC JASMIN platform; http://www.jasmin.ac.uk/)
## PFTs: BET-Tr, BET-Te, BDT, NET, NDT, C3, C4, ESh, DSh

year1=1979
year2=2016
nbpft=5

data1=nc.Dataset('~/outputs/2d/WFDEI_global.all.PFTs.month.2d.meso.'+str(year1)+'.'+str(year2)+'.nc')

lat1=data1.variables['lat'][:]
lon1=data1.variables['lon'][:]


data2=nc.Dataset('~/ancils/1979_2016.vpd.nc')


# In[6]:


# Labels model outputs

## PFTs dependent
label1='D13Cleaf'
title1=u'$\mathregular{\u0394^{13}C}$ (‰)'

label2='ci_leaf'
title2=r'$C_\mathrm{i}$ (Pa)'

label3='gpp'
title3=r'GPP (g C m$^{-2}$ d$^{-1}$)'

label4='npp'
title4=r'NPP (g C m$^{-2}$ d$^{-1}$)'

label5='gppc13'
title5=r'$\mathregular{^{13}C}$ GPP (g C m$^{-2}$ d$^{-1}$)'

label6='respc13'
title6=r'$\mathregular{^{13}C}$ RA (g C m$^{-2}$ d$^{-1}$)'

label7='nppc13'
title7=r'$\mathregular{^{13}C}$ NPP (g C m$^{-2}$ d$^{-1}$)'

label8='lai'
title8=u'LAI (m$^{-2}$ m$^{-2}$)'

label9='cc_leaf'
title9=r'$C_\mathrm{c}$ (Pa)'

label10='anetc'
title10=r'Ac (g C m$^{-2}$ d$^{-1}$)'

label11='fsmc'
title11=u'$\mathregular{\u03B2_{soil}}$ (unitless)'

label12='d13Cleaf'
title12=u'$\mathregular{\u03B4^{13}C_{leaf}}$ (‰)'

label13='d13Cplant'
title13=u'$\mathregular{\u03B4^{13}C_{plant}}$ (‰)'


## no PFT dependent
label14='et_stom'
title14=r'T$_r$ (kg H$_2$O m$^{-2}$ d$^{-1}$)'

label15='et_tot'
title15=r'ET (kg H$_2$O m$^{-2}$ d$^{-1}$)'

label16='gs'
title16=r'G$_c$ (kg H$_2$O m$^{-2}$ d$^{-1}$)'

label17='Tair'
title17=r'$T_{air}$ (degC)'

label18='smc_avail_top'
title18=r'$SWC_\mathrm{top 1m}$ (kg m$^{-2}$)'

label19='d13Catm'
title19=u'$\mathregular{\u03B4^{13}C_{atm}}$ (‰)'

label20='latent_heat'
title20=r'LE (W s$^{-1}$)'

label21='co2'
title21=r'CO$_2$ (ppm)'

label22='patm'
title22=u'P$_{atm}$ (Pa)'

label23='iWUE'
title23=u'iWUE ($\mathregular{\u03BC}$mol mol$^{-1}$)'

label24='ci/ca'
title24=r'$C_\mathrm{i}$/$C_\mathrm{a}$'

label25='WUE_t'  ## anetc_2d/et_stom
title25=r'WUE (gC kgH$_2$O$^{-1}$)'   

label26='WUE_v1'  ## gpp_2d/et_tot 
title26=r'WUE (gC kgH$_2$O$^{-1}$)'

label27='WUE_v2'   ## gpp_2d/et_stom 
title27=r'WUE (gC kgH$_2$O$^{-1}$)'

label28='WUE_v3'   ## npp_2d/et_stom 
title28=r'WUE (gC kgH$_2$O$^{-1}$)'

label29='IWUE'   ## np.average(f.variables['anetc'][:,0:j,0,:],axis=1)*1e6/gs 
title29=r'IWUE (gC kgH$_2$O$^{-1}$)'


# In[7]:


## Extracting variables
var1=data1.variables[label1][:,:,:,0:nbpft]
var3=data1.variables[label3][:,:,:,0:nbpft]
var17=data1.variables[label17][:,:,:]
var21=data1.variables[label21][:,:,:]
var11=data1.variables[label11][:,:,:,0:nbpft]
var11[var11==0]=np.nan

vpd=data2.variables['vpd']

del data1


# In[9]:


## Calculating yearly average of [CO2]atm and Tair
finalco2=[]
cy=0
finalT=[]
cy1=0


for i in range(456):
        if i%12==11:
            cy+=var21[:,:,i]     
            cy1+=var17[:,:,i]                      
            finalco2.append(cy/12)
            finalT.append(cy1/12)
            cy=0
            cy1=0
        else:
            cy+=var21[:,:,i]  
            cy1+=var17[:,:,i]  
finalco2 = np.array(finalco2)
finalT = np.array(finalT)

finalco2[finalco2==0]=np.nan

finalvpd=np.array(vpd)
finalvpd[finalvpd<=0]=np.nan

del vpd


# In[12]:


## Filtering model ouputs

var1[var1<0] = np.nan
var1[var1==0] = np.nan
var1[var1>100] = np.nan

var3[var3<0] = np.nan
var3[var3==0] = np.nan


# In[14]:


## Calculating yearly weighted average of D13C, GPP and beta (fsmc)
final=[]
final2=[]
final3=[]
cy=0
cy2=0
cy3=0

for z in range(5):
    for i in range(456):
            if i%12==11:
                cy+=var1[:,:,i,z]*var3[:,:,i,z]
                cy2+=var3[:,:,i,z]
                cy3+=var11[:,:,i,z]                      
                final.append(cy/cy2)
                final2.append(cy2/12)
                final3.append(cy3/12)
                cy=0
                cy2=0
                cy3=0
            else:
                cy+=var1[:,:,i,z]*var3[:,:,i,z]   
                cy2+=var3[:,:,i,z]       
                cy3+=var11[:,:,i,z]  
final = np.array(final)
final2 = np.array(final2)
final3 = np.array(final3)


## Filtering data
final[final==0] = np.nan
final[final>30] = np.nan

final2[final2==0] = np.nan
final3[final3==0] = np.nan

finalT[finalT==0]=np.nan

del var1, var3, var11


## Reorganising files
finalf = np.ma.empty((38,360,720,5))
final2f = np.ma.empty((38,360,720,5))
final3f = np.ma.empty((38,360,720,5))

finalf.mask = True
final2f.mask = True
final3f.mask = True

finalf[:,:,:,0] = final[0:38,:,:]
finalf[:,:,:,1] = final[38:(38+38),:,:]
finalf[:,:,:,2] = final[76:(76+38),:,:]
finalf[:,:,:,3] = final[114:(114+38),:,:]
finalf[:,:,:,4] = final[152:(152+38),:,:]

final2f[:,:,:,0] = final2[0:38,:,:]
final2f[:,:,:,1] = final2[38:(38+38),:,:]
final2f[:,:,:,2] = final2[76:(76+38),:,:]
final2f[:,:,:,3] = final2[114:(114+38),:,:]
final2f[:,:,:,4] = final2[152:(152+38),:,:]

final3f[:,:,:,0] = final3[0:38,:,:]
final3f[:,:,:,1] = final3[38:(38+38),:,:]
final3f[:,:,:,2] = final3[76:(76+38),:,:]
final3f[:,:,:,3] = final3[114:(114+38),:,:]
final3f[:,:,:,4] = final3[152:(152+38),:,:]


# In[20]:


## Calculation average values
final = np.nanmean(finalf,axis=3) 
final2 = np.nanmean(final2f,axis=3) 
final3 = np.nanmean(final3f,axis=3) 


## Calculation average values of Tair, beta and VPD for categorial groups 

Tann = np.nanmean(finalT,axis=0)
betann = np.nanmean(final3,axis=0)
vpdann = np.nanmean(finalvpd,axis=0)


## Filtering data
Tann[Tann < -15] = np.nan
Tann[Tann > 35] = np.nan

Tann[(Tann >= -15) & (Tann < -10)] = 100
Tann[(Tann >= -10) & (Tann < -5)] = 200
Tann[(Tann >= -5) & (Tann < 0)] = 300
Tann[(Tann >= 0) & (Tann < 5)] = 400
Tann[(Tann >= 5) & (Tann < 10)] = 500
Tann[(Tann >= 10) & (Tann < 15)] = 600
Tann[(Tann >= 15) & (Tann < 20)] = 700
Tann[(Tann >= 20) & (Tann < 25)] = 800
Tann[(Tann >= 25) & (Tann < 30)] = 900
Tann[(Tann >= 30) & (Tann < 35)] = 1000

Tann = Tann/100


betann[(betann < 0.20)] = np.nan

betann[(betann >= 0.2) & (betann < 0.3)] = 100
betann[(betann >= 0.3) & (betann < 0.4)] = 200
betann[(betann >= 0.4) & (betann < 0.5)] = 300
betann[(betann >= 0.5) & (betann < 0.6)] = 400
betann[(betann >= 0.6) & (betann < 0.7)] = 500
betann[(betann >= 0.7) & (betann < 0.8)] = 600
betann[(betann >= 0.8) & (betann < 0.9)] = 700
betann[(betann >= 0.9) & (betann <= 1)] = 800

betann = betann/100



vpdann[(vpdann == 0)] = np.nan

vpdann[(vpdann > 0) & (vpdann < 0.4)] = 100
vpdann[(vpdann >= 0.4) & (vpdann < 0.8)] = 200
vpdann[(vpdann >= 0.8) & (vpdann < 1.2)] = 300
vpdann[(vpdann >= 1.2) & (vpdann < 1.6)] = 400
vpdann[(vpdann >= 1.6) & (vpdann < 2)] = 500
vpdann[(vpdann >= 2) & (vpdann < 2.4)] = 600
vpdann[(vpdann >= 2.4) & (vpdann < 2.8)] = 700
vpdann[(vpdann >= 2.8) & (vpdann < 3.4)] = 800

vpdann = vpdann/100


## Preparation outputs for figure

betann[betann==1] = 0.25
betann[betann==2] = 0.35
betann[betann==3] = 0.45
betann[betann==4] = 0.55
betann[betann==5] = 0.65
betann[betann==6] = 0.75
betann[betann==7] = 0.85
betann[betann==8] = 0.95

vpdann[vpdann==1] = 0.2
vpdann[vpdann==2] = 0.6
vpdann[vpdann==3] = 1.0
vpdann[vpdann==4] = 1.4
vpdann[vpdann==5] = 1.8
vpdann[vpdann==6] = 2.2
vpdann[vpdann==7] = 2.6
vpdann[vpdann==8] = 3.0

Tann[Tann==1] = -12.5
Tann[Tann==2] = -7.5
Tann[Tann==3] = -2.5
Tann[Tann==4] = 2.5
Tann[Tann==5] = 7.5
Tann[Tann==6] = 12.5
Tann[Tann==7] = 17.5
Tann[Tann==8] = 22.5
Tann[Tann==9] = 27.5
Tann[Tann==10] = 32.5


# # Calculations trends D13C and GPP

# In[27]:


import statsmodels.api as sm
import statsmodels.formula.api as smf

from numpy import NaN
from patsy import dmatrices


## Calculation temporal trend of D13C and GPP

trendD13CMatrix = [[0 for i in range(len(lon1))] for j in range(len(lat1))] 
trendGPPMatrix = [[0 for i in range(len(lon1))] for j in range(len(lat1))] 

for x in range(len(lat1)):
    for y in range(len(lon1)):
        final[:,x,y][final[:,x,y]<0] = np.nan
        md = smf.GLS(final[:,x,y],sm.add_constant(range(38))).fit()
        if md.pvalues[1]>0.1:
            trendD13CMatrix[x][y] = np.nan
        else:
            trendD13CMatrix[x][y] = md.params[1]      
            
        final2[:,x,y][final2[:,x,y]<0] = np.nan
        md2 = smf.GLS(final2[:,x,y],sm.add_constant(range(38))).fit()
        if md2.pvalues[1]>0.1:
            trendGPPMatrix[x][y] = np.nan
        else:
            trendGPPMatrix[x][y] = md2.params[1]          
        
trendD13CMatrix = np.squeeze(np.asarray(trendD13CMatrix))
trendGPPMatrix = np.squeeze(np.asarray(trendGPPMatrix))


# # Figure 5

# In[33]:


import math

def isNaN(string):
    return string != string


## Correlation coefficient array between D13C and GPP

corrcoefMatrix = [[0 for i in range(len(lon1))] for j in range(len(lat1))] 

for x in range(len(lat1)):
    for y in range(len(lon1)):
        if isNaN(np.corrcoef(final[:,x,y],final2[:,x,y])[0,1]):
            corrcoefMatrix[x][y] = np.nan
        else:
            md = smf.GLS(final[:,x,y],sm.add_constant(final2[:,x,y])).fit()
            if md.pvalues[1]>0.1:
                corrcoefMatrix[x][y] = np.nan
            else:
                corrcoefMatrix[x][y] = np.corrcoef(final[:,x,y],final2[:,x,y])[0,1]

corrcoefMatrix = np.squeeze(np.asarray(corrcoefMatrix))
corrcoefMatrixf = corrcoefMatrix.flatten() 


# In[35]:


## Correlation values for Figure 5

marks_0_gr_5_corr_moy_beta = np.array([
                    [ np.nan, np.nan,  np.nan,  np.nan,  np.nan, np.nan,   -0.21,  0.32,   0.27,   np.nan],
                    [ np.nan, np.nan,  0.37,    np.nan,  np.nan, np.nan,   0.43,   0.42,   0.23,   0.36],
                    [ 0.21,   0.15,    0.33,    0.29,    np.nan, 0.43,     0.42,   0.42,   0.30,   0.49],               
                    [ 0.14,   -0.12,   0.005,   0.29,    0.43,   0.47,     0.43,   0.43,   0.33,   0.48],
                    [ -0.27,  -0.31,   -0.23,   0.22,    0.39,   0.44,     0.44,   0.47,   0.35,   0.45],
                    [ -0.43,  -0.45,   -0.33,   -0.11,   0.38,   0.48,     0.44,   0.44,   0.45,   0.40],
                    [ np.nan, -0.04,   -0.18,   -0.05,   0.29,   0.44,     0.44,   0.39,   0.43,   0.19],
                    [ np.nan, 0.05,    0.12,    -0.009,  0.13,   0.25,     0.37,   0.18,   0.27,   0.12]])

nb_0_gr_5_corr_moy_beta = np.array([
                 [0,   0,   0,   0,   0,   0,    22,  99,  53,   16],
                 [11,  5,   76,  9,   0,   1,    70,  449, 277,  50],
                 [149, 84,  77,  23,  15,  35,   172, 376, 405,  51],
                 [182, 281, 151, 89,  44,  140,  194, 290, 358,  63],
                 [71,  145, 217, 114, 109, 132,  208, 289, 345,  27],
                 [63,  186, 322, 161, 209, 161,  198, 316, 378,  35],
                 [14,  174, 191, 186, 168, 190,  256, 533, 764,  62],
                 [4,   76,  87,  258, 558, 430,  411, 899, 1959, 165]])


marks_0_gr_5_corr_moy_vpd = np.array([
                    [ 0.03,    -0.13,    -0.08,  -0.01,   0.19,     0.39,   np.nan, np.nan,np.nan, np.nan],
                    [ np.nan,  -0.48,    -0.28,  0.08,    0.26,     0.36,   0.41,   0.31,  0.32,   np.nan],
                    [ np.nan,  np.nan,  np.nan,  0.51,    0.42,     0.42,   0.41,   0.40,  0.39,   0.40],               
                    [ np.nan,  np.nan,  np.nan,  np.nan,  np.nan,   0.43,   0.40,   0.31,  0.39,   0.41],
                    [ np.nan,  np.nan,  np.nan,  np.nan,  np.nan,   np.nan, 0.41,   0.27,  0.23,   0.37],
                    [ np.nan,  np.nan,  np.nan,  np.nan,  np.nan,   np.nan, np.nan, 0.35,  0.05,   0.23],
                    [ np.nan,  np.nan,  np.nan,  np.nan,  np.nan,   np.nan, np.nan, 0.46,  0.20,   -0.02],
                    [ np.nan,  np.nan,  np.nan,  np.nan,  np.nan,   np.nan, np.nan, np.nan,0.28,   0.04]])

nb_0_gr_5_corr_moy_vpd = np.array([
                 [468, 887, 949, 530, 442, 104, 15,   18,  2,    0],
                 [5,  61,  164, 281,  588, 737, 661,  773, 1344, 3],
                 [0,  0,  1,  29,     59,  190, 570,  1485,1616, 66],
                 [0,  0,  0,  0,      14,  53,  223,  604, 738,  110],
                 [0,  0,  0,  0,      0,   5,   65,   248, 308,  87],
                 [0,  0,  0,  0,      0,   0,    0,   107, 345,  67],
                 [0,  0,  0,  0,      0,   0,    0,   22,  139,  29],
                 [0,  0,  0,  0,      0,   0,    0,   0,   48,   72]])

marks_0_gr_tot_corr_moy_beta = marks_0_gr_5_corr_moy_beta
nb_0_gr_corr_moy_beta = nb_0_gr_5_corr_moy_beta

nb_0_gr_corr_moy_beta = np.transpose(nb_0_gr_corr_moy_beta)
marks_0_gr_tot_corr_moy_beta = np.transpose(marks_0_gr_tot_corr_moy_beta)


marks_0_gr_tot_corr_moy_vpd = marks_0_gr_5_corr_moy_vpd
nb_0_gr_corr_moy_vpd = nb_0_gr_5_corr_moy_vpd

nb_0_gr_corr_moy_vpd = np.transpose(nb_0_gr_corr_moy_vpd)
marks_0_gr_tot_corr_moy_vpd = np.transpose(marks_0_gr_tot_corr_moy_vpd)


nb_0_gr_per_corr_moy_beta = nb_0_gr_corr_moy_beta/np.sum(nb_0_gr_corr_moy_beta)*100
nb_0_gr_per_corr_moy_beta = np.round(nb_0_gr_per_corr_moy_beta, 1)
nb_0_gr_per_corr_moy_vpd = nb_0_gr_corr_moy_vpd/np.sum(nb_0_gr_corr_moy_vpd)*100
nb_0_gr_per_corr_moy_vpd = np.round(nb_0_gr_per_corr_moy_vpd, 1)


# In[38]:


betas=['1','0.9','0.8','0.7','0.6','0.5','0.4','0.3']
vpds=['3.0','2.6','2.2','1.8','1.4','1.0','0.6','0.2']
temps=['-10','-5','0','5','10','15','20','25','30','35']


# In[39]:


## Figure 5 -  Heatmaps trends and correlations

# Set up the subplot figure
fig_figure5 = plt.figure(1, figsize=(30,45))
gs = gspec.GridSpec(3, 3, figure=fig_figure5, width_ratios=[1, 1, 0.05], hspace=0.2)
# set rows and column
column = 0
row = 0
mpl.rcParams['xtick.direction'] = 'out'
mpl.rcParams['ytick.direction'] = 'out'
mpl.rcParams['xtick.top'] = True
mpl.rcParams['ytick.right'] = True
params = {
    'lines.linewidth':3,
    'axes.facecolor':'white',
    'xtick.color':'k',
    'ytick.color':'k',
    'axes.labelsize': 34,
    'xtick.labelsize':34,
    'ytick.labelsize':34,
    'font.size':34,
    'text.usetex': False,
    "svg.fonttype": 'none'
}
plt.rcParams.update(params)

#%%

# Figure 5a

ax = fig_figure5.add_subplot(gs[row, column])

hm=ax.imshow(marks_0_gr_5_D13C_moy_beta, cmap='BrBG',interpolation="nearest", vmin=-0.50, vmax=0.50)

ax.invert_yaxis()

ax.set_ylabel(r'Annual $T_{air}$ (°C)')
ax.set_yticks([-0.5,0.5,1.5,2.5,3.5,4.5,5.5, 6.5, 7.5, 8.5, 9.5])
ax.set_yticklabels(['-15','-10','-5','0','5','10','15','20','25','30','35'])

ax.set_xlabel(u'Annual $\mathregular{\u03B2_{soil}}$ (-)')
ax.set_xticks([0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5])
ax.set_xticklabels(['0.3','0.4','0.5','0.6','0.7','0.8','0.9','1'])

ax.tick_params(axis='y', which='both', right=False, 
                left=True, labelleft=True) 
ax.tick_params(axis='x', which='both', bottom=True, 
                top=False, labelbottom=True) 

ax.text(-0.2, 1.05, '(a)',transform=ax.transAxes,va = 'top',fontweight = 'bold',fontsize=34)

# Loop over data dimensions and create text annotations.
for i in range(len(temps)):
    for j in range(len(betas)):
        text = ax.text(j, i, nb_0_gr_per_D13C_moy_beta[i, j],
                       ha="center", va="center", color="black",fontsize=18)

        
# Figure 5b

column += 1

ax = fig_figure5.add_subplot(gs[row, column])

hm=ax.imshow(marks_0_gr_5_D13C_moy_vpd, cmap='BrBG',interpolation="nearest", vmin=-0.50, vmax=0.50)

ax.invert_yaxis()

ax.set_ylabel(r'Annual $T_{air}$ (°C)')
ax.set_yticks([-0.5,0.5,1.5,2.5,3.5,4.5,5.5, 6.5, 7.5, 8.5, 9.5])
ax.set_yticklabels(['-15','-10','-5','0','5','10','15','20','25','30','35'])

ax.set_xlabel(u'Annual $D$ (kPa)')
ax.set_xticks([0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5])
ax.set_xticklabels(['0.4','0.8','1.2','1.6','2.0','2.4','2.8','3.2'])

ax.tick_params(axis='y', which='both', right=False, 
                left=True, labelleft=True) 
ax.tick_params(axis='x', which='both', bottom=True, 
                top=False, labelbottom=True) 


ax.text(-0.2, 1.05, '(b)',transform=ax.transAxes,va = 'top',fontweight = 'bold',fontsize=34)


column += 1
plt.colorbar(hm,extend='both',orientation='vertical').set_label(u'$\mathregular{\u0394^{13}C}$ trend scores (‰ over 1979-2016)')


# Loop over data dimensions and create text annotations.
for i in range(len(temps)):
    for j in range(len(vpds)):
        text = ax.text(j, i, nb_0_gr_per_D13C_moy_vpd[i, j],
                       ha="center", va="center", color="black",fontsize=18)


        # Figure 5c
gs = gspec.GridSpec(3, 4, figure=fig_figure5, width_ratios=[0.02, 1, 0.03, 0.10], hspace=0.25)

row = 1
column = 1

ax = fig_figure5.add_subplot(gs[row, 1], projection=ccrs.PlateCarree())


# Add lat/lon grid lines to the figure
gl = ax.gridlines(linestyle='solid', color='white', linewidth=0.5, alpha=0.5)
gl.yformatter=LATITUDE_FORMATTER
gl.ylocator = mticker.FixedLocator([-40, -20, 0, 20, 40, 60, 80])
gl.ylabels_left=True
gl = ax.gridlines(linestyle='solid', color='white', linewidth=0.5, alpha=0.5)
gl.xformatter=LONGITUDE_FORMATTER
gl.xlabels_bottom=True
ax.coastlines()

line = np.arange(-0.9, 0.9, 0.05)
diff = plt.contourf(lon1, lat1, corrcoefMatrix,line, cmap = 'RdBu_r', extend='both', transform=ccrs.PlateCarree(central_longitude=0))

ax.text(-0.1, 1.05, '(c)',transform=ax.transAxes,va = 'top',fontweight = 'bold',fontsize=34)

column += 1
ax=fig_figure5.add_subplot(gs[row,column])
ax=plt.gca()
fig_figure5.colorbar(diff, ax, orientation='vertical').set_label(u'Correlation $\mathregular{\u0394^{13}C}$ - GPP')


# Figure 5d

gs = gspec.GridSpec(3, 3, figure=fig_figure5, width_ratios=[1, 1, 0.05], hspace=0.2)

row += 1
column = 0
ax = fig_figure5.add_subplot(gs[row, column])

hm=ax.imshow(marks_0_gr_tot_corr_moy_beta, cmap='RdBu_r',interpolation="nearest", vmin=-0.5, vmax=0.5)

ax.invert_yaxis()

ax.set_ylabel(r'Annual $T_{air}$ (°C)')
ax.set_yticks([-0.5,0.5,1.5,2.5,3.5,4.5,5.5, 6.5, 7.5, 8.5, 9.5])
ax.set_yticklabels(['-15','-10','-5','0','5','10','15','20','25','30','35'])

ax.set_xlabel(u'Annual $\mathregular{\u03B2_{soil}}$ (-)')
ax.set_xticks([0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5])
ax.set_xticklabels(['0.3','0.4','0.5','0.6','0.7','0.8','0.9','1'])

ax.tick_params(axis='y', which='both', right=False, 
                left=True, labelleft=True) 
ax.tick_params(axis='x', which='both', bottom=True, 
                top=False, labelbottom=True) 

ax.text(-0.2, 1.05, '(d)',transform=ax.transAxes,va = 'top',fontweight = 'bold',fontsize=34)

#column += 1
#plt.colorbar(hm,extend='both',orientation='vertical').set_label(u'$\mathregular{\u0394^{13}C}$ - GPP correlation scores')

# Loop over data dimensions and create text annotations.
for i in range(len(temps)):
    for j in range(len(betas)):
        text = ax.text(j, i, nb_0_gr_per_corr_moy_beta[i, j],
                       ha="center", va="center", color="black",fontsize=18)

        
# Figure 5e

column += 1

ax = fig_figure5.add_subplot(gs[row, column])

hm=ax.imshow(marks_0_gr_tot_corr_moy_vpd, cmap='RdBu_r',interpolation="nearest", vmin=-0.5, vmax=0.5)

ax.invert_yaxis()

ax.set_ylabel(r'Annual $T_{air}$ (°C)')
ax.set_yticks([-0.5,0.5,1.5,2.5,3.5,4.5,5.5, 6.5, 7.5, 8.5, 9.5])
ax.set_yticklabels(['-15','-10','-5','0','5','10','15','20','25','30','35'])

ax.set_xlabel(u'Annual $D$ (kPa)')
ax.set_xticks([0.5,1.5,2.5,3.5,4.5,5.5,6.5,7.5])
ax.set_xticklabels(['0.4','0.8','1.2','1.6','2.0','2.4','2.8','3.2'])

ax.tick_params(axis='y', which='both', right=False, 
                left=True, labelleft=True) 
ax.tick_params(axis='x', which='both', bottom=True, 
                top=False, labelbottom=True) 


ax.text(-0.2, 1.05, '(e)',transform=ax.transAxes,va = 'top',fontweight = 'bold',fontsize=34)


column += 1
plt.colorbar(hm,extend='both',orientation='vertical').set_label(u'$\mathregular{\u0394^{13}C}$ - GPP correlation scores')


# Loop over data dimensions and create text annotations.
for i in range(len(temps)):
    for j in range(len(vpds)):
        text = ax.text(j, i, nb_0_gr_per_corr_moy_vpd[i, j],
                       ha="center", va="center", color="black",fontsize=18)

        
#%%

fig_figure5.savefig('~/outputs/Figures/Figure5_final.jpg', bbox_inches='tight')

plt.close()


# # Preparing data for Figures 3 and 4

# In[47]:


## Model outputs for global run with photorespiratory effect only (available upon request through the NERC JASMIN platform; http://www.jasmin.ac.uk/)
import netCDF4 as nc

year1=1979
year2=2016
nbpft=5

data2=nc.Dataset('~/outputs/2d/WFDEI_global.all.PFTs.month.2d.photo.D13Cleaf.'+str(year1)+'.'+str(year2)+'.nc')

var2f=data2.variables[label1][:,:,:,0:nbpft] 
var2f2=data2.variables[label3][:,:,:,0:nbpft]


## Filtering
var2f[var2f<0] = np.nan
var2f[var2f>100] = np.nan

var2f2[var2f2<0] = np.nan
var2f2[var2f2>100] = np.nan


# In[49]:


## Calculation yearly weighted average D13C and GPP
finalf=[]
cy=0
cy2=0

for z in range(5):
    for i in range(456):
            if i%12==11:
                cy+=var2f[:,:,i,z]*var2f2[:,:,i,z]
                cy2+=var2f2[:,:,i,z]                
                finalf.append(cy/cy2)
                cy=0
                cy2=0
            else:
                cy+=var2f[:,:,i,z]*var2f2[:,:,i,z]         
                cy2+=var2f2[:,:,i,z]       
finalf = np.array(finalf)

del var2f, var2f2


## Reorganising files
finalf[finalf == 0] = np.nan
finalf[finalf > 30] = np.nan

finalff = np.ma.empty((38,360,720,5))
finalff.mask = True

finalff[:,:,:,0] = finalf[0:38,:,:]
finalff[:,:,:,1] = finalf[38:(38+38),:,:]
finalff[:,:,:,2] = finalf[76:(76+38),:,:]
finalff[:,:,:,3] = finalf[114:(114+38),:,:]
finalff[:,:,:,4] = finalf[152:(152+38),:,:]

finalf = np.nanmean(finalff,axis=3) 

del finalff


## Calculation difference with or without mesophyll effect
differ = finalf-final


# # Figure 3

# In[81]:


## Figure 3

# Set up the subplot figure

fig_figure3 = plt.figure(1, figsize=(27,35))
gs = gspec.GridSpec(4, 3, figure=fig_figure3, width_ratios=[0.15, 1, 0.05], hspace=0.2)
# set rows and column
column = 0
row = 0
mpl.rcParams['xtick.direction'] = 'out'
mpl.rcParams['ytick.direction'] = 'out'
mpl.rcParams['xtick.top'] = True
mpl.rcParams['ytick.right'] = True
params = {
    'lines.linewidth':3,
    'axes.facecolor':'white',
    'xtick.color':'k',
    'ytick.color':'k',
    'axes.labelsize': 34,
    'xtick.labelsize':34,
    'ytick.labelsize':34,
    'font.size':34,
    'text.usetex': False,
    "svg.fonttype": 'none'
}
plt.rcParams.update(params)

#%%

# Figure 3a

ax = fig_figure3.add_subplot(gs[row, column])

ax.scatter(np.nanmean(np.nanmean(differ2,axis=0),axis=1), lat1, color='k', marker='o', s=20)


ax.set_xlabel(u'$\mathregular{\u0394 \u0394^{13}C}_{photo}$ (‰)')
ax.set_ylim((-70,99))
ax.tick_params(axis='y', which='both', right=False, 
                left=False, labelleft=False) 
ax.tick_params(axis='x', which='both', bottom=True, 
                top=False, labelbottom=True) 

column += 1
ax = fig_figure3.add_subplot(gs[row, column], projection=ccrs.PlateCarree())


# Add lat/lon grid lines to the figure
gl = ax.gridlines(linestyle='solid', color='white', linewidth=0.5, alpha=0.5)
gl.yformatter=LATITUDE_FORMATTER
gl.ylocator = mticker.FixedLocator([-40, -20, 0, 20, 40, 60, 80])
gl.ylabels_left=True
gl = ax.gridlines(linestyle='solid', color='white', linewidth=0.5, alpha=0.5)
gl.xformatter=LONGITUDE_FORMATTER
gl.xlabels_bottom=True
ax.coastlines()

line = np.arange(-4, 4, 0.1)
diff = plt.contourf(lon1, lat1, np.nanmean(differ2,axis=0),line, cmap = 'RdBu_r', extend='both', transform=ccrs.PlateCarree(central_longitude=0))

ax.text(-0.1, 1.05, '(a)',transform=ax.transAxes,va = 'top',fontweight = 'bold',fontsize=34)

ax=fig_figure3.add_subplot(gs[row,2])
ax=plt.gca()
fig_figure3.colorbar(diff, ax, orientation='vertical').set_label(u'$\mathregular{\u0394 \u0394^{13}C}_{photo}$ (‰)')


# Figure 3b
row += 1
column = 0
ax = fig_figure3.add_subplot(gs[row, column])

ax.scatter(np.nanmean(np.nanmean(differ,axis=0),axis=1), lat1, color='k', marker='o', s=20)


ax.set_xlabel(u'$\mathregular{\u0394 \u0394^{13}C}_{meso}$ (‰)')
ax.set_ylim((-70,99))
ax.tick_params(axis='y', which='both', right=False, 
                left=False, labelleft=False) 
ax.tick_params(axis='x', which='both', bottom=True, 
                top=False, labelbottom=True) 

column += 1
ax = fig_figure3.add_subplot(gs[row, column], projection=ccrs.PlateCarree())


# Add lat/lon grid lines to the figure
gl = ax.gridlines(linestyle='solid', color='white', linewidth=0.5, alpha=0.5)
gl.yformatter=LATITUDE_FORMATTER
gl.ylocator = mticker.FixedLocator([-40, -20, 0, 20, 40, 60, 80])
gl.ylabels_left=True
gl = ax.gridlines(linestyle='solid', color='white', linewidth=0.5, alpha=0.5)
gl.xformatter=LONGITUDE_FORMATTER
gl.xlabels_bottom=True
ax.coastlines()

line = np.arange(1, 2.5, 0.05)
diff = plt.contourf(lon1, lat1, np.nanmean(differ,axis=0),line, cmap = 'RdBu_r', extend='both', transform=ccrs.PlateCarree(central_longitude=0))

ax.text(-0.1, 1.05, '(b)',transform=ax.transAxes,va = 'top',fontweight = 'bold',fontsize=34)

ax=fig_figure3.add_subplot(gs[row,2])
ax=plt.gca()
fig_figure3.colorbar(diff, ax, orientation='vertical').set_label(u'$\mathregular{\u0394 \u0394^{13}C}_{meso}$ (‰)')


# Figure 3c
row += 1
column = 0

ax = fig_figure3.add_subplot(gs[row, column])
ax.scatter(np.nanmean(np.nanmean(final,axis=0),axis=1), lat1, color='k', marker='o', s=20)

ax.set_xlabel(title1)
ax.set_xlim((19,24))
ax.set_xticks([19,21.5,24])
ax.set_xticklabels([19,21.5,24])
ax.set_ylim((-70,99))
ax.tick_params(axis='y', which='both', right=False, 
                left=False, labelleft=False) 
ax.tick_params(axis='x', which='both', bottom=True, 
                top=False, labelbottom=True) 

column += 1
ax = fig_figure3.add_subplot(gs[row, column], projection=ccrs.PlateCarree())


# Add lat/lon grid lines to the figure
gl = ax.gridlines(linestyle='solid', color='white', linewidth=0.5, alpha=0.5)
gl.yformatter=LATITUDE_FORMATTER
gl.ylocator = mticker.FixedLocator([-40, -20, 0, 20, 40, 60, 80])
gl.ylabels_left=True
gl = ax.gridlines(linestyle='solid', color='white', linewidth=0.5, alpha=0.5)
gl.xformatter=LONGITUDE_FORMATTER
gl.xlabels_bottom=True
ax.coastlines()

line = np.arange(16, 24, 0.1)
diff1 = plt.contourf(lon1, lat1, np.nanmean(final,axis=0),line,cmap = 'BrBG', extend='both', transform=ccrs.PlateCarree(central_longitude=0))

#plt.scatter(Leaf.Lon, Leaf.Lat, c = Leaf.D13C, cmap = 'BrBG', s=150, linewidths = 1.1, edgecolors='black')

ax.text(-0.1, 1.05, '(c)',transform=ax.transAxes,va = 'top',fontweight = 'bold',fontsize=34)

ax=fig_figure3.add_subplot(gs[row,2])
ax=plt.gca()
fig_figure3.colorbar(diff1, ax, orientation='vertical').set_label(title1)



# Figure 3d
row += 1
column = 0

ax = fig_figure3.add_subplot(gs[row, column])

ax.scatter(np.repeat(0,360), lat1, color='gray', marker='o', s=2)

ax.scatter(np.nanmean(trendD13CMatrix,axis=1), lat1, color='k', marker='o', s=20)

ax.set_xlabel(u'$\mathregular{\u0394^{13}C}$ trend')
ax.set_xlim((-0.02, 0.02))
ax.set_xticks([-0.02, 0, 0.02])
ax.set_xticklabels([-0.02, 0 ,0.02])
ax.set_ylim((-75,80))
ax.tick_params(axis='y', which='both', right=False, 
                left=False, labelleft=False) 
ax.tick_params(axis='x', which='both', bottom=True, 
                top=False, labelbottom=True) 

column += 1
ax = fig_figure3.add_subplot(gs[row, column], projection=ccrs.PlateCarree())


# Add lat/lon grid lines to the figure
gl = ax.gridlines(linestyle='solid', color='white', linewidth=0.5, alpha=0.5)
gl.yformatter=LATITUDE_FORMATTER
gl.ylocator = mticker.FixedLocator([-40, -20, 0, 20, 40, 60, 80])
gl.ylabels_left=True
gl = ax.gridlines(linestyle='solid', color='white', linewidth=0.5, alpha=0.5)
gl.xformatter=LONGITUDE_FORMATTER
gl.xlabels_bottom=True
ax.coastlines()

line = np.arange(-0.04, 0.04, 0.001)
diff2 = plt.contourf(lon1, lat1, trendD13CMatrix,line, cmap = 'BrBG', extend='both', transform=ccrs.PlateCarree(central_longitude=0))

#plt.scatter(TR.Lon, TR.Lat, c = TR.Slope, cmap = 'BrBG', s=150, linewidths = 1.1, edgecolors='black')

ax.text(-0.05, 1.05, '(d)',transform=ax.transAxes,va = 'top',fontweight = 'bold',fontsize=34)


ax=fig_figure3.add_subplot(gs[row,2])
ax=plt.gca()
fig_figure3.colorbar(diff2, ax, orientation='vertical').set_label(u'$\mathregular{\u0394^{13}C}$ trend (‰ yr$^{-1}$)')


#%%
fig_figure3.savefig('~/outputs/Figures/Figure3_final.jpg', bbox_inches='tight')
plt.close()


# # Global estimates calculation for Figure 4

# In[54]:


delt_totC13 = []
delt_totC13f = []
totcarb = []

for i in np.arange(0,38,1):  
    delt_totC13.append(np.nanmean(final[i,:,:]))
    delt_totC13f.append(np.nanmean(finalf[i,:,:]))
    totc13 = np.array(delt_totC13)
    totc13f = np.array(delt_totC13f)
    totcarb.append(np.nanmean(final2[i,:,:]))
    cvar = np.array(totcarb)


# In[56]:


## Model outputs for global run with photorespiratory and mesophyll effects (available upon request through the NERC JASMIN platform; http://www.jasmin.ac.uk/)
data3=nc.Dataset('~/outputs/2d/WFDEI_global.all.PFTs.month.2d.meso.'+str(year1)+'.'+str(year2)+'.nc')

var2=data3.variables[label2][:,:,:,0:nbpft]
var21=data3.variables[label21][:,:,:]
var22=data3.variables[label22][:,:,:]


## Conversion CO2 ppm into Pa
co2_Pa = var21*var22/1e6 
#np.mean(co2_Pa), np.min(co2_Pa), np.max(co2_Pa)

del data3

## removing negative values
var2[var2<0] = np.nan
var2[var2>60] = np.nan


# In[61]:


## Calculation yearly weighted average of leaf ci
finalci=[]
cy=0

for z in range(5):
    for i in range(456):
            if i%12==11:
                cy+=var2[:,:,i,z]              
                finalci.append(cy/12)
                cy=0
            else:
                cy+=var2[:,:,i,z]           
finalci = np.array(finalci)

del var2

## reorganising files
finalf2 = np.ma.empty((38,360,720,5))
finalf2.mask = True

finalf2[:,:,:,0] = finalci[0:38,:,:]
finalf2[:,:,:,1] = finalci[38:(38+38),:,:]
finalf2[:,:,:,2] = finalci[76:(76+38),:,:]
finalf2[:,:,:,3] = finalci[114:(114+38),:,:]
finalf2[:,:,:,4] = finalci[152:(152+38),:,:]

finalci = np.nanmean(finalf2,axis=3) 

co2_Pa = np.array(np.average(np.average(co2_Pa,axis=1),axis=0))


# In[67]:


# Calculation [CO2]atm yearly average 
def year_av(initi, final):
    final=[]
    cy=0
    
    for i in range(len(initi)):
        if i%12==11:
            cy+=initi[i]
            final.append(cy/12)
            cy=0
        else:
            cy+=initi[i]            
    return final

co2 = []
co2 = year_av(co2_Pa, co2)

co2 = np.array(co2)

co2_Pa2 = []
co2_Pa2 = year_av(co2_Pa, co2_Pa2)

co2_Pa2 = np.array(co2_Pa2)

finalci = np.transpose(finalci)


## Calculation ci/ca
var24 = finalci/co2_Pa2
var24 = np.transpose(var24)

## removing negative values
var24[var24<=0] = np.nan
var24[var24>1] = np.nan


# In[74]:


## Calculation yearly average [CO2]atm
final4=[]
cy=0

for i in range(456):
    if i%12==11:
            cy+=var21[:,:,i]
            final4.append(cy/12)
            cy=0
    else:
            cy+=var21[:,:,i]         
            
final4 = np.array(final4)

co2 = np.array(np.average(np.average(final4,axis=2),axis=1))


# In[77]:


## Calculation for figure 4

## D13C for simple discrimination with b = 30 permil

D13C_time = 4.4 + (30-4.4)*var24

## iWUE

var23 = final4/1.6*(1-var24)

## D13C with photorespiratory effect

differ2 = D13C_time - finalf


# In[84]:


## Script for Taylor diagram

#!/usr/bin/env python
# Copyright: This document has been placed in the public domain.

"""
Taylor diagram (Taylor, 2001) implementation.
Note: If you have found these software useful for your research, I would
appreciate an acknowledgment.
"""

__version__ = "Time-stamp: <2018-12-06 11:43:41 ycopin>"
__author__ = "Yannick Copin <yannick.copin@laposte.net>"

import numpy as NP
import matplotlib.pyplot as PLT


class TaylorDiagram(object):
    """
    Taylor diagram.
    Plot model standard deviation and correlation to reference (data)
    sample in a single-quadrant polar plot, with r=stddev and
    theta=arccos(correlation).
    """

    def __init__(self, refstd,
                 fig=None, rect=111, label='_', srange=(0, 1.5), extend=False):
        """
        Set up Taylor diagram axes, i.e. single quadrant polar
        plot, using `mpl_toolkits.axisartist.floating_axes`.
        Parameters:
        * refstd: reference standard deviation to be compared to
        * fig: input Figure or None
        * rect: subplot definition
        * label: reference label
        * srange: stddev axis extension, in units of *refstd*
        * extend: extend diagram to negative correlations
        """

        from matplotlib.projections import PolarAxes
        import mpl_toolkits.axisartist.floating_axes as FA
        import mpl_toolkits.axisartist.grid_finder as GF

        self.refstd = refstd            # Reference standard deviation

        tr = PolarAxes.PolarTransform()

        # Correlation labels
        rlocs = NP.array([0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1])
        if extend:
            # Diagram extended to negative correlations
            self.tmax = NP.pi
            rlocs = NP.concatenate((-rlocs[:0:-1], rlocs))
        else:
            # Diagram limited to positive correlations
            self.tmax = NP.pi/2
        tlocs = NP.arccos(rlocs)        # Conversion to polar angles
        gl1 = GF.FixedLocator(tlocs)    # Positions
        tf1 = GF.DictFormatter(dict(zip(tlocs, map(str, rlocs))))

        # Standard deviation axis extent (in units of reference stddev)
        #self.smin = srange[0] * self.refstd
        #self.smax = srange[1] * self.refstd
        self.smin = srange[0] * self.refstd
        self.smax = srange[1] * self.refstd

        ghelper = FA.GridHelperCurveLinear(
            tr,
            extremes=(0, self.tmax, self.smin, self.smax),
            grid_locator1=gl1, tick_formatter1=tf1)

        if fig is None:
            fig = PLT.figure()

        ax = FA.FloatingSubplot(fig, rect, grid_helper=ghelper)
        fig.add_subplot(ax)

        # Adjust axes
        ax.axis["top"].set_axis_direction("bottom")   # "Angle axis"
        ax.axis["top"].toggle(ticklabels=True, label=True)
        ax.axis["top"].major_ticklabels.set_axis_direction("top")
        ax.axis["top"].label.set_axis_direction("top")
        ax.axis["top"].label.set_text("Correlation")

        ax.axis["left"].major_ticklabels.set_visible(False)  # "X axis"
        ax.axis["left"].set_axis_direction("bottom")  # "X axis"
        ax.axis["left"].label.set_text("Standard deviation")
        #ax.set_xticks([1,2.5,4.5,6.5,9])
        
        ax.axis["right"].set_axis_direction("top")    # "Y-axis"
        ax.axis["right"].toggle(ticklabels=True)
        ax.axis["right"].major_ticklabels.set_axis_direction(
            "bottom" if extend else "left")

        if self.smin:
            ax.axis["bottom"].toggle(ticklabels=False, label=False)
        else:
            ax.axis["bottom"].set_visible(False)          # Unused

        self._ax = ax                   # Graphical axes
        self.ax = ax.get_aux_axes(tr)   # Polar coordinates

        # Add reference point and stddev contour
        l, = self.ax.plot([0], self.refstd, 'k*',
                          ls='', ms=25, label=label)
        t = NP.linspace(0, self.tmax)
        r = NP.zeros_like(t) + self.refstd
        self.ax.plot(t, r, 'k--', label='_')

        # Collect sample points for latter use (e.g. legend)
        self.samplePoints = [l]

    def add_sample(self, stddev, corrcoef, *args, **kwargs):
        """
        Add sample (*stddev*, *corrcoeff*) to the Taylor
        diagram. *args* and *kwargs* are directly propagated to the
        `Figure.plot` command.
        """

        l, = self.ax.plot(NP.arccos(corrcoef), stddev,
                          *args, **kwargs)  # (theta, radius)
        self.samplePoints.append(l)

        return l

    def add_grid(self, *args, **kwargs):
        """Add a grid."""

        self._ax.grid(*args, **kwargs)

    def add_contours(self, levels=5, **kwargs):
        """
        Add constant centered RMS difference contours, defined by *levels*.
        """

        rs, ts = NP.meshgrid(NP.linspace(self.smin, self.smax),
                             NP.linspace(0, self.tmax))
        # Compute centered RMS difference
        rms = NP.sqrt(self.refstd**2 + rs**2 - 2*self.refstd*rs*NP.cos(ts))

        contours = self.ax.contour(ts, rs, rms, levels, **kwargs)

        return contours


# In[85]:


## Taylor diagram

# Reference dataset

data = leaf_obs_mod.Obs[(leaf_obs_mod.Discr_mod == "simple") & (leaf_obs_mod.Stom_mod == "Jacobs")]
refstd = data.std(ddof=1)           # Reference standard deviation

# Generate models
m1 = leaf_obs_mod[(leaf_obs_mod.Discr_mod == "simple") & (leaf_obs_mod.Stom_mod == "Jacobs") & (np.invert(np.isnan(leaf_obs_mod['D13C_6-8'])))]
m2 = leaf_obs_mod[(leaf_obs_mod.Discr_mod == "photoresp") & (leaf_obs_mod.Stom_mod == "Jacobs") & (np.invert(np.isnan(leaf_obs_mod['D13C_6-8'])))]
m3 = leaf_obs_mod[(leaf_obs_mod.Discr_mod == "simple") & (leaf_obs_mod.Stom_mod == "Leuning") & (np.invert(np.isnan(leaf_obs_mod['D13C_6-8'])))]
m4 = leaf_obs_mod[(leaf_obs_mod.Discr_mod == "photoresp") & (leaf_obs_mod.Stom_mod == "Leuning") & (np.invert(np.isnan(leaf_obs_mod['D13C_6-8'])))]
m5 = leaf_obs_mod[(leaf_obs_mod.Discr_mod == "simple") & (leaf_obs_mod.Stom_mod == "Medlyn") & (np.invert(np.isnan(leaf_obs_mod['D13C_6-8'])))]
m6 = leaf_obs_mod[(leaf_obs_mod.Discr_mod == "photoresp") & (leaf_obs_mod.Stom_mod == "Medlyn") & (np.invert(np.isnan(leaf_obs_mod['D13C_6-8'])))]
m7 = leaf_obs_mod[(leaf_obs_mod.Discr_mod == "simple") & (leaf_obs_mod.Stom_mod == "Prentice") & (np.invert(np.isnan(leaf_obs_mod['D13C_6-8'])))]
m8 = leaf_obs_mod[(leaf_obs_mod.Discr_mod == "photoresp") & (leaf_obs_mod.Stom_mod == "Prentice") & (np.invert(np.isnan(leaf_obs_mod['D13C_6-8'])))]
m9 = leaf_obs_mod[(leaf_obs_mod.Discr_mod == "photo+meso") & (leaf_obs_mod.Stom_mod == "Prentice") & (np.invert(np.isnan(leaf_obs_mod['D13C_6-8'])))]
  
    
data_all = [data, m1['D13C_6-8'], m2['D13C_6-8'], m3['D13C_6-8'], m4['D13C_6-8'], m5['D13C_6-8'], m6['D13C_6-8'], m7['D13C_6-8'], m8['D13C_6-8'], m9['D13C_6-8']]

names = ["JAC_simple", "JAC_photo", "LEU_simple", "LEU_photo","MED_simple", "MED_photo", "PREN_simple", "PREN_photo", "PREN_photomeso"]

# Compute stddev and correlation coefficient of models

samples = NP.array([ [m['D13C_6-8'].std(ddof=1), NP.corrcoef(m.Obs, m['D13C_6-8'])[0, 1]]
                         for m in (m1, m2, m3, m4, m5, m6, m7, m8, m9) ])


colors2 = PLT.matplotlib.cm.jet(NP.linspace(0, 1, (len(samples)+1)))


names = ["OBS","JAC_simple", "JAC_photo", "LEU_simple", "LEU_photo","MED_simple", "MED_photo", "PREN_simple", "PREN_photo", "PREN_photomeso"]

data = pd.DataFrame(np.array(data))
JAC_simple = pd.DataFrame(np.array(m1['D13C_6-8']))
JAC_photo = pd.DataFrame(np.array(m2['D13C_6-8']))
LEU_simple = pd.DataFrame(np.array(m3['D13C_6-8']))
LEU_photo = pd.DataFrame(np.array(m4['D13C_6-8']))
MED_simple = pd.DataFrame(np.array(m5['D13C_6-8']))
MED_photo = pd.DataFrame(np.array(m6['D13C_6-8']))
PREN_simple = pd.DataFrame(np.array(m7['D13C_6-8']))
PREN_photo = pd.DataFrame(np.array(m8['D13C_6-8']))
PREN_photomeso = pd.DataFrame(np.array(m9['D13C_6-8']))

data3 = {'OBS': data,'JAC_simple': JAC_simple, 'JAC_photo': JAC_photo, 'LEU_simple': LEU_simple, 'LEU_photo': LEU_photo, 'MED_simple': MED_simple, 'MED_photo': MED_photo, 'PREN_simple': PREN_simple, 'PREN_photo': PREN_photo,'PREN_photomeso': PREN_photomeso} 
df = pd.concat(data3,axis = 1)


## Consider only TR slope that are significant
TR.Slope[TR.Pval>0.05] = 0


# # Figure 1

# In[107]:


## Figure 1

# Set up the subplot figure

fig_figure1 = plt.figure(1, figsize=(25,20))
column = 0
row = 0

mpl.rcParams['xtick.direction'] = 'out'
mpl.rcParams['ytick.direction'] = 'out'
mpl.rcParams['xtick.top'] = True 
mpl.rcParams['ytick.right'] = True
params = {
    'lines.linewidth':3,
    'axes.facecolor':'white',
    'xtick.color':'k',
    'ytick.color':'k',
    'axes.labelsize': 34,
    'xtick.labelsize':34,
    'ytick.labelsize':34,
    'font.size':34,
    'text.usetex': False,
    "svg.fonttype": 'none'
}
plt.rcParams.update(params)


# Figure 1a

gs = gspec.GridSpec(2, 3, figure=fig_figure1, width_ratios=[1, 0.25,1], hspace=0.2)

ax = fig_figure1.add_subplot(gs[row, column])

box = plt.boxplot(data_all, patch_artist=True)
 
for patch, color in zip(box['boxes'], colors2):
    patch.set_facecolor(color)

ax.set_ylabel(u'$\mathregular{\u0394^{13}C}$ (‰)')
ax.set_ylim((12,28))

ax.axhline(y=np.median(df.OBS), color='gray', linewidth=1)
ax.set_xticks([1,2.5,4.5,6.5,9])

ax.set_xticklabels(["OBS","JAC", "LEU","MED", "PREN"], size=22)

ax.text(-0.05, 1.15, '(a)',transform=ax.transAxes,va = 'top',fontweight = 'bold',fontsize=34)



# Figure 1b
column += 1
fig = PLT.figure(1,figsize=(15, 15))

dia = TaylorDiagram(refstd, fig=fig, rect=222, label="Reference",
                        srange=(0, 1.1))

column += 1

colors = PLT.matplotlib.cm.jet(NP.linspace(0, 1, len(samples)))

names = ["JAC_simple", "JAC_photo", "LEU_simple", "LEU_photo","MED_simple", "MED_photo", "PREN_simple", "PREN_photo", "PREN_photomeso"]


    # Add the models to Taylor diagram
for i, (stddev, corrcoef) in enumerate(samples):
    dia.add_sample(stddev, corrcoef,
                      marker='$%d$' % (i+1), ms=20, ls='',
                       mfc=colors[i], mec=colors[i],
                       label=np.array(names[i]))

fig_figure1.legend(dia.samplePoints,
               [ p.get_label() for p in dia.samplePoints ],
               numpoints=1, prop=dict(size=20), loc=(0.2,0.39),ncol = 4)

    # Add grid
dia.add_grid()

    # Add RMS contours, and label them
contours = dia.add_contours(colors='0.5')
PLT.clabel(contours, inline=1, fontsize=10, fmt='%.2f')

ax.text(1.28, 1.15, '(b)',transform=ax.transAxes,va = 'top',fontweight = 'bold',fontsize=34)



## Figure 1c

gs = gspec.GridSpec(2, 4, figure=fig_figure1, width_ratios=[0.0, 1, 0.05, 0.1], hspace=0.2)
# set rows and column
column = 1

row =+ 1


ax = fig_figure1.add_subplot(gs[row, column], projection=ccrs.PlateCarree())

# Add lat/lon grid lines to the figure
gl = ax.gridlines(linestyle='solid', color='white', linewidth=0.5, alpha=0.5)
gl.yformatter=LATITUDE_FORMATTER
gl.ylocator = mticker.FixedLocator([-60, -40, -20, 0, 20, 40, 60, 80])
gl.ylabels_left=True
gl = ax.gridlines(linestyle='solid', color='white', linewidth=0.5, alpha=0.5)
gl.xformatter=LONGITUDE_FORMATTER
gl.xlabels_bottom=True
ax.coastlines()

plt.scatter(Leaf.Lon, Leaf.Lat, c = Leaf.D13C, cmap = 'BrBG', s=150, linewidths = 1.1, edgecolors='black')

ax.text(-0.07, 1.15, '(c)',transform=ax.transAxes,va = 'top',fontweight = 'bold',fontsize=34)

ax=fig_figure1.add_subplot(gs[row,2])
ax=plt.gca()
fig_figure1.colorbar(diff1, ax, orientation='vertical').set_label(title1)


#%%
fig_figure1.savefig('~/outputs/Figures/Figure1_final.jpg', bbox_inches='tight')
plt.close()



# # Figure 2

# In[115]:


## Figure 2

# Set up the subplot figure

fig_figure2 = plt.figure(1, figsize=(25,20))
column = 0
row = 0

mpl.rcParams['xtick.direction'] = 'out'
mpl.rcParams['ytick.direction'] = 'out'
mpl.rcParams['xtick.top'] = True 
mpl.rcParams['ytick.right'] = True
params = {
    'lines.linewidth':3,
    'axes.facecolor':'white',
    'xtick.color':'k',
    'ytick.color':'k',
    'axes.labelsize': 34,
    'xtick.labelsize':34,
    'ytick.labelsize':34,
    'font.size':34,
    'text.usetex': False,
    "svg.fonttype": 'none'
}
plt.rcParams.update(params)


## Figure 2a

gs = gspec.GridSpec(2, 3, figure=fig_figure2, width_ratios=[1, 0.05, 0.1], hspace=0.2)
# set rows and column
column = 0


ax = fig_figure2.add_subplot(gs[row, column], projection=ccrs.PlateCarree())

# Add lat/lon grid lines to the figure
gl = ax.gridlines(linestyle='solid', color='white', linewidth=0.5, alpha=0.5)
gl.yformatter=LATITUDE_FORMATTER
gl.ylocator = mticker.FixedLocator([-60, -40, -20, 0, 20, 40, 60, 80])
gl.ylabels_left=True
gl = ax.gridlines(linestyle='solid', color='white', linewidth=0.5, alpha=0.5)
gl.xformatter=LONGITUDE_FORMATTER
gl.xlabels_bottom=True
ax.coastlines()

plt.scatter(TR.Lon, TR.Lat, c = TR.Slope, cmap = 'BrBG', s=150, linewidths = 1.1, edgecolors='black')

ax.text(-0.03, 1.15, '(a)',transform=ax.transAxes,va = 'top',fontweight = 'bold',fontsize=34)

column += 1
ax=fig_figure2.add_subplot(gs[row,column])
ax=plt.gca()
fig_figure2.colorbar(diff2, ax, orientation='vertical').set_label(u'$\mathregular{\u0394^{13}C}$ trend (‰ yr$^{-1}$)')
   
    
# Figure 2b

gs = gspec.GridSpec(2, 4, figure=fig_figure2, width_ratios=[1, 0.1, 1, 0.05], hspace=0.2)
column = 0
row =+ 1

ax = fig_figure2.add_subplot(gs[row, column])

ax.scatter(TR_slope_IAV_obs_mod.Mean[(TR_slope_IAV_obs_mod.Type == "TR") & (TR_slope_IAV_obs_mod.Pval>=0.05)], TR_slope_IAV_obs_mod.Slope[(TR_slope_IAV_obs_mod.Type == "TR")  & (TR_slope_IAV_obs_mod.Pval>=0.05)], color='white', edgecolor='black', marker='o', s=100)
ax.scatter(TR_slope_IAV_obs_mod.Mean[(TR_slope_IAV_obs_mod.Type == "Jacobs") & (TR_slope_IAV_obs_mod.Pval>=0.05)], TR_slope_IAV_obs_mod.Slope[(TR_slope_IAV_obs_mod.Type == "Jacobs") & (TR_slope_IAV_obs_mod.Pval>=0.05)], color='white', edgecolor=colors2[2], marker='o', s=100)
ax.scatter(TR_slope_IAV_obs_mod.Mean[(TR_slope_IAV_obs_mod.Type == "Leuning") & (TR_slope_IAV_obs_mod.Pval>=0.05)], TR_slope_IAV_obs_mod.Slope[(TR_slope_IAV_obs_mod.Type == "Leuning")  & (TR_slope_IAV_obs_mod.Pval>=0.05)], color='white', edgecolor=colors2[3], marker='o', s=100)
ax.scatter(TR_slope_IAV_obs_mod.Mean[(TR_slope_IAV_obs_mod.Type == "Medlyn") & (TR_slope_IAV_obs_mod.Pval>=0.05)], TR_slope_IAV_obs_mod.Slope[(TR_slope_IAV_obs_mod.Type == "Medlyn") & (TR_slope_IAV_obs_mod.Pval>=0.05)], color='white', edgecolor=colors2[6], marker='o', s=100)
ax.scatter(TR_slope_IAV_obs_mod.Mean[(TR_slope_IAV_obs_mod.Type == "Prentice") & (TR_slope_IAV_obs_mod.Pval>=0.05)], TR_slope_IAV_obs_mod.Slope[(TR_slope_IAV_obs_mod.Type == "Prentice") & (TR_slope_IAV_obs_mod.Pval>=0.05)], color='white', edgecolor=colors2[8], marker='o', s=100)

ax.scatter(TR_slope_IAV_obs_mod.Mean[(TR_slope_IAV_obs_mod.Type == "TR") & (TR_slope_IAV_obs_mod.Pval<0.05)], TR_slope_IAV_obs_mod.Slope[(TR_slope_IAV_obs_mod.Type == "TR")  & (TR_slope_IAV_obs_mod.Pval<0.05)], color='black', marker='o', s=150,alpha=0.3)
ax.scatter(TR_slope_IAV_obs_mod.Mean[(TR_slope_IAV_obs_mod.Type == "Jacobs") & (TR_slope_IAV_obs_mod.Pval<0.05)], TR_slope_IAV_obs_mod.Slope[(TR_slope_IAV_obs_mod.Type == "Jacobs") & (TR_slope_IAV_obs_mod.Pval<0.05)], color=colors2[2], marker='o', s=150,alpha=0.3)
ax.scatter(TR_slope_IAV_obs_mod.Mean[(TR_slope_IAV_obs_mod.Type == "Leuning") & (TR_slope_IAV_obs_mod.Pval<0.05)], TR_slope_IAV_obs_mod.Slope[(TR_slope_IAV_obs_mod.Type == "Leuning")  & (TR_slope_IAV_obs_mod.Pval<0.05)], color=colors2[3], marker='o', s=150,alpha=0.3)
ax.scatter(TR_slope_IAV_obs_mod.Mean[(TR_slope_IAV_obs_mod.Type == "Medlyn") & (TR_slope_IAV_obs_mod.Pval<0.05)], TR_slope_IAV_obs_mod.Slope[(TR_slope_IAV_obs_mod.Type == "Medlyn") & (TR_slope_IAV_obs_mod.Pval<0.05)], color=colors2[6],  marker='o', s=150,alpha=0.3)
ax.scatter(TR_slope_IAV_obs_mod.Mean[(TR_slope_IAV_obs_mod.Type == "Prentice") & (TR_slope_IAV_obs_mod.Pval<0.05)], TR_slope_IAV_obs_mod.Slope[(TR_slope_IAV_obs_mod.Type == "Prentice") & (TR_slope_IAV_obs_mod.Pval<0.05)], color=colors2[8],  marker='o', s=150,alpha=0.3)

ax.scatter(np.mean(TR_slope_IAV_obs_mod.Mean[(TR_slope_IAV_obs_mod.Type == "TR")]), np.mean(TR_slope_IAV_obs_mod.Slope[(TR_slope_IAV_obs_mod.Type == "TR")]),color="black", marker='o', s=250)
ax.scatter(np.mean(TR_slope_IAV_obs_mod.Mean[(TR_slope_IAV_obs_mod.Type == "Jacobs")]), np.mean(TR_slope_IAV_obs_mod.Slope[(TR_slope_IAV_obs_mod.Type == "Jacobs")]),color=colors2[2], marker='o', s=250)
ax.scatter(np.mean(TR_slope_IAV_obs_mod.Mean[(TR_slope_IAV_obs_mod.Type == "Leuning")]), np.mean(TR_slope_IAV_obs_mod.Slope[(TR_slope_IAV_obs_mod.Type == "Leuning")]),color=colors2[3], marker='o', s=250)
ax.scatter(np.mean(TR_slope_IAV_obs_mod.Mean[(TR_slope_IAV_obs_mod.Type == "Medlyn")]), np.mean(TR_slope_IAV_obs_mod.Slope[(TR_slope_IAV_obs_mod.Type == "Medlyn")]),color=colors2[6], marker='o', s=250)
ax.scatter(np.mean(TR_slope_IAV_obs_mod.Mean[(TR_slope_IAV_obs_mod.Type == "Prentice")]), np.mean(TR_slope_IAV_obs_mod.Slope[(TR_slope_IAV_obs_mod.Type == "Prentice")]),color=colors2[8], marker='o', s=250)


ax.errorbar(np.mean(TR_slope_IAV_obs_mod.Mean[(TR_slope_IAV_obs_mod.Type == "TR")]), np.mean(TR_slope_IAV_obs_mod.Slope[(TR_slope_IAV_obs_mod.Type == "TR")]), 
            xerr = np.std(TR_slope_IAV_obs_mod[(TR_slope_IAV_obs_mod.Type == "TR")].Mean), 
            yerr = np.std(TR_slope_IAV_obs_mod[(TR_slope_IAV_obs_mod.Type == "TR")].Slope),
           capsize = 1,elinewidth=10,color="black")

ax.errorbar(np.mean(TR_slope_IAV_obs_mod.Mean[(TR_slope_IAV_obs_mod.Type == "Jacobs")]), np.mean(TR_slope_IAV_obs_mod.Slope[(TR_slope_IAV_obs_mod.Type == "Jacobs")]), 
            xerr = np.std(TR_slope_IAV_obs_mod[(TR_slope_IAV_obs_mod.Type == "Jacobs")].Mean), 
            yerr = np.std(TR_slope_IAV_obs_mod[(TR_slope_IAV_obs_mod.Type == "Jacobs")].Slope),
           capsize = 1,elinewidth=10,color=colors2[2])

ax.errorbar(np.mean(TR_slope_IAV_obs_mod.Mean[(TR_slope_IAV_obs_mod.Type == "Leuning")]), np.mean(TR_slope_IAV_obs_mod.Slope[(TR_slope_IAV_obs_mod.Type == "Leuning")]), 
            xerr = np.std(TR_slope_IAV_obs_mod[(TR_slope_IAV_obs_mod.Type == "Leuning")].Mean), 
            yerr = np.std(TR_slope_IAV_obs_mod[(TR_slope_IAV_obs_mod.Type == "Leuning")].Slope),
           capsize = 1,elinewidth=10,color=colors2[3])

ax.errorbar(np.mean(TR_slope_IAV_obs_mod.Mean[(TR_slope_IAV_obs_mod.Type == "Medlyn")]), np.mean(TR_slope_IAV_obs_mod.Slope[(TR_slope_IAV_obs_mod.Type == "Medlyn")]), 
            xerr = np.std(TR_slope_IAV_obs_mod[(TR_slope_IAV_obs_mod.Type == "Medlyn")].Mean), 
            yerr = np.std(TR_slope_IAV_obs_mod[(TR_slope_IAV_obs_mod.Type == "Medlyn")].Slope),
           capsize = 1,elinewidth=10,color=colors2[6])

ax.errorbar(np.mean(TR_slope_IAV_obs_mod.Mean[(TR_slope_IAV_obs_mod.Type == "Prentice")]), np.mean(TR_slope_IAV_obs_mod.Slope[(TR_slope_IAV_obs_mod.Type == "Prentice")]), 
            xerr = np.std(TR_slope_IAV_obs_mod[(TR_slope_IAV_obs_mod.Type == "Prentice")].Mean), 
            yerr = np.std(TR_slope_IAV_obs_mod[(TR_slope_IAV_obs_mod.Type == "Prentice")].Slope),
           capsize = 1,elinewidth=10,color=colors2[8])

    
    
ax.set_xlabel(u'Mean $\mathregular{\u0394^{13}C}$ (‰)')
ax.set_xlim((15,27))

ax.set_ylabel(u'Slope $\mathregular{\u0394^{13}C}$ (‰ yr$^{-1}$)')
ax.set_ylim((-0.08,0.08))

ax.set_xticks([16,18,20,22,24,26])
ax.set_xticklabels([16,18,20,22,24,26])

ax.set_yticks([-0.08,-0.06,-0.04,-0.02,0,0.02,0.04,0.06,0.08])
ax.set_yticklabels([-0.08,-0.06,-0.04,-0.02,0,0.02,0.04,0.06,0.08])

ax.text(-0.05, 1.15, '(b)',transform=ax.transAxes,va = 'top',fontweight = 'bold',fontsize=34)



# Figure 2c

column += 2

ax = fig_figure2.add_subplot(gs[row, column])

ax.scatter(TR_slope_IAV_obs_mod.Mean[(TR_slope_IAV_obs_mod.Type == "TR") ], TR_slope_IAV_obs_mod.IAV[(TR_slope_IAV_obs_mod.Type == "TR") ], color='black', marker='o', s=100,alpha=0.3)
ax.scatter(TR_slope_IAV_obs_mod.Mean[(TR_slope_IAV_obs_mod.Type == "Jacobs") ], TR_slope_IAV_obs_mod.IAV[(TR_slope_IAV_obs_mod.Type == "Jacobs") ], color=colors2[1], marker='o', s=100,alpha=0.3)
ax.scatter(TR_slope_IAV_obs_mod.Mean[(TR_slope_IAV_obs_mod.Type == "Leuning")], TR_slope_IAV_obs_mod.IAV[(TR_slope_IAV_obs_mod.Type == "Leuning") ], color=colors2[3], marker='o', s=100,alpha=0.3)
ax.scatter(TR_slope_IAV_obs_mod.Mean[(TR_slope_IAV_obs_mod.Type == "Medlyn") ], TR_slope_IAV_obs_mod.IAV[(TR_slope_IAV_obs_mod.Type == "Medlyn")], color=colors2[6], marker='o', s=100,alpha=0.3)
ax.scatter(TR_slope_IAV_obs_mod.Mean[(TR_slope_IAV_obs_mod.Type == "Prentice")], TR_slope_IAV_obs_mod.IAV[(TR_slope_IAV_obs_mod.Type == "Prentice") ], color=colors2[8], marker='o', s=100,alpha=0.3)

ax.scatter(np.mean(TR_slope_IAV_obs_mod.Mean[(TR_slope_IAV_obs_mod.Type == "TR")]), np.mean(TR_slope_IAV_obs_mod.IAV[(TR_slope_IAV_obs_mod.Type == "TR")]),color="black", marker='o', s=250)
ax.scatter(np.mean(TR_slope_IAV_obs_mod.Mean[(TR_slope_IAV_obs_mod.Type == "Jacobs")]), np.mean(TR_slope_IAV_obs_mod.IAV[(TR_slope_IAV_obs_mod.Type == "Jacobs")]),color=colors2[2], marker='o', s=250)
ax.scatter(np.mean(TR_slope_IAV_obs_mod.Mean[(TR_slope_IAV_obs_mod.Type == "Leuning")]), np.mean(TR_slope_IAV_obs_mod.IAV[(TR_slope_IAV_obs_mod.Type == "Leuning")]),color=colors2[3], marker='o', s=250)
ax.scatter(np.mean(TR_slope_IAV_obs_mod.Mean[(TR_slope_IAV_obs_mod.Type == "Medlyn")]), np.mean(TR_slope_IAV_obs_mod.IAV[(TR_slope_IAV_obs_mod.Type == "Medlyn")]),color=colors2[6], marker='o', s=250)
ax.scatter(np.mean(TR_slope_IAV_obs_mod.Mean[(TR_slope_IAV_obs_mod.Type == "Prentice")]), np.mean(TR_slope_IAV_obs_mod.IAV[(TR_slope_IAV_obs_mod.Type == "Prentice")]),color=colors2[8], marker='o', s=250)


ax.errorbar(np.mean(TR_slope_IAV_obs_mod.Mean[(TR_slope_IAV_obs_mod.Type == "TR")]), np.mean(TR_slope_IAV_obs_mod.IAV[(TR_slope_IAV_obs_mod.Type == "TR")]), 
            xerr = np.std(TR_slope_IAV_obs_mod[(TR_slope_IAV_obs_mod.Type == "TR")].Mean), 
            yerr = np.std(TR_slope_IAV_obs_mod[(TR_slope_IAV_obs_mod.Type == "TR")].IAV),
           capsize = 1,elinewidth=10,color="black")

ax.errorbar(np.mean(TR_slope_IAV_obs_mod.Mean[(TR_slope_IAV_obs_mod.Type == "Jacobs")]), np.mean(TR_slope_IAV_obs_mod.IAV[(TR_slope_IAV_obs_mod.Type == "Jacobs")]), 
            xerr = np.std(TR_slope_IAV_obs_mod[(TR_slope_IAV_obs_mod.Type == "Jacobs")].Mean), 
            yerr = np.std(TR_slope_IAV_obs_mod[(TR_slope_IAV_obs_mod.Type == "Jacobs")].IAV),
           capsize = 1,elinewidth=10,color=colors2[2])

ax.errorbar(np.mean(TR_slope_IAV_obs_mod.Mean[(TR_slope_IAV_obs_mod.Type == "Leuning")]), np.mean(TR_slope_IAV_obs_mod.IAV[(TR_slope_IAV_obs_mod.Type == "Leuning")]), 
            xerr = np.std(TR_slope_IAV_obs_mod[(TR_slope_IAV_obs_mod.Type == "Leuning")].Mean), 
            yerr = np.std(TR_slope_IAV_obs_mod[(TR_slope_IAV_obs_mod.Type == "Leuning")].IAV),
           capsize = 1,elinewidth=10,color=colors2[3])

ax.errorbar(np.mean(TR_slope_IAV_obs_mod.Mean[(TR_slope_IAV_obs_mod.Type == "Medlyn")]), np.mean(TR_slope_IAV_obs_mod.IAV[(TR_slope_IAV_obs_mod.Type == "Medlyn")]), 
            xerr = np.std(TR_slope_IAV_obs_mod[(TR_slope_IAV_obs_mod.Type == "Medlyn")].Mean), 
            yerr = np.std(TR_slope_IAV_obs_mod[(TR_slope_IAV_obs_mod.Type == "Medlyn")].IAV),
           capsize = 1,elinewidth=10,color=colors2[6])

ax.errorbar(np.mean(TR_slope_IAV_obs_mod.Mean[(TR_slope_IAV_obs_mod.Type == "Prentice")]), np.mean(TR_slope_IAV_obs_mod.IAV[(TR_slope_IAV_obs_mod.Type == "Prentice")]), 
            xerr = np.std(TR_slope_IAV_obs_mod[(TR_slope_IAV_obs_mod.Type == "Prentice")].Mean), 
            yerr = np.std(TR_slope_IAV_obs_mod[(TR_slope_IAV_obs_mod.Type == "Prentice")].IAV),
           capsize = 1,elinewidth=10,color=colors2[8])

    
ax.legend(labels= ['OBS','JAC','LEU','MED','PREN'],numpoints=1, prop=dict(size=28), loc=(-0.9,-0.3),ncol=5)

ax.set_xlabel(u'Mean $\mathregular{\u0394^{13}C}$ (‰)')
ax.set_xlim((15,27))

ax.set_ylabel(u'IAV $\mathregular{\u0394^{13}C}$ (‰)')
ax.set_ylim((0,1.2))

ax.set_xticks([16,18,20,22,24,26])
ax.set_xticklabels([16,18,20,22,24,26])

ax.set_yticks([0,0.2,0.4,0.6,0.8,1.0,1.2])
ax.set_yticklabels([0,0.2,0.4,0.6,0.8,1.0,1.2])


ax.text(-0.06, 1.15, '(c)',transform=ax.transAxes,va = 'top',fontweight = 'bold',fontsize=34)


#%%         
fig_figure2.savefig('~/outputs/Figures/Figure2_final.jpg', bbox_inches='tight')
plt.close()



# # Figure 4

# In[95]:


cica_ave = []
iWUE_ave = []
D13C_ave = []

for i in np.arange(0,38,1):  
    cica_ave.append(np.nanmean(var24[i,:,:]))
    totcica = np.array(cica_ave)
    iWUE_ave.append(np.nanmean(var23[i,:,:]))
    totiWUE = np.array(iWUE_ave)    
    D13C_ave.append(np.nanmean(D13C_time[i,:,:]))
    totD13C = np.array(D13C_ave)  


# In[125]:


color = ['darkblue', 'dodgerblue', '#80b1d3', 'darkcyan', '#8dd3c7', 'darkseagreen', 'darkgreen', 'olive', 'gold', 
         'orange', 'peachpuff', '#fb8072', 'red', 'hotpink', '#fccde5','#bebada' ]


# In[127]:


## Figure 4

# Set up the subplot figure

fig_figure4 = plt.figure(1, figsize=(30,20))

mpl.rcParams['xtick.direction'] = 'out'
mpl.rcParams['ytick.direction'] = 'out'
mpl.rcParams['xtick.top'] = True 
mpl.rcParams['ytick.right'] = True
params = {
    'lines.linewidth':3,
    'axes.facecolor':'white',
    'xtick.color':'k',
    'ytick.color':'k',
    'axes.labelsize': 34,
    'xtick.labelsize':34,
    'ytick.labelsize':34,
    'font.size':34,
    'text.usetex': False,
    "svg.fonttype": 'none'
}
plt.rcParams.update(params)


# Figure 4a
gs = gspec.GridSpec(3, 2, figure=fig_figure4, hspace=0.2,  width_ratios=[1, 0.47])
column = 0
row = 0

ax = fig_figure4.add_subplot(gs[row, column])

ax.plot(year, (totc13-np.average(totc13)), color=color[0])
ax.plot(year, (totc13f-np.average(totc13f)), color=color[1])
ax.plot(year, (totD13C-np.average(totD13C)), color=color[3])

ax.plot(year, resultff.intercept + resultff.slope*year, color=color[3], label='fitted line', ls='dotted')
ax.plot(year, resultf.intercept + resultf.slope*year, color=color[1], label='fitted line', ls='dotted')
ax.plot(year, result.intercept + result.slope*year, color=color[0], label='fitted line', ls='dotted')


ax.set_ylabel(u'Standardized $\mathregular{\u0394^{13}C}$ (‰)')
ax.set_ylim((-0.11,0.11))
ax.set_xlim((1978,2017))
ax.text(-0.05, 1.15, '(a)',transform=ax.transAxes,va = 'top',fontweight = 'bold',fontsize=34)

ax.text(0.05, 0.95, u'$\mathregular{\u0394^{13}C_{simple}}$ = ' f'{resultff.slope:.4f} ± {resultff.stderr:.4f} (p < 0.001)', color=color[3],fontweight = 'bold',transform=ax.transAxes,va = 'top',fontsize=20)
ax.text(0.55, 0.95, u'$\mathregular{\u0394^{13}C_{photo}}$ = ' f'{resultf.slope:.4f} ± {resultf.stderr:.4f} (p = 0.14)', color=color[1],fontweight = 'bold',transform=ax.transAxes,va = 'top',fontsize=20)
ax.text(0.38, 0.15, u'$\mathregular{\u0394^{13}C_{photo+meso}}$ = ' f'{result.slope:.4f} ± {result.stderr:.4f} (p = 0.43)', color=color[0],fontweight = 'bold',transform=ax.transAxes,va = 'top',fontsize=20)


# Figure 4b
gs = gspec.GridSpec(3, 3, figure=fig_figure4, hspace=0.6, wspace=0.3)
column = 0
row += 1

ax = fig_figure4.add_subplot(gs[row, column])

ax.plot(year, totcica, color=color[0])

ax.plot(year, resultcica.intercept + resultcica.slope*year, color=color[0], label='fitted line', ls='dotted')


ax.set_ylabel(title24)
ax.set_ylim((0.755,0.764))
ax.set_xlim((1978,2017))
ax.text(-0.12, 1.15, '(b)',transform=ax.transAxes,va = 'top',fontweight = 'bold',fontsize=34)

ax.text(0.05, 0.95, f'{resultcica.slope:.4f} ± {resultcica.stderr:.4f} yr$^-$$^1$ (p < 0.001)',fontweight = 'bold',transform=ax.transAxes,va = 'top',fontsize=20)


# Figure 4c
column=+1
ax = fig_figure4.add_subplot(gs[row, column])

ax.plot(year, totiWUE, color=color[0])

ax.plot(year, resultiWUE.intercept + resultiWUE.slope*year, color=color[0], label='fitted line', ls='dotted')


ax.set_ylabel(title23)
ax.set_ylim((49,64))
ax.set_xlim((1978,2017))
ax.text(-0.12, 1.15, '(c)',transform=ax.transAxes,va = 'top',fontweight = 'bold',fontsize=34)

ax.text(0.02, 0.95, f' {resultiWUE.slope:.2f} ± {resultiWUE.stderr:.2f}' u' $\mathregular{\u03BC}$mol mol$^-$$^1$ yr$^-$$^1$',fontweight = 'bold',transform=ax.transAxes,va = 'top',fontsize=20)
ax.text(0.05, 0.85, f'(p < 0.001)',fontweight = 'bold',transform=ax.transAxes,va = 'top',fontsize=20)


#%%
fig_figure4.savefig('~/outputs/Figures/Figure4_final.jpg', bbox_inches='tight')
plt.close()


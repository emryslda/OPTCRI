#!/usr/bin/env python
# coding: utf-8
from pyfiglet import Figlet

f = Figlet( font='slant')
print("")
print(f.renderText('OPTCRI'))
print("author: Di Antonio Ludovico")
print("mail: ludovico.diantonio@lisa.ipsl.fr")
print("OPTCRI V1.0")


import yaml
from tracemalloc import stop
from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd 
from glob import glob
import datetime
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from datetime import timedelta,datetime
import re
from matplotlib.ticker import ScalarFormatter,AutoMinorLocator
import matplotlib.dates as mdates
import miepython
import xarray as xr
import matplotlib
import pathlib
import sys
matplotlib.use('Agg')

print("End Import Libraries...")
print("")
print("Make LUT program...")



# READ INPUTS
with open('../../inputs/config/input.yaml', 'r') as file:
    inputs = yaml.safe_load(file)

site=inputs["SITE"]
verbose=inputs["verbose"]

PATH=inputs["INPUT_obs"]["PATH"]
PATH_IN=PATH+"/"+inputs["INPUT_obs"]["PATH_INPUTS"]
aux_path=PATH_IN+'/aux_inputs/'+site
DATA_IN=inputs["INPUT_obs"]["DATA_INPUT"]
data_path=PATH_IN+'/data/'+DATA_IN


PATH_OUT=inputs["OUTPUTS"]["PATH_OUT"]
pathlib.Path(PATH_OUT).mkdir(parents=True, exist_ok=True)
pathlib.Path(PATH_OUT+"/ri").mkdir(parents=True, exist_ok=True)

lab=inputs["OUTPUTS"]["lab"]

dstart=inputs["PERIOD"]["dstart"]
dend=inputs["PERIOD"]["dend"]
douts=dstart.replace("-","").replace(":","").replace(" ","_")
doute=dend.replace("-","").replace(":","").replace(" ","_")

tlab=inputs["INPUT_obs"]["time_dimension"]

wl_path=inputs["INPUT_opt"]['wl']
wavel=pd.read_csv(aux_path+'/'+wl_path,names=["wavel"])["wavel"].values

save_out=inputs["OUTPUTS"]["save_out"]
plot_inputs=inputs["OUTPUTS"]["plot_inputs"]
plot_outputs=inputs["OUTPUTS"]["plot_outputs"]
plot_size=inputs["OUTPUTS"]["plot_size"]



# LOAD SIZE DISTRIBUTION DATA

diamdlogf=inputs["INPUT_obs"]['particle_size_distribution']["diameters"]
diamdlog=pd.read_csv(aux_path+"/"+diamdlogf,sep="\t")

diam=np.array(diamdlog["D"].values)
scol=diamdlog["D"].astype(str).tolist()
dlog=diamdlog["dlog"].values

if (len(dlog)!=len(scol)):
    raise Exception("Sorry, number of dlog not corresponds to the number of diameters.")


shapef=inputs["INPUT_obs"]['particle_size_distribution']["shape_factor"]
shape_factor=pd.read_csv(aux_path+"/"+shapef)["X"].values
print("")
print("mean shape_factor:", np.mean(shape_factor))
print("")
diam=diam/np.array(shape_factor)

# FUNCTIONS USEFUL
def init_RI(nmin=1.1,nmax=2,kmin=0.001,kmax=0.15,pnx=0.01,pkx=0.001):
    nvec=np.arange(nmin,nmax+pnx,pnx)
    kvec=np.arange(kmin,kmax+pkx,pkx)
    return nvec,kvec
def init_params(diam,wavel):
    radius=diam/2
    nmtom=1e-9 #nanometers to meters
    cross_section_area=np.pi*((nmtom*radius)**(2))  #cross sectional area)
    sizep=(2*np.pi*(np.ones([len(wavel),len(radius)])*radius).transpose())/wavel #sizeparam (105,7)
    return sizep,cross_section_area
    
def init_mie(nvec,kvec,wavel,sizep):
    nl=len(nvec)
    kl=len(kvec)
    wal=len(wavel)
    chil=sizep.shape[0]
    qext=np.ones([wal,nl,kl,chil])
    qabs=np.ones([wal,nl,kl,chil])
    qsca=np.ones(([wal,nl,kl,chil]))
    qback=np.ones(([wal,nl,kl,chil]))
    g=np.ones(([wal,nl,kl,chil]))
    qsca_noc=np.ones(([wal,nl,kl,chil]))
    return qext,qabs,qsca,qback,g,qsca_noc

def QSCA(m,x,tmin,tmax,nt):
    S1,S2 = miepython.mie_S1_S2(m,x,mu,'wiscombe')
    S12=(np.abs(S1)**(2)+np.abs(S2)**(2))
    np.sum(S12*stheta*dtheta[0]/(x*x))
    return np.sum(S12*stheta*dtheta[0]/(x*x))
def form(number):
    return '{:.3f}'.format(number)


# PARAMETERS INITIALIZATION
# modif pour le qsca  
inputs["INPUT_opt"]["nmin"]
nmin=inputs["INPUT_opt"]["nmin"]
nmax=inputs["INPUT_opt"]["nmax"]
kmin=inputs["INPUT_opt"]["kmin"]
kmax=inputs["INPUT_opt"]["kmax"]
pnx=inputs["INPUT_opt"]["dnx"]
pkx=inputs["INPUT_opt"]["dkx"]
nwavel=inputs["INPUT_opt"]["nwavel"]
comm=form(nmin)+"_"+form(nmax)+"_"+form(kmin)+"_"+form(kmax)+"_"+form(pnx)+"_"+form(pkx)+"_"+str(nwavel)+"_"+str(len(diam))

tmin=inputs["INPUT_opt"]["tmin"]#min angle
tmax=inputs["INPUT_opt"]["tmax"] #max angle between 0 and 180
nt=inputs["INPUT_opt"]["nt"] #nangles
theta = np.linspace(tmin,tmax,nt) #angles
dtheta=np.diff(theta*np.pi/180) #radians
mu=np.cos(theta*np.pi/180) #cos(theta)
stheta=np.sin(theta*np.pi/180) #sintheta
nvec,kvec=init_RI(nmin=nmin,nmax=nmax,kmin=kmin,kmax=kmax,pnx=pnx,pkx=pkx)
#evite division by zero

if kvec[0]==0:
    kvec[0]=0.000001

sizep,cross_section_area=init_params(diam,wavel)
qext,qabs,qsca,qback,g,qsca_noc=init_mie(nvec,kvec,wavel,sizep)

LUT=inputs["LUT"]["LUT_flag"]
if LUT==0:
    try:
        np.load(LUT_path+"/"+"qext_"+comm+".npy")
        print("LUT present...please use LUT_flag=1 or delete the existing files")
        sys.exit()
    except:
        pass
LUT_path=inputs["LUT"]["LUT_folder"]
print("LUT in ",LUT_path)
nwl=len(wavel)
nki=len(kvec)
nni=len(nvec)

if LUT==0:
    ####initialize qext, qsca and qabs at different wavelenghts
    """
    initialize the wavelenght qext, qsca, qabs
    # it takes time, go for a coffee
    
    """
    print("Creating LUT...you may go for a long coffee...")
    for iwl in range(nwl):
        print("LUT for wavelength:",wavel[iwl],'nm')
        for ini in range(nni):
            for iki in range(nki):
                n_i=nvec[ini]
                k_i=kvec[iki]
                #qsca counting all the sphere
                qext[iwl,ini,iki,:],qsca[iwl,ini,iki,:],qback[iwl,ini,iki,:], g[iwl,ini,iki,:] = miepython.mie(complex(str(n_i)+'-'+ str(k_i)+"j"), sizep[:,iwl])
                qabs[iwl,ini,iki,:]=qext[iwl,ini,iki,:]-qsca[iwl,ini,iki,:]
                #integrate over the angle of nephelo using scattering amplitudes
                for ichi in range(len(sizep[:,iwl])):
                    S1,S2 = miepython.mie_S1_S2(complex(str(n_i)+'-'+ str(k_i)+"j"),sizep[ichi,iwl],mu,'wiscombe')
                    S12=(np.abs(S1)**(2)+np.abs(S2)**(2))
                    qsca_noc[iwl,ini,iki,ichi]=np.sum(S12*stheta*dtheta[0]/(sizep[ichi,iwl]*sizep[ichi,iwl]))
    # SAVE LUT
    np.save(LUT_path+"/"+"qext_"+comm,qext)
    np.save(LUT_path+"/"+"qsca_"+comm,qsca)
    np.save(LUT_path+"/"+"qback_"+comm,qback)
    np.save(LUT_path+"/"+"g_"+comm,g)
    np.save(LUT_path+"/"+"qsca_nocalib_"+comm,qsca_noc)
    print("LUT Ok, created and saved at:",LUT_path+"/")
    print("\now run the code again with LUT_flag=1")
    sys.exit()

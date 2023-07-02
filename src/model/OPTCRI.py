#!/usr/bin/env python
# coding: utf-8
from pyfiglet import Figlet

f = Figlet( font='slant')
print("")
print(f.renderText('OPTCRI'))
print("author: Di Antonio Ludovico")
print("mail: ludovico.diantonio@lisa.ipsl.fr")
print("")


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

# READ INPUTS
with open('../config/input.yaml', 'r') as file:
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

###################################################################################
LUT=inputs["LUT"]["LUT_flag"]

if LUT==0:
    try:
        np.load(LUT_path+"/"+"qext_"+comm+".npy")
        print("LUT present...please use LUT_flag=1 or delete the existing files")
        sys.exit()
    except:
        pass
LUT_path=inputs["LUT"]["LUT_folder"]
###################################################################################


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


#LOAD SIZE DISTRIBUTION DATA
colnames=[tlab]+scol
filename=data_path+"/"+inputs["INPUT_obs"]['particle_size_distribution']["filename"]
tsize=inputs["INPUT_obs"]['particle_size_distribution']["type"]
PSD=pd.read_csv(filename,sep="\t")
PSD.columns=colnames
print("Reading size:",filename)
print("size type chosen:",tsize)
PSD[tlab]=PSD[tlab].apply(lambda x: pd.to_datetime(x,format="%Y-%m-%d %H:%M"))
PSD=PSD[(PSD[tlab]>=dstart) & (PSD[tlab]<dend)]

if tsize=="volume":
    number=dlog*PSD[scol]/(4/3*np.pi*(diam/2)**(3))
    number[number<0]=np.nan
    number[tlab]=PSD[tlab]
elif tsize=="number":
    number=dlog*PSD[scol]
    number[number<0]=np.nan
    number[tlab]=PSD[tlab]
    


scac=[]
absc=[]
wl_list=[]
for iwl in range(0,len(wavel)):
    absc.append("ABS_COEFF("+str(wavel[iwl])+")")
    scac.append("SCA_COEFF("+str(wavel[iwl])+")")
    wl_list.append(str(wavel[iwl]))

# READING ABSORPTION
colnames=[tlab]+absc
filename=data_path+"/"+inputs["INPUT_obs"]['absorption']["filename"]
print("Reading abs:",filename)
absorption=pd.read_csv(filename,sep="\t")
absorption.columns=colnames
absorption[tlab]=absorption[tlab].apply(lambda x: pd.to_datetime(x,format="%Y-%m-%d %H:%M"))
absorption=absorption[(absorption[tlab]>=dstart) & (absorption[tlab]<dend)]
absorption.index=np.arange(0,len(absorption))

if (len(absorption.columns)!=len(wavel)+1):
    raise Exception("Sorry, verify that you put the correct number of columns in the absorption file TIME,NWL*ABS_COEFF(WL)")


# READING SCATTERING
colnames=[tlab]+scac
filename=data_path+"/"+inputs["INPUT_obs"]['scattering']["filename"]
scattering=pd.read_csv(filename,sep="\t")
scattering.columns=colnames
print("Reading sca:",filename)
scattering[tlab]=scattering[tlab].apply(lambda x: pd.to_datetime(x,format="%Y-%m-%d %H:%M"))
scattering=scattering[(scattering[tlab]>=dstart) & (scattering[tlab]<dend)]
scattering.index=np.arange(0,len(scattering))
if (len(scattering.columns)!=len(wavel)+1):
    raise Exception("Sorry, verify that you put the correct number of columns in the scattering file TIME, NWL*SCAT_COEFF(WL)")


tscattering=scattering[tlab]
tabsorption=absorption[tlab]
tPSD=PSD[tlab]

common_dates=np.intersect1d(tscattering, tabsorption)
common_dates=np.intersect1d(common_dates, tPSD)

scattering=scattering[scattering[tlab].isin(common_dates)]
absorption=absorption[absorption[tlab].isin(common_dates)]
number=number[number[tlab].isin(common_dates)]
absorption.index=np.arange(0,len(absorption))
scattering.index=np.arange(0,len(scattering))


print("loaded inputs first lines..:")
print("Absorption")
print(absorption.head())
print("Scattering")
print(scattering.head())
print("Size")
print(number.head())


scac=[]
absc=[]
wl_list=[]
for iwl in range(0,len(wavel)):
    absc.append("ABS_COEFF("+str(wavel[iwl])+")")
    scac.append("SCA_COEFF("+str(wavel[iwl])+")")
    wl_list.append(str(wavel[iwl]))
 

cref_correction=inputs["INPUT_obs"]['absorption']["cref"]["cref_correction"]
cref_corr_type=inputs["INPUT_obs"]['absorption']["cref"]["cref_corr_type"]
def cref_yus(SSA,Cf,ms):
    return Cf+(ms/100)*(SSA/(1-SSA))

if cref_correction==1:

    print("")
    print("Cref Correction activated:") 

    if cref_corr_type==0:

        cref_factor=inputs["INPUT_obs"]['absorption']["cref"]["cref_factor"]
        print("Cref correction used:",cref_factor)
        cref_standard=inputs["INPUT_obs"]['absorption']["cref"]["cref_standard"]
        print("cref standard:",cref_standard)
        print("cref factor:",cref_factor)
        print("")
        absorption[absc]=absorption[absc]*cref_standard/cref_factor
    elif cref_corr_type==1:

        print("Yus-Diez et al, 2021 Cref correction")
        cref_standard=inputs["INPUT_obs"]['absorption']["cref"]["cref_standard"]
        print("cref standard:",cref_standard)
        print("")
        Cf=inputs["INPUT_obs"]['absorption']["cref"]["Cf"]
        ms=inputs["INPUT_obs"]['absorption']["cref"]["ms"]
        print("Cref calculated from: ",
              '{:.2f}'.format(Cf),"+",
              '{:.2f}'.format(ms),"(SSA/1-SSA)")
        
        wl_ssa=inputs["INPUT_obs"]['absorption']["cref"]["wl_ssa"]
        
        ssa_laba="ABS_COEFF("+str(wl_ssa)+")"
        ssa_labs="SCA_COEFF("+str(wl_ssa)+")"
        #SSA=(scattering[ssa_labs]/(absorption[ssa_laba]+scattering[ssa_labs]))
        #cref_factor_tmp=np.mean(cref_yus(SSA,Cf,ms))
        cref_factor_tmp=inputs["INPUT_obs"]['absorption']["cref"]["cref_factor"]
        print("First guess average Cref used:", cref_factor_tmp)
        SSA_new=(scattering[ssa_labs]/((absorption[ssa_laba]*cref_standard/cref_factor_tmp)+scattering[ssa_labs]))
        SSA_new=SSA_new.fillna(np.nanmean(SSA_new))
        cref_factor=cref_yus(SSA_new,Cf,ms).values
        print("Cref used:", cref_factor)
        for ab in absc:
            absorption[ab]=absorption[ab]*cref_standard/cref_factor
        with open(PATH_OUT+'/Cref'+lab+'_'+douts+"_"+doute+".txt", 'w') as f:
            f.write('Cref: First guess:\n')
            f.write(str(cref_factor_tmp))
            f.write('\nCref:\n')
            f.write(str(cref_factor))
    
x1=scattering[scac+[tlab]]
x2=absorption[absc+[tlab]]
x3=number[scol+[tlab]]
#print("ABS")
#print(x1[x1[tlab]=="2022-06-25 07:00:00"])
#print("SCA")
#print(x2[x2[tlab]=="2022-06-25 07:00:00"])
#print("SMPS")
#print(x3[x3[tlab]=="2022-06-25 07:00:00"])
df_m=x1.merge(x2,on=tlab).merge(x3,on=tlab)
df_m=df_m.set_index(tlab)
df_m=df_m.dropna()
df_m=df_m[(df_m.index>=dstart) & (df_m.index<dend) ]


df_m.reset_index().to_csv(PATH_OUT+"/ri/inputs_"+lab+"_"+douts+"_"+doute+".csv",index=False,sep=";")


if plot_inputs==1:
    fig, ax = plt.subplots(figsize=(15,5),sharex=True)
    fig.subplots_adjust(right=0.75)

    twin1 = ax.twinx()
    twin2 = ax.twinx()
    ds=dstart
    de=dend
    twin1.scatter(absorption[(absorption[tlab]>=ds) & (absorption[tlab]<de)][tlab],
             absorption[(absorption[tlab]>=ds) & (absorption[tlab]<de)][absc[0]],color="r")

    ax.scatter(number[(number[tlab]>=ds) & (number[tlab]<de)][tlab],
             number[(number[tlab]>=ds) & (number[tlab]<de)][scol].sum(axis=1),color="k")
    ax.set_ylabel("NTOT")
    ax.set_yscale("log")
    twin1.set_ylabel("$\sigma_{abs}$ [Mm-1]")
    twin1.tick_params(colors='r', which='both')
    twin1.yaxis.label.set_color('r')
    twin2.set_ylabel("$\sigma_{sca}$ [Mm-1]")
    twin2.scatter(scattering[(scattering[tlab]>=ds) & (scattering[tlab]<de)][tlab],
             scattering[(scattering[tlab]>=ds) & (scattering[tlab]<de)][scac[0]],color="b")
    twin2.tick_params(colors='b', which='both')
    twin2.yaxis.label.set_color('b')
    # Offset the right spine of twin2.  The ticks and label have already been
    # placed on the right by twinx above.
    twin2.spines.right.set_position(("axes", 1.1))
    ax.set_xlim([pd.to_datetime(ds)-timedelta(hours=1),pd.to_datetime(de)+timedelta(hours=1)])
    plt.savefig(PATH_OUT+"/plots/distscaabs.png")
    
if plot_size==1:
    print("")
    print("Plot input size...")
    print("")
    for itime in range(0,len(number)):
        try:
            x=diam/1000
            y=number[scol].iloc[itime]/dlog
            t=number[tlab].iloc[itime]
            print(t)
            if np.sum(np.isnan(y))/len(y)!=1:
                fig, ax = plt.subplots(figsize=(10,5),sharex=True)
                fig.subplots_adjust(right=0.75)
                ax.scatter(x,
                         y,color="k",marker="s")
                ax.plot(x,
                         y,color="k")
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.set_xticks([0.01,0.02,0.03,0.05,0.1,0.2,0.3,0.5,1],
                       [0.01,0.02,0.03,0.05,0.1,0.2,0.3,0.5,1])
                ax.set_yticks([0.1,1,10,100,1000,10000,10000,100000])
                #ax.set_yticks(np.arange(0,15000,2000))
                #ax.yaxis.set_minor_locator(MultipleLocator(200))
                ax.set_ylabel('dN/dlog(D) (#/cm-3)')
                ax.set_xlabel('D (um)')
                ax.set_title(number[tlab].iloc[itime])
                
                plt.savefig(PATH_OUT+"/plots/dNdlog_"+t.strftime("%Y%m%d_%H%M%S")+".png")
                plt.close(fig)
        except:
            pass

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
    
def init_vars(df_m,nvec,kvec,wavel):
    timel=len(df_m)
    nl=len(nvec)
    kl=len(kvec)
    wal=len(wavel)
    bscam=np.zeros([timel,wal,nl,kl])
    bscamtot=np.zeros([timel,wal,nl,kl])
    babsm=np.zeros([timel,wal,nl,kl])
    RMSD=np.ones(([timel,wal,nl,kl]))
    n=np.ones([nl,kl])
    k=np.ones([nl,kl])
    return n,k,RMSD,babsm,bscam,bscamtot
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

def init_csca(df_m,wavel,sizep):
    timel=len(df_m)
    wal=len(wavel)
    chil=sizep.shape[0]
    csca=np.ones(([timel,wal]))
    return csca
def out_vars(df_m,nvec,kvec,wavel):
    timel=len(df_m)
    nl=len(nvec)
    kl=len(kvec)
    wal=len(wavel)
    chil=sizep.shape[0]
    bscaobs=np.zeros([timel,wal])
    babsobs=np.zeros([timel,wal])
    babsmod=np.zeros([timel,wal])
    bscamod=np.zeros([timel,wal])
    nout=np.ones([timel,wal])
    kout=np.ones([timel,wal])
    return nout,kout,babsobs,bscaobs,babsmod,bscamod

def QSCA(m,x,tmin,tmax,nt):
    S1,S2 = miepython.mie_S1_S2(m,x,mu,'wiscombe')
    S12=(np.abs(S1)**(2)+np.abs(S2)**(2))
    np.sum(S12*stheta*dtheta[0]/(x*x))
    return np.sum(S12*stheta*dtheta[0]/(x*x))
def form(number):
    return '{:.3f}'.format(number)
def verbose_print_1(wl,realm,imm):
    print("Wavelength:",wl,"nm")
    print("nmod:",'{:.4f}'.format(realm))
    print("kmod:",'{:.3f}'.format(imm))
def verbose_print_2(wl,ntot,realm,imm,scao,abso,scam,absm,RMSD):
    print("Wavelength:",wl,"nm")
    print("Ntot:",'{:.3f}'.format(ntot),"#/cm3")
    print("nmod:",'{:.4f}'.format(realm))
    print("kmod:",'{:.3f}'.format(imm))
    print("Scattering OBS:",'{:.3f}'.format(scao),'Scattering MOD:',scam)
    print("Absorption OBS:",'{:.3f}'.format(abso),'Absorption MOD:',absm)
    print("RMSD:",'{:.4f}'.format(RMSD))
    print("MOD-OBS Sca coeff:",'{:.2f}'.format(scam[0]-scao))
    print("MOD-OBS Abs coeff:",'{:.2f}'.format(absm[0]-abso))
def save_n_k(df_m,nout,kout,wavel,PATH_OUT,comm,tlab,douts,doute):
    kdf=pd.DataFrame(kout)
    kdf=pd.DataFrame(kout)
    kdf.index=df_m.reset_index()[tlab]
    kdfout=kdf.reset_index()
    kdfout.columns=[tlab]+list(map(str,list(wavel)))
    kdfout.to_csv(PATH_OUT+"/ri/"+lab+"_k_"+comm+"_"+douts+"_"+doute+".csv",index=False,sep=";")
    ndf=pd.DataFrame(nout)
    ndf.index=df_m.reset_index()[tlab]
    ndfout=ndf.reset_index()
    ndfout.columns=[tlab]+list(map(str,list(wavel)))
    ndfout.to_csv(PATH_OUT+"/ri/"+lab+"_n_"+comm+"_"+douts+"_"+doute+".csv",index=False,sep=";")


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
comm=form(nmin)+"_"+form(nmax)+"_"+form(kmin)+"_"+form(kmax)+"_"+form(pnx)+"_"+form(pkx)+"_"+str(nwavel)

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
n,k,RMSD,babsm,bscam,bscamtot=init_vars(df_m,nvec,kvec,wavel)
nout,kout,babsobs,bscaobs,babsmod,bscamod=out_vars(df_m,nvec,kvec,wavel)
qext,qabs,qsca,qback,g,qsca_noc=init_mie(nvec,kvec,wavel,sizep)
csca=init_csca(df_m,wavel,sizep)




#LUT=inputs["LUT"]["LUT_flag"]
#if LUT==0:
#    try:
#        np.load(LUT_path+"/"+"qext_"+comm+".npy")
#        print("LUT present...please use LUT_flag=1 or delete the existing files")
#        sys.exit()
#    except:
#        pass
#LUT_path=inputs["LUT"]["LUT_folder"]

ntime=len(df_m)
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
else:
    #Read LUT
    print("LUT READ from", LUT_path)
    qext=np.load(LUT_path+"/"+"qext_"+comm+".npy")
    qsca=np.load(LUT_path+"/"+"qsca_"+comm+".npy")
    qback=np.load(LUT_path+"/"+"qback_"+comm+".npy")
    g=np.load(LUT_path+"/"+"g_"+comm+".npy")
    qsca_noc=np.load(LUT_path+"/"+"qsca_nocalib_"+comm+".npy")


for itime in range(ntime):  
    print("---------------------------------------------------------")
    print("Processing...",df_m.index[itime],"...",itime+1,"of",ntime)
    dist=df_m[scol].iloc[itime,:].to_numpy()*1e6 # #/cm3 to #/m3
    #print(dist)
    for iwl in range(nwl):
        
        babsobs[itime,iwl]=df_m[absc[iwl]].iloc[itime]
        if babsobs[itime,iwl]==0.0:
            babsobs[itime,iwl]=1e-5
        bscaobs[itime,iwl]=df_m[scac[iwl]].iloc[itime]
        if bscaobs[itime,iwl]==0.0:
            bscaobs[itime,iwl]=1e-5
        
        
        for ini in range(nni):
            n_i=nvec[ini]
            for iki in range(nki):
                k_i=kvec[iki]
                #qsca counting all the sphere
                qextc, qscac, qbackc, gc = qext[iwl,ini,iki,:],qsca[iwl,ini,iki,:],qback[iwl,ini,iki,:], g[iwl,ini,iki,:]
                qabsc=qextc-qscac
                qscac=qsca_noc[iwl,ini,iki,:]   
                
                bscam[itime,iwl,ini,iki]=1e6*np.sum(cross_section_area*qscac*dist) #m-1 to Mm-1
                babsm[itime,iwl,ini,iki]=1e6*np.sum(cross_section_area*qabsc*dist) #m-1 to Mm-1
                #0 180°
                bscamtot[itime,iwl,ini,iki]=1e6*np.sum(cross_section_area*qsca[iwl,ini,iki,:]*dist) #m-1 to Mm-1
                #minimize errors
                absterm=((babsobs[itime,iwl]-babsm[itime,iwl,ini,iki])/babsm[itime,iwl,ini,iki])**(2)
                scaterm=((bscaobs[itime,iwl]-bscam[itime,iwl,ini,iki])/bscam[itime,iwl,ini,iki])**(2)
                RMSD[itime,iwl,ini,iki]=np.sqrt(scaterm+absterm)
                #################
                n[ini,iki]=n_i
                k[ini,iki]=k_i
        #find index of the minimum RMSD
        idxn,idxk=np.where(RMSD[itime,iwl,:,:]==np.nanmin(RMSD[itime,iwl,:,:]))
        babsmod[itime,iwl]=babsm[itime,iwl,idxn,idxk]
        bscamod[itime,iwl]=bscam[itime,iwl,idxn,idxk]
        nout[itime,iwl]=n[idxn,idxk]
        kout[itime,iwl]=k[idxn,idxk]
        csca[itime,iwl]=bscamtot[itime,iwl,idxn,idxk]/bscam[itime,iwl,idxn,idxk]

        if verbose==1:
            verbose_print_1(wavel[iwl],nout[itime,iwl],kout[itime,iwl])
        if verbose==2:
            verbose_print_2(wavel[iwl],np.sum(dist/1e6),nout[itime,iwl],kout[itime,iwl],
                          bscaobs[itime,iwl],babsobs[itime,iwl],
                          bscam[itime,iwl,idxn,idxk],babsm[itime,iwl,idxn,idxk],
                          np.nanmin(RMSD[itime,iwl,:,:]))
print("Calculation of truncation-corrected SSA")
bsca_corr=bscaobs*csca
SSA_tcorr=bsca_corr/(bsca_corr+babsobs)



if save_out==1:
    
    save_n_k(df_m,nout,kout,wavel,PATH_OUT,comm,tlab,douts,doute)
    outputs = xr.Dataset(data_vars={"RMSD":(["time","nwl","nni","nki"],RMSD), 
                           "babsmod_all":(["time","nwl","nni","nki"],babsm),
                           "bscamod_notruncation_all":(["time","nwl","nni","nki"],bscam),
                           "bscamod_total_all":(["time","nwl","nni","nki"],bscamtot),
                           "bscaobs":(["time","nwl"],bscaobs),
                           "babsobs":(["time","nwl"],babsobs),
                           "bscamod":(["time","nwl"],bscamod),
                           "babsmod":(["time","nwl"],babsmod),
                           "nout":(["time","nwl"],nout),
                           "kout":(["time","nwl"],kout),
                           "csca":(["time","nwl"],csca),
                           "SSA_corr":(["time","nwl"],SSA_tcorr),
                           "particle_size_distribution":(["time","diameters"],df_m[scol])}, 
    coords=dict(time=df_m.reset_index()[tlab].values,nwl=wavel,diameters=diam))

    fout=PATH_OUT+"/CRI_MODEL_"+lab+'_'+douts+"_"+doute+".nc"
    print("Saving n,k and outputs to netcdf: "+fout)
    outputs.to_netcdf(fout)


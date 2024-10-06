import xarray as xr
from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd 
from glob import glob
from datetime import datetime
from matplotlib import dates  as mdates 
import pathlib



pathf="/DATA/CHIMERE/OUTPUT/LDIANTONIO/OPTCRIOUT/"
dstart='2022-06-15 00:00:00'
dend='2022-07-25 00:00:00'
res='1H'
lab="CTRL"
sim_list=['median', 'mean','q1', 'q3', 'stdp', 'stdm', 'errm', 'errp']
size='SMPS'
cref=2.45#'yus'
site="RAMB"
DIROUT="/home/ldiantonio/OPTCRI"


def read_out_optcri(lab,sim,size,cref,pathf,site="RAMB",dstart=dstart,dend=dend):
    if isinstance(cref, float):
        crefl="{:.2f}".format(cref)
    elif isinstance(cref, str):
        crefl=cref
    path=pathf+site+"/"+lab+'_'+sim+'_'+size+"_cref"+crefl+'/ri/'
    print("reading from:",path)
    npath=glob(path+lab+"*_n_*")
    kpath=glob(path+lab+"*_k_*")
    for idx in range(0,len(npath)):
        if idx==0:
            ndf=pd.read_csv(npath[idx],sep=";")
            kdf=pd.read_csv(kpath[idx],sep=";")
        else:
            ndf=pd.concat([ndf,pd.read_csv(npath[idx],sep=";")])
            kdf=pd.concat([kdf,pd.read_csv(kpath[idx],sep=";")])
    idx = pd.date_range(dstart,dend ,freq=res)
    ndf.TIME=pd.to_datetime(ndf.TIME,format="%Y-%m-%d %H:%M:%S")
    ndf.index=pd.DatetimeIndex(ndf.TIME)
    ndf=ndf.reindex(idx, fill_value=np.nan)
    ndf.TIME=ndf.index

    kdf.TIME=pd.to_datetime(kdf.TIME,format="%Y-%m-%d %H:%M:%S")
    kdf.index=pd.DatetimeIndex(kdf.TIME)
    kdf=kdf.reindex(idx, fill_value=np.nan)
    kdf.TIME=kdf.index
    return ndf,kdf


ndf_CTRL,kdf_CTRL=read_out_optcri("CTRL","median","SMPS",2.45,pathf,site=site)
ndf2,kdf2=read_out_optcri("CTRL",sim_list[1],"SMPS",2.45,pathf,site=site)
ndf3,kdf3=read_out_optcri("CTRL",sim_list[2],"SMPS",2.45,pathf,site=site)
ndf4,kdf4=read_out_optcri("CTRL",sim_list[3],"SMPS",2.45,pathf,site=site)
#ndf5,kdf5=read_out_optcri("CTRL",sim_list[4],"SMPS",2.45,pathf,site=site)
#ndf6,kdf6=read_out_optcri("CTRL",sim_list[5],"SMPS",2.45,pathf,site=site)
ndf7,kdf7=read_out_optcri("CTRL",sim_list[6],"SMPS",2.45,pathf,site=site)
ndf8,kdf8=read_out_optcri("CTRL",sim_list[7],"SMPS",2.45,pathf,site=site)


#ndf_concat = pd.concat((ndf_CTRL, ndf2,ndf3,ndf4,ndf5,ndf6,ndf7,ndf8))
ndf_concat = pd.concat((ndf_CTRL, ndf2,ndf3,ndf4,ndf7,ndf8))
#kdf_concat = pd.concat((kdf_CTRL, kdf2,kdf3,kdf4,kdf5,kdf6,kdf7,kdf8))
kdf_concat = pd.concat((kdf_CTRL, kdf2,kdf3,kdf4,kdf7,kdf8))

ndf_mean=ndf_concat.groupby(["TIME"]).mean().reset_index()
kdf_mean=kdf_concat.groupby(["TIME"]).mean().reset_index()
ndf_std=ndf_concat.groupby(["TIME"]).std().reset_index()
kdf_std=kdf_concat.groupby(["TIME"]).std().reset_index()
ndf_med=ndf_concat.groupby(["TIME"]).quantile(.5).reset_index()
kdf_med=kdf_concat.groupby(["TIME"]).quantile(.5).reset_index()
ndf_25=ndf_concat.groupby(["TIME"]).quantile(.25).reset_index()
kdf_25=kdf_concat.groupby(["TIME"]).quantile(.25).reset_index()
kdf_75=kdf_concat.groupby(["TIME"]).quantile(.75).reset_index()
ndf_75=ndf_concat.groupby(["TIME"]).quantile(.75).reset_index()

pathlib.Path(DIROUT+"/outputs/"+site).mkdir(parents=True, exist_ok=True)
ndf_mean.to_csv(DIROUT+"/outputs/"+site+"/"+"mean_ndf.csv",index=False,sep=";")
kdf_mean.to_csv(DIROUT+"/outputs/"+site+"/"+"mean_kdf.csv",index=False,sep=";")
ndf_std.to_csv(DIROUT+"/outputs/"+site+"/"+"std_ndf.csv",index=False,sep=";")
kdf_std.to_csv(DIROUT+"/outputs/"+site+"/"+"std_kdf.csv",index=False,sep=";")
ndf_25.to_csv(DIROUT+"/outputs/"+site+"/"+"q1_ndf.csv",index=False,sep=";")
kdf_25.to_csv(DIROUT+"/outputs/"+site+"/"+"q1_kdf.csv",index=False,sep=";")
ndf_75.to_csv(DIROUT+"/outputs/"+site+"/"+"q3_ndf.csv",index=False,sep=";")
kdf_75.to_csv(DIROUT+"/outputs/"+site+"/"+"q3_kdf.csv",index=False,sep=";")
ndf_med.to_csv(DIROUT+"/outputs/"+site+"/"+"med_ndf.csv",index=False,sep=";")
kdf_med.to_csv(DIROUT+"/outputs/"+site+"/"+"med_kdf.csv",index=False,sep=";")



print("Writing output to:",DIROUT+"/outputs/"+site)

plotfig=1

if plotfig==1: 

   fig,(ax1,ax2)=plt.subplots(2,1,figsize=(20,10),sharex=True)
   wlp="520"


   ax1.plot(ndf_mean["TIME"],ndf_mean[wlp])
   ax1.plot(ndf_25["TIME"],ndf_25[wlp])
   ax1.plot(ndf_75["TIME"],ndf_75[wlp])
   ax1.fill_between(ndf_med["TIME"],ndf_med[wlp]-ndf_std[wlp],ndf_mean[wlp]+ndf_std[wlp],color="k")

   ax2.plot(kdf_mean["TIME"],kdf_mean[wlp])
   ax2.plot(kdf_25["TIME"],kdf_25[wlp])
   ax2.plot(kdf_75["TIME"],kdf_75[wlp])
   ax2.fill_between(kdf_med["TIME"],kdf_mean[wlp]-kdf_std[wlp],kdf_mean[wlp]+kdf_std[wlp],color="k")

   ax2.set_xticks([datetime(2022,6,15,12),
              datetime(2022,6,18,12),
              datetime(2022,6,21,12),
              datetime(2022,6,24,12),
              datetime(2022,6,27,12),
              datetime(2022,6,30,12),
              datetime(2022,7,3,12),
              datetime(2022,7,6,12),
              datetime(2022,7,9,12),
              datetime(2022,7,12,12),
              datetime(2022,7,15,12),
              datetime(2022,7,18,12),
              datetime(2022,7,21,12),
              datetime(2022,7,24,12)])
   ax2.set_xlim([datetime(2022,6,15,12),
              datetime(2022,7,24,12)])
#ax2.set_yscale("log")
   myFmt = mdates.DateFormatter('%m%d\n%H:%M')
   ax2.xaxis.set_major_formatter(myFmt)
   ax2.tick_params(which='major', length=9,direction="out",labelsize=18)
   ax2.tick_params(which='minor', length=6,direction="out",labelsize=18)
   ax1.tick_params(which='major', length=9,direction="out",labelsize=18)
   ax1.tick_params(which='minor', length=6,direction="out",labelsize=18)
   ax1.set_ylabel("n "+wlp+"nm",fontsize=20)
   ax2.set_ylabel("k "+wlp+"nm",fontsize=20)
   ax1.text(datetime(2022,6,11),1.8,"a)",weight="bold",fontsize=25)
   ax2.text(datetime(2022,6,11),0.150,"b)",weight="bold",fontsize=25)
   ax1.set_ylim(1.3,1.8)
   plt.show()


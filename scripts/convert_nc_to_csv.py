import xarray as xr
from matplotlib import pyplot as plt
import numpy as np
import os
import pandas as pd 
from glob import glob
site_list=["RAMB"]

PATHOUT="/DATA/CHIMERE/OUTPUT/LDIANTONIO/OPTCRIOUT/"

lab="CTRL"
sim_list=['median']
size='SMPS'
cref=2.45#'yus'

nc_file_init="CRI_MODEL_CTRL_*"

varout=["csca","SSA_corr","babsmod","bscamod","babsobs","bscaobs"]
if isinstance(cref, float):
    crefl="{:.2f}".format(cref)
elif isinstance(cref, str):
    crefl=cref

for sim in sim_list:
   for site in site_list:
      path=PATHOUT+site+"/"+lab+'_'+sim+'_'+size+"_cref"+crefl+'/'
      gpath=path+"netCDF/"+nc_file_init
      fpath=glob(gpath)
      fpath.sort()
      isExist = os.path.exists(path+"csv/")
      if not isExist:
        # Create a new directory because it does not exist
         os.makedirs(path+"csv/")
         print("The new directory is created!")

      for file in fpath:
          print(file)
          df=xr.open_dataset(file)

          for var in varout:
                tmpout=df[var].to_dataframe().reset_index()
                dname=file.split("/")[-1].replace("CRI_MODEL_CTRL_","").replace(".nc","")
                fout=path+"csv/"+var+"_"+dname+".csv"
                print("Save output:",fout)
                tmpout.to_csv(fout,sep="\t",index=False)


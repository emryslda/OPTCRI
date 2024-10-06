#!/bin/bash
export PYTHONUNBUFFERED=true

module load anaconda3-py/

env_name="OPTCRI"

conda activate $env_name
echo "Loaded conda $env_name environment"


export firstdate=20220619
export lastdate=20220620 #726
export ndays=1
export site_list="PRG"
export sim_list="median" #median mean q1 q3 errm errp"


echo "Start Optical calculation..."
echo ""
time ./optcri_loop_site.sh || exit 1 
echo "Done"

#!/bin/bash
#export PYTHONUNBUFFERED=true

#module load anaconda3-py/

#env_name="OPTCRI"

#conda activate $env_name
#echo "Loaded conda $env_name environment"


#firstdate=20220619
#lastdate=20220620 #726
#ndays=1
#site_list="PRG"
#sim_list="median" #median mean q1 q3 errm errp"
#sim_list="median"
#########################END USER INPUT

dir=$PWD
export my_awk=awk


for sim in $sim_list;do

for site in $site_list;do
	echo "Processing...$site"

	export optcriparams=optcri.$site.par
        cat optcri.$site.sed | sed "s|__SIMLAB__|${sim}|" \
                        >  $optcriparams
	echo "period:${firstdate}-${lastdate}"
	make clean
	export optcriparash=${optcriparams}.sh

	. ${dir}/scripts/read-params.sh || exit 1
	source ${optcriparash}
	while read l; do eval eval $l; done < ${optcriparash}

	di=${firstdate}
	while [ $di -lt $lastdate ] ; do

	   dstart=$(date -d $di '+%Y-%m-%d %H:%M:%S')
	   dend=`date -u -d "$di $ndays days" +%Y%m%d`
	   dend=$(date -d $dend '+%Y-%m-%d %H:%M:%S')

	   if [ "$lutflag" -eq 1 ]; then

	       echo
	       echo "Running OPTCRI" $dstart " " $dend
	       echo
	   fi    
	   
	   cat ${inputf}/config/input.sed | sed "s|__DSTART__|${dstart}|" \
	   		 | sed "s|__DEND__|${dend}|"      \
	   		 | sed "s|__PATH__|${dir}|"      \
	   		 | sed "s|__INPUTF__|${inputf}|"  \
	   		 | sed "s|__SIZEF__|${size_filename}|"     \
	   		 | sed "s|__DIAMF__|${size_diam}|"      \
	   		 | sed "s|__SIZET__|${size_type}|"      \
	   		 | sed "s|__SHAPEF__|${shapef}|"      \
	   		 | sed "s|__ABSF__|${absf}|"      \
	   		 | sed "s|__CREFC__|${cref_correction}|"      \
	   		 | sed "s|__CREFT__|${cref_correction_type}|"      \
	   		 | sed "s|__CREFS__|${cref_standard}|"      \
	   		 | sed "s|__CREFF__|${cref_factor}|"      \
	   		 | sed "s|__CF__|${Cf}|"      \
	   		 | sed "s|__MS__|${ms}|"      \
	   		 | sed "s|__SSAWL__|${wl_ssa}|"      \
	   		 | sed "s|__ABSFAC__|${absfactor}|"      \
	   		 | sed "s|__SCAF__|${scaf}|"      \
	   		 | sed "s|__SCAFAC__|${scafactor}|"      \
	   		 | sed "s|__TIMED__|${timelab}|"      \
	   		 | sed "s|__NMIN__|${nmin}|"      \
	   		 | sed "s|__NMAX__|${nmax}|"      \
	   		 | sed "s|__DNX__|${dnx}|"      \
	   		 | sed "s|__KMIN__|${kmin}|"      \
	   		 | sed "s|__KMAX__|${kmax}|"      \
	   		 | sed "s|__DKX__|${dkx}|"      \
	   		 | sed "s|__NWL__|${nwavel}|"   \
	   		 | sed "s|__WLF__|${wlf}|"      \
	   		 | sed "s|__CSCA__|${csca}|"      \
	   		 | sed "s|__TMIN__|${tmin}|"      \
	   		 | sed "s|__TMAX__|${tmax}|"      \
	   		 | sed "s|__NANG__|${nt}|"      \
	   		 | sed "s|__LUTF__|${lutfolder}|"      \
	   		 | sed "s|__LUTFLAG__|${lutflag}|"      \
	   		 | sed "s|__PATHOUT__|${outputdir}|"      \
	   		 | sed "s|__SOUT__|${sout}|"      \
	   		 | sed "s|__PIN__|${pin}|"      \
	   		 | sed "s|__POUT__|${pout}|"      \
	   		 | sed "s|__PSIZE__|${pout_size}|"      \
	   		 | sed "s|__VERB__|${verbose}|"      \
	   		 | sed "s|__UNITSIZE__|${unitsize}|"      \
	   		 | sed "s|__UNITABS__|${unitabs}|"      \
	   		 | sed "s|__UNITSCA__|${unitsca}|"      \
	   		 | sed "s|__LAB__|${labelsim}|"      \
	   		 | sed "s|__DATAIN__|${datain}|"      \
	   		 | sed "s|__SLAB__|${sitelab}|"      \
	   				 > ${inputf}/config/input.yaml
	   cd ${dir}/src/model
            
           if [ "$lutflag" -eq 0 ]; then
               echo "lutflag is equal to 0, making LUT tables...and exit"
               python make_LUT.py || exit 1
	       exit
           else
               :
           fi
             	   
	   python OPTCRI.py || exit 1 
	   
	   di=`date -u -d "$di $ndays days" +%Y%m%d`
	   cd $dir
	done
done
done

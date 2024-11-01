######################################################################################################################################################
#                                              
#       INPUT PAR FILE TO OPTCRI               
#       contact: Di Antonio Ludovico           
#       mail: ludovico.diantonio@lisa.ipsl.fr  
#                                              
#######################################################################################################################################################

# Input observations

 [$dir]         Input directory                                                                                           : $PWD
 [$sitelab]     Site name                                                                                                 : SIRTA 
 [$labelsim]    Simulation label                                                                                          : CTRL 
 [$inputf]      input folder (data, aux)                                                                                  : inputs

# size distribution 

 [$lab_size]         size distribution label (ex. SMPS,SMPS_GRIMM)                                                        : SMPS
 [$size_filename]    size distribution filename containing data                                                           : particle_size_distribution_${labelsim}.csv
 [$size_diam]        size distribution filename containing D,dlo                                                          : size_diameters.dat 
 [$size_type]        number dN/dlog (#/cm3), volume dV/log (um3/cm3)                                                      : number
 [$size_stat]        mean/median/q1/q3                                                                                    : __SIMLAB__
 [$shapef]           shape_factor                                                                                         : shape_factor.dat 
 [$unitsize]         units size distribution diameters  (nm/um)                                                           : nm

# absorption/multiple scattering correction

 [$absf]               Absorption filename                                                                                : absorption_${labelsim}.csv
 [$cref_correction]  cref correction  (0/1)                                                                               : 1
 [$cref_correction_type] (0/1)                                                                                            : 0
 [$cref_standard]  cref filter tape standard FF (ACTRIS=1.39)                                                             : 1.39 
 [$cref_factor]    cref factor for simple correction (cref_correction_type=0)                                             : 2.45
 [$Cf]             Valid only if cref_correction_type=1                                                                   : 2.50
 [$ms]             Valid only if cref_correction_type=1                                                                   : 1.6
 [$wl_ssa]         Valid only if cref_correction_type=1/wavelength (in nm) for SSA calculation for Yus diez, 2021         : 660
 [$absfactor]      absorption factor for sensitivity test                                                                 : 1
 [$unitabs]        absorption units (Mm-1/m-1)                                                                            : Mm-1

# scattering

 [$scaf]           scattering filename                                                                                    : scattering_${labelsim}.csv
 [$scafactor]      scattering factor for sensitivity test                                                                 : 1
 [$unitsca]        scattering units (Mm-1/m-1)                                                                            : Mm-1

 [$datain]     input data folder under $inputf/data containing (abs,sca,size) relative path                               : ${sitelab}/${labelsim}_${size_stat}_${lab_size}

# time dimension

 [$timelab]       time dimension name in the input file (common name for sca,abs,size)                                    : TIME

# Complex refractive index input parameters

 [$nmin]             Minimum n for the retrievals                                                                         : 1.2
 [$nmax]             Maximum n for the retrievals                                                                         : 2      
 [$dnx]              dn for iteration (0.01 suggested)                                                                    : 0.01
 [$kmin]             Minimum k for the retrievals                                                                         : 0.
 [$kmax]             Maximum k for the retrievals                                                                         : 0.2
 [$dkx]              dk for iteration (0.001 suggested)                                                                   : 0.001
 [$nwavel]           number of wavelenghts of input file                                                                  : 7
 [$wlf]              wavelenghths file (nm)                                                                               : wl_input.dat
 [$csca]             Truncation scattering efficiency calculation                                                         : 1
 [$tmin]             Minimum scattering angle (0,180)                                                                     : 9
 [$tmax]             Maximum scattering angle (0,180)                                                                     : 170
 [$nt]               number scattering angle (300 suggested)                                                              : 300

# Look up tables instructions

 [$lutfolder]        Look up table folder    (full_path)                                                                  : ${dir}/${inputf}/LUT
 [$lutflag]          Look up table present (0/1)                                                                          : 1

# Output directory 

 [$commdir]          Output main directory                                                                                : /DATA/CHIMERE/OUTPUT/LDIANTONIO/OPTCRIOUT/${sitelab} 
 [$outputdir]        Output folder name      (full_path)                                                                  : ${commdir}/${labelsim}_${size_stat}_${lab_size}_cref2.45

# Save and plot

 [$sout]             save output                                                                                          : 1
 [$pin]              plot inputs                                                                                          : 0
 [$pout]             plot outputs                                                                                         : 1
 [$pout_size]        plot size                                                                                            : 0
 [$verbose]          (0/1/2)                                                                                              : 2

SITE: __SLAB__
INPUT_obs:
    PATH: __PATH__
    PATH_INPUTS: __INPUTF__
    DATA_INPUT: __DATAIN__
    particle_size_distribution: #dN/dlOG  #/cm3 units or dV/dlog um3/cm3 expected TIME+diameters midpoints columns
        filename: __SIZEF__
        diameters: __DIAMF__
        type: __SIZET__ #volume or number
        units: __UNITSIZE__
        shape_factor: __SHAPEF__
    absorption:
        filename: __ABSF__
        cref: 
            cref_correction: __CREFC__
            cref_corr_type: __CREFT__ #0 use cref_factor 1 # Yus-DIez et al, 2021 
            cref_standard: __CREFS__
            cref_factor: __CREFF__ #2.52 #
            Cf: __CF__ # Yus-diez et al, 2021 
            ms: __MS__
            wl_ssa: __SSAWL__
        units: __UNITABS__
        factor: __ABSFAC__
    scattering:
        filename: __SCAF__
        units: __UNITSCA__
        factor: __SCAFAC__
    time_dimension: __TIMED__ #name of the time dimension must be all in UTC
INPUT_opt:
    nmin: __NMIN__
    nmax: __NMAX__
    dnx: __DNX__
    kmin: __KMIN__
    kmax: __KMAX__
    dkx: __DKX__
    nwavel: __NWL__
    wl: __WLF__ #wavelength in nm
    Csca_calculation: __CSCA__
    tmin: __TMIN__ #min angle nephelometer inputs
    tmax: __TMAX__ #max angle between 0 and 180
    nt: __NANG__ #nangles
LUT:
    LUT_folder: __LUTF__
    LUT_flag: __LUTFLAG__ #Put 1 if LUT are present
OUTPUTS:
    PATH_OUT: __PATHOUT__
    lab:  __LAB__
    save_out: __SOUT__
    plot_inputs: __PIN__
    plot_outputs: __POUT__
    plot_size: __PSIZE__
PERIOD:
    dstart: '__DSTART__'
    dend: '__DEND__'
verbose: __VERB__

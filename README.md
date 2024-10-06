```
   ____  ____  ________________  ____
  / __ \/ __ \/_  __/ ____/ __ \/  _/
 / / / / /_/ / / / / /   / /_/ // /
/ /_/ / ____/ / / / /___/ _, _// /
\____/_/     /_/  \____/_/ |_/___/

```
author: Di Antonio Ludovico

mail: ludovico.diantonio@lisa.ipsl.fr
OPTCRI V1.0

---


## Program Version Information

**Version**: `v1.0`
**Reference**: Di Antonio et al, 2024

This program is used to calculate the **aerosol complex refractive index** from:
- **Absorption**
- **Scattering**
- **Size distribution surface data**

---

### Inputs

input data are under inputs folder:

1. aux_inputs contains:
    1. 1.size_diameters.dat (a file with two columns 1: particle diameters (nm) and dlog)
    2. shape_factor.dat for each particle diameter
    3. wl_input.dat, a column of wavelengths in nm for the OPTCRI output i.e. (370, 470, 520,...))
4. config contains an input.sed file to drive the configuration of the program
3. data contains the input tab-separated files of absorption, scattering and size distribution data. Date is in the %Y-%m-%d %H:%M:%S format. the data folder contains a subfolder for the station (i.e. PRG) and other subfolders for the sensitivity tests inputs (i.e. CTRL_mean_SMPS). An example of the file is:


```
inputs/data/PRG/CTRL_mean_SMPS/absorption_CTRL.csv
TIME                    ABS_COEFF(370)          ABS_COEFF(470)          ABS_COEFF(520)          ABS_COEFF(590)          ABS_COEFF(660)          ABS_COEFF(880)          ABS_COEFF(950)
2022-06-15 00:00:00     16.297511078834184      11.97029132079622       10.46485663028755       8.781774642197588       7.633109543864801       5.471914732366452       5.092368220873601
2022-06-15 01:00:00     15.177214676861267      11.47543332318966       10.13322301922112       8.60217719644147        7.495997828071392       5.415566507655092       5.070814452298837

```

5. LUT (contains the Look up tables created from created with the LUT=0 option in the *.sed file)


Before launching the program,
install the conda environment: 

```bash
conda env create -f CONDA_ENV.yaml
```

For the **PRG (urban)** site, modify the following file:

optcri.PRG.sed

and launch the program as:

```bash
./run.sh
```

Please, report any issues to @mail

### Reference

Di Antonio, L., Di Biagio, C., Formenti, P., Gratien, A., Michoud, V., Cantrell, C., Bauville, A., Bergé, A., Cazaunau, M., Chevaillier, S., Cirtog, M., Coll, P., D'Anna, B., de Brito, J. F., De Haan, D. O., Dignum, J. R., Deshmukh, S., Favez, O., Flaud, P.-M., Gaimoz, C., Hawkins, L. N., Kammer, J., Language, B., Maisonneuve, F., Močnik, G., Perraudin, E., Petit, J.-E., Acharja, P., Poulain, L., Pouyes, P., Pronovost, E. D., Riffault, V., Roundtree, K. I., Shahin, M., Siour, G., Villenave, E., Zapf, P., Foret, G., Doussin, J.-F., and Beekmann, M.: Aerosol spectral optical properties in the Paris urban area, and its peri−urban and forested surroundings during summer 2022 from ACROSS surface observations, EGUsphere [preprint], https://doi.org/10.5194/egusphere-2024-2299, 2024. 


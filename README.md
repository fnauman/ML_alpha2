# Exploring helical dynamos with machine learning
[![Binder][binder-badge]][binder-url]
[![Google Colab][colab-badge]][colab-url]

[binder-badge]: https://mybinder.org/badge.svg
[binder-url]: https://mybinder.org/v2/gh/fnauman/ML_alpha2/master
[colab-badge]: https://colab.research.google.com/assets/colab-badge.svg
[colab-url]: https://colab.research.google.com/github/fnauman/ML_alpha2/blob/master/vertical_profiles.ipynb

(Click the Binder or Colab link to open the notebooks work with them in the cloud.)

This repository holds interactive python notebooks in relation to the "Exploring helical dynamos with machine learning" article.

## How to get started

1) Download the helically-forced turbulence simulation dataset from the links provided in the notebooks. Within the notebook, one can download a xy averaged profiles with a command like this 
```
!wget -O mfields_R15e3.npz  https://sid.erda.dk/public/archives/0cf0c2b6d34e20da8971304b06d8f913/pencil/alpha2/shock_k10_R15000_256_xyaver_alpha_eta/mfields.npz
```
2) Use numpy.load to read the file. For example, 
```
mf15 = np.load('mfields_R15e3.npz')
list(mf15)
# ['tres', Resistive time: 1/(kf^2 * eta) (divide tt by tres to get time array in Resistive times)
   'Rm',   Turbulent magnetic Reynolds number: urms/(kf * eta)
   'uave', RMS velocity: urms
   'kf',   Forcing wavemode
   'tt',   Time array (code units)
   'bxm',  xy-averaged B_x field
   'bym',  xy-averaged B_y field
   'b2tot',xy-averaged B^2 TOTAL field
   'u2tot',xy-averaged U^2 TOTAL field
   'emfx', xy-averaged EMF_x field
   'emfy', xy-averaged EMF_y field
   'jxm',  xy-averaged J_x field
   'jym']  xy-averaged J_y field
```
Each of the fields has dimension time x vertical coordinate.
3) install external python packages by running `pip install -r requirements.txt`



## Data 

### Data used for the paper

Our complete data set is available from Zenodo. It consists of a set of helically forced turbulence simulations (256^3) with a varying magnetic Reynolds number (Rm).

### Source for all 3D snapshots including xy-averaged data with the full init/config files
https://sid.erda.dk/public/archives/0cf0c2b6d34e20da8971304b06d8f913/published-archive.html

All the files in pencil/alpha2 correspond to the runs for alpha^2 reported here. The PENCIL code dumps 3D snapshot files for each processor separately so snapshot 5 for Rm = 500 can be accessed here (for proc60):
pencil/alpha2/k10_R500_256_test_xyaver_alpha_eta/data/proc60/VAR5

## Summary of runs reported in the paper

 Name | Rm   | Rm_t   | t_res  | v_rms
 -----|------|--------|--------|-------|
 R5e2 | 500  | 1.68   | 4.97   | 0.0337 
 R1e3 | 1000 |  4.44  | 9.94   | 0.0446
 R2e3 | 2000 | 10.31  | 19.88  | 0.052 
 R3e3 | 3000 | 16.71  | 30.12  | 0.055
 R4e3 | 4000 | 22.64  | 39.76  | 0.057 
 R5e3 | 5000 | 28.72  | 49.70  | 0.058 
 R6e3 | 6000 | 34.61  | 59.63  | 0.058  
 R7e3 | 7000 | 40.41  | 69.58  | 0.058 
 R8e3 | 8000 | 46.22  | 79.52  | 0.058 
 R9e3 | 9000 | 50.16  | 89.55  | 0.056 
 R1e4* | 10000 | 55.79 | 99.40 | 0.056 
 R15e4* | 15000 | 82.12 | 149.1 | 0.055 


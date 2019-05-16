[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/fnauman/ML_alpha2/master)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/fnauman/ML_alpha2/]

Click the Binder or Colab link to open the notebooks in binder and work with them in the cloud.

# Exploring helical dynamos with machine learning

This repository holds interactive python notebooks in relation to the "Exploring helical dynamos with machine learning" article.

## How to get started

1) Download the helically-forced turbulence simulation dataset from zenodo.
2) unpack it to the repository directory, you should now have `alpha2/RmXXXX` directories that store the various compressed (and averaged) simulation runs.
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


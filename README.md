# Exploring helical dynamos with machine learning
[![Binder][binder-badge]][binder-url]
[![Google Colab][colab-badge]][colab-url]

[binder-badge]: https://mybinder.org/badge.svg
[binder-url]: https://mybinder.org/v2/gh/fnauman/ML_alpha2/master
[colab-badge]: https://colab.research.google.com/assets/colab-badge.svg
[colab-url]: https://colab.research.google.com/github/fnauman/ML_alpha2/blob/master/

<!--https://colab.research.google.com/github/fnauman/ML_alpha2/blob/master/vertical_profiles.ipynb -->

(Click the *Binder* or *Colab* links to open the notebooks and work with them in the cloud.)

[arxiv.org/abs/1905.08193](https://arxiv.org/abs/1905.08193)

**Authors**: [Farrukh Nauman](http://fnauman.github.io/) and [Joonas Nättilä](http://natj.github.io/)

**Summary**: We use regularized linear regression, random forests and Bayesian (Markov Chain Monte Carlo) to select the appropriate model for the turbulent electromotive force that feeds large scale magnetic field growth. We find that regularized linear regression performs the best, and this is due to the low dimensional organized dataset (helically forced turbulence leads to high signal to noise ratio) considered here.

## How to get started

1) Download the helically-forced turbulence simulation dataset using the [download_data.ipynb](download_data.ipynb) notebook. 
2) Install external python packages by running `pip install -r requirements.txt`
3) Explore the data using the provided notebooks!

## Content

Currently the analysis consists of:
- temporal and vertical visualizations
   - [temporal_profiles.ipynb](temporal_profiles,ipynb)
   - [vertical_profiles.ipynb](vertical_profiles.ipynb)
- Random forests and LASSO:
   - temporal profile fits 
     - Kinematic: [rfandlasso_temporal_kinematic.ipynb](rfandlasso_temporal_kinematic.ipynb)
     - Close to saturation: [rfandlasso_temporal_saturation.ipynb](rfandlasso_temporal_saturation.ipynb)
   - vertical profile fits
     - Linear basis: [rfandlasso_vertical_nopoly.ipynb](rfandlasso_vertical_nopoly.ipynb)
     - Polynomial basis: [rfandlasso_vertical_poly.ipynb](rfandlasso_vertical_poly.ipynb)
- Bayesian fits (with MCMC) in [mcmc.ipynb](mcmc.ipynb)


## Data 

Data links are provided [here](download_data.ipynb)

Simplest way to download files is:

```
from preprocess import fetch_data
fetch_data() # by default downloads only 4 data files
# Use fetch_data(all=True) to download the entire dataset in the folder alpha2

```

Some notebooks assume that the data is already downloaded and is in a folder "alpha2/". 

If you have trouble running the code, please raise an issue.

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
 
 Data is stored in (compressed) numpy file format. You can load the data using `np.load()` and see different components with `list()`:
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
   
   z-array can be generated like this:
   z_arr = np.linspace(0,2*np.pi,256)
```
Each of the fields have a dimension of `time x vertical` coordinate.


## Citing this work
If you find this work useful, please cite this as:
```
@ARTICLE{2019arXiv190508193N,
       author = {{Nauman}, Farrukh and {N{\"a}ttil{\"a}}, Joonas},
        title = "{Exploring helical dynamos with machine learning}",
      journal = {arXiv e-prints},
     keywords = {Astrophysics - Solar and Stellar Astrophysics, Astrophysics - Astrophysics of Galaxies, Computer Science - Machine Learning},
         year = "2019",
        month = "May",
          eid = {arXiv:1905.08193},
        pages = {arXiv:1905.08193},
archivePrefix = {arXiv},
       eprint = {1905.08193},
 primaryClass = {astro-ph.SR},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2019arXiv190508193N},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```

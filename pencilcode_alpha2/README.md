## PENCIL code configuration files

We used the publicly available finite difference solver (FORTRAN): 
[pencil-code][https://github.com/pencil-code/pencil-code]

Detailed manual on how to install, build and use the code can be found [here][http://pencil-code.nordita.org/doc/manual.pdf].

In this folder, we provide configuration files that are required to run the code once its compiled. 
It should reproduce our results. 
Note that this will create 3D data snapshots + xy-averaged fields. 
The latter was used to train our machine learning models.

[create_mfields.py][create_mfields.py] is the python script that we used to read the xy-averaged data.

```
import pencilnew as pcn

xyave = pcn.read.aver(plane_list=['xy'])
param = pcn.read.param(param2=True)
```

This requires pencilnew to be in your python path [see full instructions][https://github.com/pencil-code/pencil-code/tree/master/python/pencilnew]

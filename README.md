# Ewald_mag
Python based Ewald summation code for evaluating stray magnetic field of finite 3D cuboidal magnetized samples.

Packages required:  numpy, scipy.special, scipy.integrate , pandas.

All relevant functions are in core.py file. Executable file is Ewald_mag.py.
To run code, put init python file, core.py and Ewald_mag.py in same folder.

To run code, run "python Ewald_mag.py" on terminal in working directory. 

Code generates 6 column .dat files for chosen magnetization modes over a plane at a fixed height above magnetized sample inside 'bmaps_serial' folder in working directory of code. 

xpos (x-position of magnetic field probe) | ypos (y- position of magnetic field probe) | zpos (z-position of magnetic field probe | bx (x- component of magnetic field at probe point) | by (y- component of magnetic field at probe point) | bz (z- component of magnetic field at probe point)


README
------

This codebase implements Step-wise orientation method in Cryo-EM 3D structure reconstruction given a set of picked particles from electron micrographs.

This code is provided "as-is" with no warranty or explicit support.  It is made available for personal and academic use ONLY.  Any commercial use of this code without explicit written consent of the authors is strictly prohibited. Use of this in academic publications should cite:

Fast algorithm for determining orientations using angular correlation functions and Bayesian statistics. Lanqing Huang, Haiguang Liu, BioRxiv, 
doi: https://doi.org/10.1101/074732.

Lots of codes are copyright Marcus A. Brubaker and Ali Punjani, 2015 for [cryoem-cvpr2015](https://github.com/mbrubake/cryoem-cvpr2015).

Questions or comments regarding this code should be directed to Lanqing Huang (lqhuang@csrc.ac.cn).


Getting Started
---------------

To generate data, use the genphantomdata.py script.  Running the command:

$ ./genphantomdata.py 40000 1AON.mrc examplectfs.par Data/1AON

will generate a stack of 40,000 randomly distributed views of the density in 1AON.mrc using the CTF parameters specified in the (supplied) par file.

Once data has been generated, you can run the reconstruction with the command:

$ ./runexp.py exp/1AON_sagd_is

and follow it's progress with:

$ tail -f exp/1AON_sagd_is/stdout


Running other data
------------------

Currently the code can only handle specific forms of input.  Images must be in MRC stack files, CTF parameters must be estimated and provided in the form of a defocus.txt or .par file.  The bfactor and other microscope parameters must be set by hand in the exp/*/*.params files.

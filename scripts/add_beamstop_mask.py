#!/usr/bin/env python
from __future__ import print_function, division

import os
import sys
sys.path.append(os.path.dirname(sys.path[0]))
try:
    import cPickle as pickle  # python 2
except ImportError:
    import pickle  # python 3
import socket
import argparse
from random import randint

from matplotlib import pyplot as plt
from numpy import unravel_index, log, maximum

from cryoio import mrc
from geometry import gen_dense_beamstop_mask


parser = argparse.ArgumentParser()
parser.add_argument("mrcs_files", help="list of mrcs files.", nargs='+')

args = parser.parse_args()

mrcs_files = args.mrcs_files

if not isinstance(mrcs_files, list):
    mrcs_files = [mrcs_files]

for ph in mrcs_files:
    M = mrc.readMRC(ph)
    N = M.shape[0]
    mask_3D = gen_dense_beamstop_mask(N, 3, 0.01, psize = 18)

    mrc.writeMRC(ph, M*mask_3D, psz=18)

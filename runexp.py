#!/usr/bin/env python

from __future__ import print_function, division

import sys
import os

usagestr = "Usage: %run -i run.py exp/1"
if len(sys.argv) < 2:
	print(usagestr)
	raise Exception("No Experiment Directory Provided.")
expbase = sys.argv[1]
if not os.path.isdir(expbase):
	print(usagestr)
	raise Exception("Experiment Directory Does not Exist.")
kargs = {}
for a in sys.argv[2:]:
	key,val = a.split('=',1)
	kargs[key] = val

import reconstructor

CO = reconstructor.CryoOptimizer(expbase, kargs)

CO.run()


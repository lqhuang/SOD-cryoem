#!/home/lqhuang/Programs/EMAN2/extlib/bin/python

# import IPython.lib.inputhook

import EMAN2
from EMAN2 import *
from emapplication import EMApp

app = EMApp()
# IPython.lib.inputhook.enable_qt4(app)

from emimage import image_update


image_update()


# EMAN2.GUIMode = True
# EMAN2.app = app

object = EMData("/home/lqhuang/Git/orientation-python/particle/EMD-6044.map")

display(object)

sym = Symmetries.get("c1")
orients = sym.gen_orientations("eman", {"delta":10})

euler_display(orients)

# proj = [object.project("standard", t) for t in orients]

# display(proj)

recon=Reconstructors.get("fourier", {"sym":"c1","size":(pad,pad,pad),"mode":"gauss_2","verbose":True})
recon.setup()
scores = []


# -*- coding: utf-8 -*-

from EMAN2 import *

object = EMData("/home/lqhuang/Git/orientation-python/particle/EMD-6044.map")

# display(object)

sym = Symmetries.get("c1")
orients = sym.gen_orientations("eman", {"delta": 30})

# euler_display(orients)

proj = [object.project("standard", t) for t in orients]

i = enumerate(proj)

print proj 

# display(proj)
# display(object)

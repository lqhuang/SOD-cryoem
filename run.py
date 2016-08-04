import os
import numpy as np 
import matplotlib.pyplot as plt
from cryoio import mrc

V = mrc.readMRC("./particle/1AON.mrc")

plt.plot([1,2,3,4], [2,4,6,8])
plt.show()
# mlab.contour3d(V)

print("Hello")

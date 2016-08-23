import numpy as np
from matplotlib import pyplot as plt

from cryoio import mrc
import geom



file = 'particle/EMD-2325.map'

model = mrc.readMRC(file)

R1 = geom.rotmat3D_EA(0, 0, 0)
R2 = geom.rotmat3D_EA(np.pi/2, np.pi/2, np.pi/2)



proj1 = np.sum(model, axis=2)

proj2 = np.sum(geom.)

plt.imshow(proj1)
plt.show()

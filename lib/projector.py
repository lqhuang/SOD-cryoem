import EMAN2
import os
from EMAN2 import *
from PyQt4 import QtCore, QtGui, QtOpenGL
from emapplication import EMApp
import IPython.lib.inputhook
from emimage import image_update

app = EMApp()
EMAN2.GUIMode = True
EMAN2.app = app


e = EMData()

A = test_image(0)

B = test_image(1)

# display(A)


a = test_image()

display(a)

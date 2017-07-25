from __future__ import print_function, division

import sys

try:
    from PyQt4 import QtCore
    from PyQt4.QtGui import QWidget, QApplication, QVBoxLayout, QSplitter, QMainWindow
except ModuleNotFoundError:
    from PyQt5 import QtCore
    from PyQt5.QtWidgets import QWidget, QApplication, QVBoxLayout, QSplitter, QMainWindow

# from qtvis import *
from cryoio import mrc
import cryoem

class MRCVisualizerQWidget(QWidget):
    def __init__(self, parent=None, mrcfiles=[]):
        QWidget.__init__(self, parent)

        Ms = [mrc.readMRC(mrcfile) for mrcfile in mrcfiles]

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(0)

        for M in Ms:
            self.splitter_main_bottom = QSplitter(self)
            layout.addWidget(self.splitter_main_bottom)

            self.splitter_main_bottom.setOrientation(QtCore.Qt.Horizontal)
            self.sliceplot_widget = SlicePlotQWidget()
            self.splitter_main_bottom.addWidget(self.sliceplot_widget)
            self.densityplot_widget = MayaviQWidget()
            self.splitter_main_bottom.addWidget(self.densityplot_widget)

            self.alignedM,self.R = cryoem.align_density(M, upsamp=1.0)
            self.densityplot_widget.setup(self.alignedM)
            self.sliceplot_widget.setup(M, self.R)


if __name__ == "__main__":
    # Don't create a new QApplication, it would unhook the Events
    # set by Traits on the existing QApplication. Simply use the
    # '.instance()' method to retrieve the existing one.
    app = QApplication.instance()

    print(sys.argv)
    if len(sys.argv) >= 2:
        mrcfiles = sys.argv[1:]
    else:
        assert False, 'Need mrc file as argument'

    # container = MRCVisualizerQWidget(mrcfiles=mrcfiles)
    window = QMainWindow()
    window.setWindowTitle("CryoEM MRC Visualizer")
    # window.setCentralWidget(container)
    window.show()

    # Start the main event loop.
    app.exec_()

from __future__ import print_function, division

# First, and before importing any Enthought packages, set the ETS_TOOLKIT
# environment variable to qt4, to tell Traits that we will use Qt.
import sys
import os
os.environ['ETS_TOOLKIT'] = 'qt4'
# By default, the PySide binding will be used. If you want the PyQt bindings
# to be used, you need to set the QT_API environment variable to 'pyqt'
#os.environ['QT_API'] = 'pyqt'

# To be able to use PySide or PyQt4 and not run in conflicts with traits,
# we need to import QtGui and QtCore from pyface.qt
from pyface.qt import QtGui, QtCore
# Alternatively, you can bypass this line, but you need to make sure that
# the following lines are executed before the import of PyQT:
#   import sip
#   sip.setapi('QString', 2)

from traits.api import HasTraits, Range, Instance, on_trait_change
from traitsui.api import View, Item, Group

from mayavi.core.api import PipelineBase
from mayavi.core.ui.api import MayaviScene, MlabSceneModel, SceneEditor
from mayavi import mlab

# from qtvis import *
from cryoio import mrc
import cryoem
from visualizer import plot_density


# The mrc visualization
class MrcVisualization(HasTraits):
    level = Range(0, 100, 20) #mode='spinner')

    scene = Instance(MlabSceneModel, ())

    density_plot = Instance(PipelineBase)

    # the layout of the dialog screated
    view = View(Item('scene', editor=SceneEditor(scene_class=MayaviScene),
                     height=250, width=300, show_label=False),
                Group('level'),
                resizable=True # We need this to resize with the parent widget
                )

    alignedM = None
    color=(0.75, 0.75, 0.75)
    opacity=1

    @on_trait_change('level,scene.activated')
    def update_plot(self):
        # This function is called when the view is opened. We don't
        # populate the scene when the view is not yet open, as some
        # VTK features require a GLContext.

        # We can do normal mlab calls on the embedded scene.
        # self.scene.mlab.test_points3d()
        if self.alignedM is None:
            pass
        else:
            if self.density_plot is None:
                self.density_plot = self.plot_density(self.alignedM)
            else:
                # FIXME: update plot with specific level of contour
                pass

    def plot_density(self, s, level=0.2, ret_contour=False):
        self.scene.mlab.gcf().scene.background = (1,1,1)
        self.scene.mlab.gcf().scene.foreground = (0,0,0)
        
        src = self.scene.mlab.pipeline.scalar_field(s)
        
        mins = s.min()
        ptps = s.ptp()
        curr_contour = mins + level * ptps

        if ret_contour:
            return src, curr_contour
        else:
            density_plot = self.scene.mlab.pipeline.iso_surface(src, contours=[curr_contour,],
                opacity=self.opacity, color=self.color)
            return density_plot

    def setup(self, alignedM):
        self.alignedM = alignedM


class MayaviQWidget(QtGui.QWidget):
    def __init__(self, parent=None):
        QtGui.QWidget.__init__(self, parent)
        self.curr_layout = QtGui.QVBoxLayout(self)
        self.curr_layout.setContentsMargins(0,0,0,0)
        self.curr_layout.setSpacing(0)

    def setup(self, alignedM, filename=None):
        if filename:
            label = QtGui.QLabel(self)
            label.setText("Model: {}".format(filename))
            label.setAlignment(QtCore.Qt.AlignTop | QtCore.Qt.AlignHCenter)
            self.curr_layout.addWidget(label)

        self.visualization = MrcVisualization()
        self.visualization.setup(alignedM)

        # If you want to debug, beware that you need to remove the Qt
        # input hook.
        #QtCore.pyqtRemoveInputHook()
        #import pdb ; pdb.set_trace()
        #QtCore.pyqtRestoreInputHook()

        # The edit_traits call will generate the widget to embed.
        self.ui = self.visualization.edit_traits(parent=self,
                                                 kind='subpanel').control
        self.curr_layout.addWidget(self.ui)
        self.ui.setParent(self)


class MRCVisualizerQWidget(QtGui.QWidget):
    def __init__(self, parent=None, mrcfiles=[]):
        QtGui.QWidget.__init__(self, parent)

        Ms = [mrc.readMRC(mrcfile) for mrcfile in mrcfiles]

        layout = QtGui.QHBoxLayout(self)
        layout.setContentsMargins(0,0,0,0)
        layout.setSpacing(0)

        for i, M in enumerate(Ms):
            filename = os.path.basename(mrcfiles[i])

            self.splitter_main_bottom = QtGui.QSplitter(self)
            layout.addWidget(self.splitter_main_bottom)

            self.splitter_main_bottom.setOrientation(QtCore.Qt.Horizontal)
            # self.sliceplot_widget = SlicePlotQWidget()
            # self.splitter_main_bottom.addWidget(self.sliceplot_widget)
            self.densityplot_widget = MayaviQWidget()
            self.splitter_main_bottom.addWidget(self.densityplot_widget)

            self.alignedM,self.R = cryoem.align_density(M, upsamp=1.0)
            self.densityplot_widget.setup(self.alignedM, filename=filename)
            # self.sliceplot_widget.setup(M, self.R)


if __name__ == "__main__":
    # Don't create a new QApplication, it would unhook the Events
    # set by Traits on the existing QApplication. Simply use the
    # '.instance()' method to retrieve the existing one.
    app = QtGui.QApplication.instance()

    print(sys.argv)
    if len(sys.argv) >= 2:
        mrcfiles = sys.argv[1:]
    else:
        assert False, 'Need mrc file as argument'

    container = MRCVisualizerQWidget(mrcfiles=mrcfiles)
    window = QtGui.QMainWindow()
    window.setWindowTitle("CryoEM MRC Visualizer")
    window.setCentralWidget(container)
    window.show()

    # Start the main event loop.
    app.exec_()

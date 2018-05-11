import os.path as op
import glob
import subprocess
import time
import xml.etree.ElementTree as ET

#from shapely.geometry import Polygon
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from descartes.patch import PolygonPatch
from collections import OrderedDict as odict
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.collections import PatchCollection

from tools import *

import lost
import mesh as meshmod

class MeshTools:
    @staticmethod
    def buildModelFileFromMesh(basename, reset_potential, threshold_potential):
        bind_exe = op.join(getMiindAppsPath(), 'MatrixGenerator', 'Bind')
        subprocess.call([
          bind_exe,
          basename + '.mesh',
          basename + '.stat',
          basename + '.rev',
          str(reset_potential),
          str(threshold_potential)
          ])

    @staticmethod
    def generateStubFidFile(basename):
        with open(basename + '.fid', 'w') as fidfile:
            fid = ET.Element('Fiducual')
            fidfile.write(ET.tostring(fid))
        return basename + '.fid'

    @staticmethod
    def buildMatrixFileFromModel(basename, spike_shift_v, fidfile=None, num_mc_points=10,
                                    spike_shift_w=0, reset_shift_w=0):
        if not fidfile:
            fidfile = ModelGenerator.generateStubFidFile(basename)

        matrix_generator_exe = op.join(getMiindAppsPath(), 'MatrixGenerator', 'MatrixGenerator')
        subprocess.call([
          matrix_generator_exe,
          basename + '.model',
          fidfile,
          str(num_mc_points),
          str(spike_shift_v),
          str(spike_shift_w),
          str(reset_shift_w)
          ])

    @staticmethod
    def plotMesh(filename, ax=None):
        mesh = meshmod.Mesh(filename)
        # Mesh puts an empty cell (used for the stationary strip) by default
        # remove it before displaying
        mesh.cells.pop(0)

        polygons = odict(
            ((i, j),
            Polygon([(float(x), float(y))
                     for x, y in zip(cell.vs, cell.ws)]))
            for i, cells in enumerate(mesh.cells)
            for j, cell in enumerate(cells)
        )

        patches = [
            PolygonPatch(polygon)
            for polygon in polygons.values()
        ]

        if ax is None:
            fig, ax = plt.subplots()
        md = mesh.dimensions()
        p = PatchCollection(patches, alpha=1, edgecolors='k',
                            facecolors='w')
        ax.add_collection(p)
        ax.set_xlim(md[0])
        ax.set_ylim(md[1])

        def onclick(event):
            toolbar = plt.get_current_fig_manager().toolbar
            if (event.button is not 1) or (event.ydata is None) or (event.xdata is None) or toolbar.mode!='':
                return
            point = Point(event.xdata, event.ydata)
            for poly_key, poly_val in polygons.iteritems():
                if poly_val.contains(point):
                    print 'Clicked Cell ' + str(poly_key[0]) + ', ' + str(poly_key[1])

        cid = fig.canvas.mpl_connect('button_press_event', onclick)

        plt.show()
        return ax

    @staticmethod
    def plotLost(lost_path, **kwargs):
        print '\nPoints indicate locations of lost mass in the transition matrix. (No points = No mass loss)'
        print 'Click four locations to draw a quad to surround an area of points.'
        print 'Continue adding quads until all points are covered.'
        print 'This does not need to be accurate and you should ensure that quads cover ares of possible loss as well as where points are located.'
        print '(Think about the mesh and where there are gaps).'
        print 'Smaller quads decrease the search time for the Matrix Generator.'
        print '\n'
        print 'Left Click to place points.'
        print 'Mouse Wheel to zoom in and out.'
        print '\'d\' to delete the quad currently under the mouse pointer.'
        print '\'c\' to clear all quads.'
        print 'Double-Click to write the created quads to the Fiducial file and quit.'
        print '\n'
        from tools_lost import (add_fiducial, extract_base,
                                    plot_lost, read_fiducial,
                                    onclick, zoom_fun, onkey)
        backend = matplotlib.get_backend().lower()
        if backend not in ['qt4agg', 'qt5agg']:
            print('Warning: backend not recognized as working with "lost.py", ' +
                  'if you do not encounter any issues with your current backend ' +
                  '{}, please add it to this list.'.format(backend))
        curr_points = []
        fig = plt.figure()
        ax = plot_lost(lost_path)
        fid_fname = extract_base(lost_path) + '.fid'
        patches = read_fiducial(fid_fname)
        quads = copy.deepcopy(patches)
        for patch in patches:
            add_fiducial(ax, patch)

        fig.canvas.mpl_connect('button_press_event',
                               lambda event: onclick(event, ax, fid_fname,
                                                     curr_points, quads))
        fig.canvas.mpl_connect('scroll_event', lambda event: zoom_fun(event, ax))
        fig.canvas.mpl_connect('key_press_event',
                               lambda event: onkey(event, ax, fid_fname, quads))
        plt.show()

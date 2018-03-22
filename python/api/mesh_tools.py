import os.path as op
from tools import *
import glob
import subprocess
import time
import xml.etree.ElementTree as ET
import lost
from shapely.geometry import Polygon
from descartes.patch import PolygonPatch
import mesh as meshmod
from collections import OrderedDict as odict

class ModelGenerator:
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
    def lost(filename):
        lost.main(['lost.py', filename])

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
        return ax

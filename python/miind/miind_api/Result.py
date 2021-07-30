import os.path as op
import glob
import subprocess
import time
import matplotlib.pyplot as plt

from collections import OrderedDict as odict
from shapely.geometry import Polygon
from descartes.patch import PolygonPatch
from matplotlib.collections import PatchCollection

from .tools import *

import miind.mesh3 as meshmod

class Result(object):
    def __init__(self, io, nodename):
        self.io = io
        self.nodename = nodename
        self.nodeindex, model = self.io.getModelFilenameAndIndexFromNode(nodename)
        self.modelname, self.modelfname = split_fname(model, '.model')
        self.modelpath = op.join(self.io.xml_location, self.modelfname)

    @property
    def fnames(self):
        self._fnames, self._times = find_density_fnames(
            self.modelfname, self.nodeindex, self.io.getOutputDirectory())
        return self._fnames

    @property
    def times(self):
        if not hasattr(self, '_times'):
            self.fnames
        return self._times

    @property
    def mesh(self):
        if not hasattr(self, '_mesh'):
            mesh = meshmod.Mesh(None)
            mesh.FromXML(self.modelpath)
            self._mesh = mesh
        return self._mesh

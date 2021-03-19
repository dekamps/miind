import xml.etree.ElementTree as ET
import xml.dom.minidom as minidom
from pprint import pprint
import json
import os
import os.path as op
import glob
import numpy as np
import copy
from collections import Mapping, OrderedDict
import miind.directories3 as directories


class cd:
    """Context manager for changing the current working directory"""
    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)

def getMiindBuildPath(): #Used in development mode
    return os.path.join(directories.miind_python_dir(), '..', '..', 'build')

def getMiindPythonPath():
    return os.path.join(directories.miind_python_dir())

def getMiindAppsPath():
    
    # if miind has been installed, try to find apps in .../share/python/miind/apps
    build_path = os.path.join(directories.miind_python_dir(), '..', '..', 'apps')

    # if this is python miind, try to find apps in .../miind/build/apps
    if not os.path.isdir(build_path):
        build_path = os.path.join(directories.miind_python_dir(), 'build', 'apps')
    # otherwise, we're in dev mode (build directory)
    if not os.path.isdir(build_path):
        build_path = os.path.join(getMiindBuildPath(), 'apps')

    return build_path

def split_fname(fname, ext):
    fname = op.split(fname)[1]
    if not ext.startswith('.'):
        ext = '.' + ext
    if fname.endswith(ext):
        modelname = op.splitext(fname)[0]
        modelfname = fname
    else:
        modelname = fname
        modelfname = fname + ext
    return modelname, modelfname

def replace(value, string, *args):
    for a in args:
        value = value.replace(a, string)
    return value

def find_density_fnames(modelfname, nodeindex, directory):
    fnames = glob.glob(op.join(directory, modelfname + '_mesh', 'density_mesh_' + str(nodeindex) + '*'))
    if len(fnames) == 0:
        fnames = glob.glob(op.join(directory, 'densities', 'node_' + str(nodeindex) + '*'))
        if len(fnames) == 0:
            raise ValueError('No density output found for node index : {}'.format(nodeindex))

    fnames = sorted(fnames, key=get_density_time)
    times = [get_density_time(f) for f in fnames]
    return fnames, times

def read_density(filename):
    f = open(filename, 'r')
    line = f.readline().split()
    data = [float(x) for x in line[2::3]]
    coords = [(int(i), int(j)) for i, j in zip(line[::3], line[1::3])]
    return data, coords

def get_density_time(path):
    fname = op.split(path)[-1]
    return fname.split('_')[-2]

def calc_mass(mesh, density, coords):
    masses = [mesh.cells[i][j].area * dens
              for (i, j), dens in zip(coords, density)]
    return masses

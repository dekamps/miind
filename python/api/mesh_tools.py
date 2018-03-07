import os.path as op
from tools import *
import glob
import subprocess
import time
import xml.etree.ElementTree as ET

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

#!/usr/bin/env python3

import sys
import pylab
import numpy
import subprocess
import matplotlib.pyplot as plt
from miind.miind_api.tools import *

matrix_generator_exe = op.join(getMiindAppsPath(), 'MatrixGenerator', 'MatrixGenerator')
print(subprocess.check_output([
              matrix_generator_exe,
              'area',
              'not_there.model',
              'not_there.fid',
              '1000',
              '0.1',
              '0.0',
              '0.0',
              '-use_area_calculation'
              ]))




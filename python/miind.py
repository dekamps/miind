#!/usr/bin/env python

import argparse
import codegen
import sys
import os
import directories
import jobs


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description='Converts XML files into executable')

    parser.add_argument('--d', help = 'Provide a packaging directory.',nargs = '?')
    parser.add_argument('xml file', metavar='XML File', nargs = '*', help = 'Will create an entry in the build tree for each XML file, provided the XML file is valid.')
    parser.add_argument('--m', help = 'A list of model and matrix files that will be copied to every executable directory.''', nargs='+')
    parser.add_argument('--mpi', help = 'Enable MPI.', action='store_true')
    parser.add_argument('--openmp', help = 'Enable OPENMP.', action='store_true')
    parser.add_argument('--no_root', help = 'Disable ROOT.', action='store_true')
    args = parser.parse_args()

    filename = vars(args)['xml file']
    dirname  = vars(args)['d']
    modname  = vars(args)['m']
    enable_mpi = vars(args)['mpi']
    enable_openmp = vars(args)['openmp']
    disable_root = vars(args)['no_root']

    if dirname == None:
        fn = filename[0]
        directories.add_executable(fn,modname, enable_mpi, enable_openmp, disable_root)    
    else:
        directories.add_executable(dirname, filename, modname, enable_mpi, enable_openmp, disable_root)


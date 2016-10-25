#!/usr/bin/env python

import directories
import sys
import os
import subprocess

class cd:
    """Context manager for changing the current working directory"""
    def __init__(self, newPath):
        self.newPath = os.path.expanduser(newPath)

    def __enter__(self):
        self.savedPath = os.getcwd()
        os.chdir(self.newPath)

    def __exit__(self, etype, value, traceback):
        os.chdir(self.savedPath)


if __name__ == "__main__":
    
    if len(sys.argv) != 2:
        print 'submit requirs exactly one argument: the subdirectory under which the jobs are organized.'
        sys.exit()
    dir=sys.argv[1]

    # investigate the directory
    with cd(dir):
        files=subprocess.check_output(["ls"]).split()
        if 'CMakeLists.txt' in files:
            subprocess.call(['cmake', '-DCMAKE_BUILD_TYPE=Release'])
            subprocess.call(['make'])
        else:
            # all directories in this one should contain a CMakeLists.txt
            for file in files:
                with cd(file):
                    localfiles=subprocess.check_output(["ls"]).split()
                    
                    if 'CMakeLists.txt' in localfiles:
                        subprocess.call(['cmake','-DCMAKE_BUILD_TYPE=Release', '.'])
                        subprocess.call(['make'])
                        # file name is the same as directory name, if miind was called
                        subprocess.call('./' + file)

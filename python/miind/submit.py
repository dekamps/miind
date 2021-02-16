#!/usr/bin/env python

import miind.directories3 as directories
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
    
    # always copy sub.sh to the calling directory, unless it's already there
    # this gives the user the opportunity to adapt sub.sh to the local queueing system
    subpath = os.path.join(directories.miind_root(),'python')

    # either there are two arguments
    if len(sys.argv) == 1 or  len(sys.argv) > 4:
        print ('Either a single argument containing the local submission directory, which is where the executable will be run, or')
        print ('two arguments, which forces queue submission, where the second is the directory where submit.py is called from, or ')
        print ('three arguments with the third argument a local filename containing and alternative script for sub.sh')
        sys.exit()

    dir=sys.argv[1]

    if len(sys.argv) == 2:
        local = True
        queue = False

    if len(sys.argv) == 3:
        queue = True	
        local = False
        calldir = sys.argv[2] 
        if not os.path.exists(os.path.join(calldir,"sub.sh")):
            subprocess.call(["cp",os.path.join(subpath,"sub.sh"),calldir])    

    if len(sys.argv) == 4:
        raise UnImplemented


    # or there are 3/4 with the thirds argument indicating that the submission script will be used for queue submission
    # and the 4th argument the filename of a local file that should replace the miind provided submission script.
    
    # investigate the directory
    with cd(dir):
        files=subprocess.check_output(["ls"]).split()
        if 'CMakeLists.txt' in files:
            subprocess.call(['cmake', '-DCMAKE_BUILD_TYPE=Release'])
            subprocess.call(['make'])
        else:
            # all directories in this one should contain a CMakeLists.txt
            for fi in files:
                with cd(fi):
                    localfiles=subprocess.check_output(["ls"]).split()
                    # subprocess returns bytes, and this difference is important in Python 3
                    if b'CMakeLists.txt' in localfiles:
                        subprocess.call(['cmake','-DCMAKE_BUILD_TYPE=Release', '.'])
                        subprocess.call(['make'])
			# copy the submission shells script from the calling directory
			# this job could have done that directly, but this way the user can adapt the submission script in the calling directory
                        subprocess.call(['cp',os.path.join(subpath,'sub.sh'),'.'])
                        # file name is the same as directory name, if miind was called
                    
                        if local == True:
                            subprocess.call('./' + fi.decode('ascii'))
                        if queue == True:
                            subprocess.call(['qsub', 'sub.sh', './' + fi.decode('ascii')])

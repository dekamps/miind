#!/usr/bin/env python

import subprocess 
import miind.submit as submit
import sys
import os
import miind.directories as directories

dir=sys.argv[1]

build = os.path.join(directories.miind_root(),'build')
path  = os.path.join(build,'jobs',dir)

with submit.cd(os.path.join(directories.miind_root(),'python')):
	subprocess.call(["cp","sub.sh",path])

with submit.cd(build):
        subprocess.call(['make'])

with submit.cd(path):
	f=open('joblist')
	lines=f.readlines()
	for line in lines:
		name=line.strip()
		subprocess.call(['qsub','./sub.sh',name])

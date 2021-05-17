#!/usr/bin/env python3

import sys
import os
import shutil
import miind.directories3

# Copy examples to current directory
print('Copying example files to: ', os.getcwd() + '/examples') 
file_dir = miind.directories3.miind_python_dir() + '/build/examples/'
shutil.copytree(file_dir, os.getcwd() + '/examples')
print('Success.')
#!/bin/bash

echo "Manylinux2014 yum cannot install python3-devel so for python 3.8 and 3.9, we need to manually download and install them to get access to Python.h for the extension."

yum install gcc openssl-devel bzip2-devel libffi-devel -y
curl -O https://www.python.org/ftp/python/3.8.1/Python-3.8.1.tgz
tar -xzf Python-3.8.1.tgz
cd Python-3.8.1/
./configure --enable-optimizations
make install

echo "Installed Python 3.8. Attempting to install Python 3.9."

yum install gcc openssl-devel bzip2-devel libffi-devel -y
curl -O https://www.python.org/ftp/python/3.9.1/Python-3.9.1.tgz
tar -xzf Python-3.9.1.tgz
cd Python-3.9.1/
./configure --enable-optimizations
make install


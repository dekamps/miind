#!/bin/bash

echo "Manylinux2014 yum cannot install python3-devel so for python 3.8 and 3.9, we need to manually download, build from source, and install to get access to Python.h for the extension."

version=$(python -V 2>&1 | grep -Po '(?<=Python )(.+)')
if [[ -z "$version" ]]
then
    echo "No Python!" 
fi

parsedVersion=$(echo "${version//./}")

echo "Found Python Version: $parsedVersion"

if [[ "$parsedVersion" -eq "370" ]]
then 
    yum install gcc openssl-devel bzip2-devel libffi-devel -y
	curl -O https://www.python.org/ftp/python/3.7.1/Python-3.7.1.tgz
	tar -xzf Python-3.7.1.tgz
	cd Python-3.7.1/
	./configure --enable-optimizations
	make install
fi

if [[ "$parsedVersion" -eq "380" ]]
then 
    yum install gcc openssl-devel bzip2-devel libffi-devel -y
	curl -O https://www.python.org/ftp/python/3.8.1/Python-3.8.1.tgz
	tar -xzf Python-3.8.1.tgz
	cd Python-3.8.1/
	./configure --enable-optimizations
	make install
fi

if [[ "$parsedVersion" -eq "390" ]]
then 
    yum install gcc openssl-devel bzip2-devel libffi-devel -y
	curl -O https://www.python.org/ftp/python/3.9.1/Python-3.9.1.tgz
	tar -xzf Python-3.9.1.tgz
	cd Python-3.9.1/
	./configure --enable-optimizations
	make install
fi


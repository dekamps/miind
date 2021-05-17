#!/bin/bash

echo "Manylinux2014 yum cannot install python3-devel so for python 3.7, 3.8 and 3.9, we need to manually download, build from source, and install to get access to Python.h for the extension."

version=$(python -V 2>&1 | grep -Po '(?<=Python )(.+)')
if [[ -z "$version" ]]
then
    echo "No Python!" 
fi

parsedVersion=$(echo "${version//./}")
parsedVersionRed=$(echo "${parsedVersion:0:3}")

echo "Found Python Version: $parsedVersionRed"

if [[ "$parsedVersionRed" -gt "360" && "$parsedVersionRed" -lt "370" ]]
then 
    yum install gcc openssl-devel bzip2-devel libffi-devel -y
	curl -O https://www.python.org/ftp/python/3.6.1/Python-3.6.1.tgz
	tar -xzf Python-3.6.1.tgz
	cd Python-3.6.1/
	./configure --enable-optimizations
	make install
fi

if [[ "$parsedVersionRed" -gt "370" && "$parsedVersionRed" -lt "380" ]]
then 
    yum install gcc openssl-devel bzip2-devel libffi-devel -y
	curl -O https://www.python.org/ftp/python/3.7.1/Python-3.7.1.tgz
	tar -xzf Python-3.7.1.tgz
	cd Python-3.7.1/
	./configure --enable-optimizations
	make install
fi

if [[ "$parsedVersionRed" -gt "380" && "$parsedVersionRed" -lt "390" ]]
then 
    yum install gcc openssl-devel bzip2-devel libffi-devel -y
	curl -O https://www.python.org/ftp/python/3.8.1/Python-3.8.1.tgz
	tar -xzf Python-3.8.1.tgz
	cd Python-3.8.1/
	./configure --enable-optimizations
	make install
fi

if [[ "$parsedVersionRed" -gt "390" && "$parsedVersionRed" -lt "400" ]]
then 
    yum install gcc openssl-devel bzip2-devel libffi-devel -y
	curl -O https://www.python.org/ftp/python/3.9.1/Python-3.9.1.tgz
	tar -xzf Python-3.9.1.tgz
	cd Python-3.9.1/
	./configure --enable-optimizations
	make install
fi


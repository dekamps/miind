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


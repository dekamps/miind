# Use the unofficial root docker as a parent image
FROM nvidia/cuda

USER root

# Set the working directory
WORKDIR /usr/share

# Install any needed packages
RUN apt update
RUN DEBIAN_FRONTEND=noninteractive apt install cmake-curses-gui g++ libx11-dev libxpm-dev libxft-dev libxext-dev binutils lsb-core libboost-all-dev libgsl0-dev libfftw3-dev freeglut3-dev mesa-utils libxmu-dev libxi-dev python3 python3-pip python3-scipy python3-tk openmpi-bin openssh-client openssh-server libopenmpi-dev nvidia-cuda-toolkit gcc-6 g++-6 -y
RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-6 60 --slave /usr/bin/g++ g++ /usr/bin/g++-6
RUN python3 -mpip install matplotlib
RUN python3 -mpip install shapely
RUN python3 -mpip install descartes
RUN python3 -mpip install mpi4py

# Define python include for boost python
ENV CPLUS_INCLUDE_PATH "$CPLUS_INCLUDE_PATH:/usr/include/python3.6/" 

WORKDIR /root

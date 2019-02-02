# Use the unofficial root docker as a parent image
FROM nvidia/cuda

USER root

# Set the working directory to /app
WORKDIR /usr/share

# Copy the current directory contents into the container at /app
ADD miind_tar /usr/share

RUN tar -xzf miind.tar.gz

# Install any needed packages specified in requirements.txt
RUN apt-get update
RUN apt-get -y install apt-utils
RUN apt-get -y install cmake
RUN apt-get -y install cmake-curses-gui
RUN apt-get -y install python python-dev
RUN apt-get -y install libx11-dev
RUN apt-get -y install libxpm-dev
RUN apt-get -y install libxft-dev
RUN apt-get -y install libxext-dev
RUN apt-get -y install binutils
RUN apt-get -y install lsb-core
RUN apt-get -y install python3-scipy
RUN apt-get -y install libgsl0-dev
RUN apt-get -y install libnl-utils
RUN apt-get -y install libfftw3-dev
RUN apt-get -y install freeglut3-dev
RUN apt-get -y install mesa-utils
RUN apt-get -y install nvidia-cuda-toolkit
RUN apt-get -y install libxmu-dev libxi-dev
RUN apt-get -y install python3-pip python3-dev build-essential
RUN apt-get -y install vim
RUN apt-get -y install mlocate
RUN apt-get -y install wget
RUN pip3 install --upgrade pip
RUN python3 -mpip install matplotlib
RUN python3 -mpip install shapely
RUN python3 -mpip install descartes
RUN apt-get -y install openmpi-bin openssh-client openssh-server libopenmpi-dev

# Define python include for boost python
ENV CPLUS_INCLUDE_PATH "$CPLUS_INCLUDE_PATH:/usr/include/python3.6/" 

# Add environment variables to bashrc
RUN /bin/bash -c "echo 'CPLUS_INCLUDE_PATH="\$CPLUS_INCLUDE_PATH:/usr/include/python3.6/"' >> ~/.bashrc"
RUN /bin/bash -c "echo 'PATH="\$PATH:/usr/share/miind/python"' >> ~/.bashrc"
RUN /bin/bash -c "echo 'PYTHONPATH="/usr/lib:\$PYTHONPATH:/usr/share/miind/python"' >> ~/.bashrc"
RUN /bin/bash -c "echo 'LD_LIBRARY_PATH="/usr/lib/:/usr/local/lib/:\$LD_LIBRARY_PATH"' >> ~/.bashrc"
RUN /bin/bash -c "echo 'OMP_NUM_THREADS=1' >> ~/.bashrc"
RUN /bin/bash -c "echo 'export PATH PYTHONPATH CPLUS_INCLUDE_PATH LD_LIBRARY_PATH OMP_NUM_THREADS' >> ~/.bashrc"
#RUN /bin/bash -c "echo 'source /usr/local/bin/thisroot.sh' >> ~/.bashrc"

WORKDIR /root

RUN /bin/bash -c "cp -r /usr/share/miind/examples ."

WORKDIR /root/examples

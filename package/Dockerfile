# Use the unofficial root docker as a parent image
FROM rootproject/root-ubuntu16

USER root

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
ADD . /app

# Install any needed packages specified in requirements.txt
RUN apt-get update
RUN apt-get -y install libx11-dev
RUN apt-get -y install libxpm-dev
RUN apt-get -y install libxft-dev
RUN apt-get -y install libxext-dev
RUN apt-get -y install binutils
RUN apt-get -y install lsb-core
RUN apt-get -y install python-scipy
RUN apt-get -y install libboost-all-dev
RUN apt-get -y install libgsl0-dev
RUN apt-get -y install libnl-utils
RUN apt-get -y install python-pip python-dev build-essential
RUN pip install --upgrade pip
RUN python -mpip install matplotlib
RUN apt-get -y install python-tk
RUN python -mpip install shapely
RUN python -mpip install descartes
RUN python -mpip install mpi4py
RUN apt -y install ./miind_1.04-1_amd64.deb
RUN /bin/bash -c "source /usr/local/bin/thisroot.sh"

# Define python include for boost python
ENV CPLUS_INCLUDE_PATH "$CPLUS_INCLUDE_PATH:/usr/include/python2.7/" 

WORKDIR /usr/share/miind/build

RUN /bin/bash -c "cmake .. -DCMAKE_BUILD_TYPE=Release -DENABLE_MPI=ON -DENABLE_ROOT=ON -DENABLE_OPENMP=ON"
RUN /bin/bash -c "make"

# Add environment variables to bashrc
RUN /bin/bash -c "echo 'CPLUS_INCLUDE_PATH="$CPLUS_INCLUDE_PATH:/usr/include/python2.7/"' >> ~/.bashrc"
RUN /bin/bash -c "echo 'PATH="$PATH:/usr/share/miind/python"' >> ~/.bashrc"
RUN /bin/bash -c "echo 'PYTHONPATH="/usr/lib:$PYTHONPATH:/usr/share/miind/python"' >> ~/.bashrc"
RUN /bin/bash -c "echo 'LD_LIBRARY_PATH="/usr/lib/:$LD_LIBRARY_PATH"' >> ~/.bashrc"
RUN /bin/bash -c "echo 'OMP_NUM_THREADS=1' >> ~/.bashrc"
RUN /bin/bash -c "echo 'export PATH PYTHONPATH CPLUS_INCLUDE_PATH LD_LIBRARY_PATH OMP_NUM_THREADS' >> ~/.bashrc"
RUN /bin/bash -c "echo 'source /usr/local/bin/thisroot.sh' >> ~/.bashrc"

WORKDIR ~

RUN /bin/bash -c "cp -r /usr/share/miind/examples ."

WORKDIR ~/examples

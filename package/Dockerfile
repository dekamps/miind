# Use the unofficial root docker as a parent image
FROM nvidia/cuda

USER root

# Set the working directory
WORKDIR /usr/share

# Install any needed packages specified in requirements.txt
RUN python3 -mpip install matplotlib
RUN python3 -mpip install shapely
RUN python3 -mpip install descartes
RUN python3 -mpip install mpi4py

# Define python include for boost python
ENV CPLUS_INCLUDE_PATH "$CPLUS_INCLUDE_PATH:/usr/include/python3.6/" 

WORKDIR /root

RUN /bin/bash -c "cp -r /usr/share/miind/examples ."

WORKDIR /root/examples

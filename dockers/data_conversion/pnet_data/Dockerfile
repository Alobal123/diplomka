# Start with ubuntu base image
FROM ubuntu:bionic

WORKDIR /
# Install basic utils and python
RUN apt-get update -yq && apt-get install -yq \
    apt-utils \
    build-essential \
    python3-dev \
    python3-pip \
    python3-numpy \
    libhdf5-dev \
    cmake \
    git

# Install python libraries        
RUN pip3 install h5py sobol_seq
RUN pip3 install git+git://github.com/fwilliams/point-cloud-utils

# Copy changed files
COPY *.py ./
COPY *.ini ./

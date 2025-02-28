FROM mcr.microsoft.com/devcontainers/python:1-3.13-bullseye
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get install -y tzdata && \
    ln -fs /usr/share/zoneinfo/Asia/Tokyo /etc/localtime && \
    dpkg-reconfigure --frontend noninteractive tzdata
RUN pip3 install --upgrade pip
RUN apt-get update \
    && apt-get install -y wget \
    bash \
    git \
    gcc \
    g++ \
    gfortran \
    liblapack-dev \
    libamd2 \
    libcholmod3 \
    libopencv-dev \
    libmetis-dev \
    libsuitesparse-dev \
    libnauty2-dev \
    sqlite3 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && pip3 install --upgrade setuptools wheel pip

FROM mcr.microsoft.com/devcontainers/python:1.2.4-3.13-bookworm
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
    libamd2 \
    libcholmod3 \
    liblapack-dev \
    libmetis-dev \
    libnauty2-dev \
    libopencv-dev \
    libsqlite3-dev \
    libsuitesparse-dev \
    sqlite3 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && pip3 install --upgrade setuptools wheel pip
WORKDIR /usr/src/
COPY ./apps /usr/src/apps
COPY ./local.sqlite /usr/src/local.sqlite
COPY ./requirements.txt /usr/src/requirements.txt
RUN pip install --default-timeout=100 -r requirements.txt
ENV FLASK_APP="apps.app:create_app('local')"
ENV IMAGE_URL="/storage/images/"
EXPOSE 5000
CMD [ "flask", "run", "-h", "0.0.0.0" ]

FROM ubuntu:18.04
MAINTAINER Sergio GQ <sergioalbertogq@gmail.com>

# Supress warnings about missing front-end. As recommended at:
# http://stackoverflow.com/questions/22466255/is-it-possibe-to-answer-dialog-questions-when-installing-under-docker
ARG DEBIAN_FRONTEND=noninteractive

# Essentials: developer tools, build tools, OpenBLAS
RUN apt-get update && apt-get install -y --no-install-recommends \
apt-utils git curl vim unzip openssh-client wget \
build-essential cmake \
libopenblas-dev

RUN apt-get update && apt-get install -y software-properties-common gcc && \
    add-apt-repository -y ppa:deadsnakes/ppa

#
# Python 3.6
#
# For convenience, alias (but don't sym-link) python & pip to python3 & pip3 as recommended in:
# http://askubuntu.com/questions/351318/changing-symlink-python-to-python3-causes-problems
RUN apt-get update && apt-get install -y python3.6 python3-distutils python3-pip python3-apt python3-tk && \
pip3 install --no-cache-dir --upgrade pip setuptools && \
echo "alias python='python3'" >> /root/.bash_aliases && \
echo "alias pip='pip3'" >> /root/.bash_aliases
# Pillow and it's dependencies
RUN apt-get install -y --no-install-recommends libjpeg-dev zlib1g-dev
RUN pip3 --no-cache-dir install Pillow
# Science libraries and other common packages
RUN pip3 --no-cache-dir install \
numpy scipy sklearn scikit-image==0.13.1 pandas matplotlib Cython requests rake_nltk turicreate

#
# Jupyter Notebook
#
# Allow access from outside the container, and skip trying to open a browser.
# NOTE: disable authentication token for convenience. DON'T DO THIS ON A PUBLIC SERVER.
RUN pip3 --no-cache-dir install jupyter && \
mkdir /root/.jupyter && \
echo "c.NotebookApp.ip = '0.0.0.0'" \
"\nc.NotebookApp.open_browser = False" \
"\nc.NotebookApp.token = ''" \
> /root/.jupyter/jupyter_notebook_config.py
EXPOSE 8888

#
# Tensorflow 1.6.0 - CPU
#
RUN pip3 install --no-cache-dir --upgrade tensorflow tensorflow_hub 

# Expose port for TensorBoard
EXPOSE 6006

#
# OpenCV 3.4.1
#
# Dependencies
RUN add-apt-repository "deb http://security.ubuntu.com/ubuntu xenial-security main" && apt update && apt install libjasper1 libjasper-dev
RUN apt-get install -y --no-install-recommends \
libjpeg8-dev libtiff5-dev libpng-dev \
libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libgtk2.0-dev \
liblapacke-dev checkinstall
# Get source from github
RUN git clone -b 3.4.1 --depth 1 https://github.com/opencv/opencv.git /usr/local/src/opencv
# Compile
RUN cd /usr/local/src/opencv && mkdir build && cd build && \
cmake -D CMAKE_INSTALL_PREFIX=/usr/local \
-D BUILD_TESTS=OFF \
-D BUILD_PERF_TESTS=OFF \
-D PYTHON_DEFAULT_EXECUTABLE=$(which python3) \
.. && \
make -j"$(nproc)" && \
make install

#
# Caffe
#
# Dependencies
RUN apt-get install -y --no-install-recommends \
cmake libprotobuf-dev libleveldb-dev libsnappy-dev libopencv-dev \
libhdf5-serial-dev protobuf-compiler liblmdb-dev libgoogle-glog-dev \
libboost-all-dev && \
pip3 install lmdb
RUN apt install -y caffe-cpu

#
# Java
#
# Install JDK (Java Development Kit), which includes JRE (Java Runtime
# Environment). Or, if you just want to run Java apps, you can install
# JRE only using: apt install default-jre
RUN apt-get install -y --no-install-recommends default-jdk

#
# Keras 2.1.5
#
RUN pip3 install --no-cache-dir --upgrade h5py pydot_ng keras

#
# PyTorch 0.3.1
#
RUN pip3 install torch==1.8.1+cpu torchvision==0.9.1+cpu torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

#
# PyCocoTools
#
# Using a fork of the original that has a fix for Python 3.
# I submitted a PR to the original repo (https://github.com/cocodataset/cocoapi/pull/50)
# but it doesn't seem to be active anymore.
RUN pip3 install --no-cache-dir git+https://github.com/waleedka/coco.git#subdirectory=PythonAPI

# 
# Install Protocol
#
RUN apt-get install -y autoconf automake libtool curl python-dev && cd /home/ && \
    curl -OL https://github.com/google/protobuf/releases/download/v3.2.0/protoc-3.2.0-linux-x86_64.zip && \
    unzip protoc-3.2.0-linux-x86_64.zip -d protoc3 && \
    mv protoc3/bin/* /usr/local/bin/ && \
    mv protoc3/include/* /usr/local/include/

#
# Install Seaborn
#
RUN pip3 install --no-cache-dir --upgrade seaborn

#
# Install Plotly and Cufflinks
#
RUN pip3 install --no-cache-dir --upgrade plotly cufflinks

#
# Set up environment
#
ENV PYTHONPATH=/root/workspace/models:/root/workspace/models/research:/root/workspace/models/research/slim:$PYTHONPATH

WORKDIR "/root"
CMD ["/bin/bash"]

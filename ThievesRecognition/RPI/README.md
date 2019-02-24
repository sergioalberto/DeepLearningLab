# Set up TensorFlow Object Detection API on the Raspberry Pi

## Update the Raspberry Pi
```
sudo apt-get update
sudo apt-get dist-upgrade
sudo apt-get install python-picamera python3-picamera
```

## Install TensorFlow
```
mkdir tf
cd tf
wget https://github.com/lhelontra/tensorflow-on-arm/releases/download/v1.12.0/tensorflow-1.12.0-cp35-none-linux_armv7l.whl
sudo pip3 install tensorflow-1.12.0-cp35-none-linux_armv7l.whl
sudo apt-get install libatlas-base-dev
sudo pip3 install pillow lxml jupyter matplotlib cython
sudo apt-get install python-tk
```

## Install OpenCV
```
sudo apt-get install libjpeg-dev libtiff5-dev libjasper-dev libpng12-dev
sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
sudo apt-get install libxvidcore-dev libx264-dev
sudo apt-get install qt4-dev-tools
pip3 install opencv-python
```

## Compile and Install Protobuf
```
sudo apt-get install autoconf automake libtool curl
wget https://github.com/google/protobuf/releases/download/v3.5.1/protobuf-all-3.5.1.tar.gz
tar -zxvf protobuf-all-3.5.1.tar.gz
cd protobuf-3.5.1
./configure
make
make check 
sudo make install
cd python
export LD_LIBRARY_PATH=../src/.libs
python3 setup.py build --cpp_implementation 
python3 setup.py test --cpp_implementation
sudo python3 setup.py install --cpp_implementation
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=cpp
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION=3
sudo ldconfig
protoc
sudo reboot now
```

## Thieves Detect
```
python3 thieves_detection.py
```

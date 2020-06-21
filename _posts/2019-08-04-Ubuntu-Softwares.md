---
layout: post
title:  "Softwares"
date:   2019-08-04 19:52:00 +0800
description: Softwares
categories: Softwares
tags: [Softwares]
location: Beijing,China
img: software-unsplash.jpg
---

# Ubuntu18.04 Softwares

## Develop

### nvidia-drives

```bash
sudo apt --purge autoremove nvidia*

sudo add-apt-repository ppa:graphics-drivers/ppa

sudo apt update

sudo apt upgrade

ubuntu-drivers list

sudo apt install nvidia-driver-VERSION_NUMBER_HERE

sudo reboot
```

### CUDA && CUDNN

```bash
# wget cuda_***_linux.run
# for exmaple cuda_10.1

sudo sh cuda_10.1.168_418.67_linux.run

export PATH=/usr/local/cuda-10.1/bin:/usr/local/cuda-10.1/NsightCompute-2019.3${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64\
                         ${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

sudo ldconfig
```

```bash
# download tgz file.
tar -xzvf cudnn-10.1-linux-x64-v7.6.1.34.tgz
sudo cp -a cuda/include/cudnn.h /usr/local/cuda-10.1/include/
sudo cp -a cuda/lib64/libcudnn* /usr/local/cuda-10.1/lib64/

# download deb file.
sudo dpkg -i libcudnn7_7.6.1.34-1+cuda10.1_amd64.deb
sudo dpkg -i libcudnn7-dev_7.6.1.34-1+cuda10.1_amd64.deb
sudo dpkg -i libcudnn7-doc_7.6.1.34-1+cuda10.1_amd64.deb
sudo ldconfig
# Verifying cudnn for deb install
cp -r /usr/src/cudnn_samples_v7/ $HOME
cd  $HOME/cudnn_samples_v7/mnistCUDNN
make clean && make
./mnistCUDNN
```

### cmake

```bash
# apt
sudo apt-get install cmake

# build from source
git clone https://github.com/Kitware/CMake.git
cd CMake
./bootstrap && make && sudo make install
```

### gflags

```bash
git clone git@github.com:gflags/gflags.git

cd gflags

mkdir build && cd build

cmake .. -DBUILD_SHARED_LIBS=ON

make

sudo make install
```

### glog

```bash
git clone git@github.com:google/glog.git

cd glog

./autogen.sh

./configure

make -j4

sudo make install
```

### googletest

```shell
git clone git@github.com:google/googletest.git

cd googletest/googletest/

git checkout release-1.8.1

mkdir build && cd build

cmake -DBUILD_SHARED_LIBS=ON -Dgtest_build_samples=ON ..

make -j4

sudo cp -a ../include/gtest/ /usr/include/

sudo cp -a libgtest_main.so libgtest.so /usr/lib/
```

### Anaconda

```bash
# download anaconda3 install shell script from offical website
wget https://repo.anaconda.com/archive/Anaconda3-2019.10-Linux-x86_64.sh
bash Anaconda3-2019.10-Linux-x86_64.sh

# To activate conda’s base environment in your shell session
eval "$(~/anaconda3/bin/conda shell.bash hook)"
# To install conda’s shell functions for easier access, first activate, then:
conda init
# If you’d prefer that conda’s base environment not be activated on startup, set the auto_activate_base parameter to false:
conda config --set auto_activate_base false
```

### ShadowSocks

```bash
```

### Proxychains4

```bash
```

### vscode

```bash
sudo snap install --classic code
```

### vim

```bash
sudo apt install vim

# ctags
sudo apt-get install ctags
```

### docker

```bash
sudo apt-get update

sudo apt-get install \
    apt-transport-https \
    ca-certificates \
    curl \
    gnupg-agent \
    software-properties-common

curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -

sudo apt-key fingerprint 0EBFCD88

sudo add-apt-repository \
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu \
   $(lsb_release -cs) \
   stable"

sudo apt-get update

sudo apt-get install docker-ce docker-ce-cli containerd.io

sudo docker run hello-world

sudo usermod -aG docker $USER # 重启后生效

newgrp - docker # 切换一下用户组（刷新缓存）
```

### nvidia-docker

```bash
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker

docker run --gpus all nvidia/cuda:10.0-base nvidia-smi
```

### boost

```bash
sudo apt-get install libboost-all-dev
```

### ProtoBuf

```bash
sudo apt-get install libprotobuf-dev protobuf-compiler
```

### Pybind11

```bash
git clone https://github.com/pybind/pybind11.git
cd pybind11
mkdir build
cd build
cmake ..
make check -j4
sudo make install
```

### doxygen

```bash
sudo apt-get install doxygen

# Tutorial
# https://cedar-renjun.github.io/2014/03/21/learn-doxygen-in-10-minutes/
```

### cpplint

```bash
pip install cpplint
```

### clang-flormat

```bash
sudo apt-get install clang-format

# in project, run the command
clang-format -style=google -dump-config > .clang-format

# modify DerivePointerAlignment false to enable PointerAlignment
```

### yapf

```bash
sudo apy-get install yapf

# in project, run the command
yapf --stype-help > .style.yapf
# modify indent_width=2

# please refer to help for more infomation
yapf -h
```

### pre-commit

```bash
pip install pre-commit

# in project dir
pre-commit install

vim .pre-commit-config.yaml

# details refer https://pre-commit.com/
```

### graphviz

```bash
sudo apt-get install graphviz

# for python
pip install graphviz
```

### opencv

```bash
# apt
sudo apt install libopencv-dev
```

```bash
# build from source
sudo apt install build-essential cmake git pkg-config libgtk-3-dev
sudo apt install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libxvidcore-dev libx264-dev
sudo apt install libjpeg-dev libpng-dev libtiff-dev gfortran openexr libatlas-base-dev
sudo apt install python3-dev python3-numpy libtbb2 libtbb-dev libdc1394-22-dev


git clone https://github.com/opencv/opencv_contrib.git
cd opencv_contrib
git checkout 3.4.10
cd ..
git clone https://github.com/opencv/opencv.git
cd opencv
git checkout 3.4.10

mkdir build_3.4.10 && cd build_3.4.10
cmake -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/usr/local \
    -DINSTALL_C_EXAMPLES=ON \
    -DINSTALL_PYTHON_EXAMPLES=ON \
    -DOPENCV_GENERATE_PKGCONFIG=ON \
    -DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
    -DBUILD_EXAMPLES=ON .. \
    -DPYTHON3_EXECUTABLE=$HOME/anaconda3/bin/python3 \
    -DPYTHON_INCLUDE_DIR=$HOME/anaconda3/include/python3.7m \
    -DPYTHON_LIBRARY=$HOME/anaconda3/lib/libpython3.7m.so \
    -DPYTHON3_NUMPY_INCLUDE_DIRS=$HOME/anaconda3/lib/python3.7/site-packages/numpy/core/include \
    -DWITH_CUDA=OFF \
    -DBUILD_TIFF=ON \
    -DOPENCV_ENABLE_NONFREE=ON ..
make -j
sudo make install

# sudo vim /etc/ld.so.conf.d/opencv.conf
# add /usr/local/lib to opencv.conf
sudo ldconfig
pkg-config --modversion opencv

# for python
# cp /usr/local/lib/python3.5/site-packages/cv2/python-3.5/cv2.cpython-35m-x86_64-linux-gnu.so $HOME/anaconda3/lib/python3.7/site-packages/
```

### BLAS

#### OpenBLAS

```bash
# apt
sudo apt-get install libopenblas-dev

# build from source
git clone https://github.com/xianyi/OpenBLAS.git
cd OpenBLAS
make -j
sudo make PREFIX=/usr/local/OpenBLAS install
```

#### ATLAS BLAS

```bash
sudo apt-get install libatlas-base-dev liblapack-dev libblas-dev
```

#### Intel MKL && MKLDNN

```bash
# mkl
# download mkl.tgz and untar
cd l_mkl_2019.5.281
sudo ./install.sh
# /etc/ld.so.conf.d/mkl.conf
# /opt/intel/lib/intel64
# /opt/intel/mkl/lib/intel64
# sudo ldconfig -v

# mkldnn

```

#### cuBLAS

```bash
# cublas will be installed when cuda is installed.
```

### Caffe

```bash
sudo apt-get install -y build-essential cmake git pkg-config
sudo apt-get install -y libprotobuf-dev libleveldb-dev libsnappy-dev libhdf5-serial-dev protobuf-compiler libatlas-base-dev libboost-all-dev libgflags-dev libgoogle-glog-dev liblmdb-dev
sudo apt-get install -y python3-dev
sudo apt-get install -y python3-numpy python3-scipy
# build and install opencv
git clone https://github.com/BVLC/caffe
cd caffe
# prepare Makefile.config https://github.com/jiweibo/Scripts/blob/master/caffe/makefile.config
make all -j4
make test
make runtest
cd python
for req in $(cat requirements.txt); do pip install $req; done
cd ..
make pycaffe -j4

vim ~/.bashrc
export PYTHONPATH=$HOME/repository/caffe/python:$PYTHONPATH
```

### eigen

```bash
sudo apt-get install libeigen3-dev
```

### gperftools

```bash
# 依赖libunwind
git clone git://git.sv.gnu.org/libunwind.git
cd libunwind
./autogen.sh
./configure
make
make install

cd $REPOS

git clone https://github.com/gperftools/gperftools.git
cd gperftools
./autogen.sh
./configure
make
make install
```

## Life

### variety

```bash
sudo apt update && sudo apt install variety
```

### VNote

```bash
# https://github.com/tamlok/vnote
```

### draw.io

```bash
# https://github.com/jgraph/drawio
```

### Netron

```bash
# https://github.com/lutzroeder/netron

Netron-3.3.2.AppImage
```

### wechat

```bash
sudo snap install electronic-wechat
```

### github pages

```shell
sudo apt-get install ruby-full build-essential zlib1g-dev

echo '# Install Ruby Gems to ~/gems' >> ~/.bashrc
echo 'export GEM_HOME="$HOME/gems"' >> ~/.bashrc
echo 'export PATH="$HOME/gems/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

gem install jekyll bundler
```

### SysMonitor

```bash
sudo add-apt-repository ppa:fossfreedom/indicator-sysmonitor
sudo apt-get update
sudo apt-get install indicator-sysmonitor
```

## Settings

### PS1 (add git branch)

```bash
# ps1
# 添加git branch信息
# 在 .bashrc中添加
function parse_git_branch {
  git branch --no-color 2> /dev/null | sed -e '/^[^*]/d' -e 's/* \(.*\)/(\1)/'
}

# 在bashrc中修改PS1
export PS1='\[\e]0;\u@\h: \w\a\]${debian_chroot:+($debian_chroot)}\[\033[01;32m\]\u@\h\[\033[00m\]:\[\033[01;34m\]\W\[\033[00m\]\[\033[01;36m\]$(parse_git_branch)\[\033[00m\]\$ '
# or
export PS1='\[\033[01;32m\]\u@\h\[\033[00m\]:\[\033[01;34m\]\W\[\033[00m\]\[\033[01;36m\]$(parse_git_branch)\[\033[00m\]\$ '
```

## Mirrors

### ubuntu清华镜像

#### 16.04LTS

```bash
# 默认注释了源码镜像以提高 apt update 速度，如有需要可自行取消注释
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ xenial main restricted universe multiverse
# deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ xenial main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ xenial-updates main restricted universe multiverse
# deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ xenial-updates main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ xenial-backports main restricted universe multiverse
# deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ xenial-backports main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ xenial-security main restricted universe multiverse
# deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ xenial-security main restricted universe multiverse

# 预发布软件源，不建议启用
# deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ xenial-proposed main restricted universe multiverse
# deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ xenial-proposed main restricted universe multiverse
```

#### 18.04LTS

```bash
# 默认注释了源码镜像以提高 apt update 速度，如有需要可自行取消注释
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic main restricted universe multiverse
# deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic-updates main restricted universe multiverse
# deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic-updates main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic-backports main restricted universe multiverse
# deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic-backports main restricted universe multiverse
deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic-security main restricted universe multiverse
# deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic-security main restricted universe multiverse

# 预发布软件源，不建议启用
# deb https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic-proposed main restricted universe multiverse
# deb-src https://mirrors.tuna.tsinghua.edu.cn/ubuntu/ bionic-proposed main restricted universe multiverse
```

### pip清华镜像

```bash
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple some-package

# or
pip install pip -U
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple
```

---

# CentOS7 Softwares

## 基本组件

```
sudo yum update

yum groupinstall "Development Tools"

# gcc485
yum -y install gcc gcc-c++

# make
yum -y install make

# cmake
yum -y install cmake

# wget
yum -y install wget

# vim

# git

# python3 && pip
```

---

# Windows Softwares


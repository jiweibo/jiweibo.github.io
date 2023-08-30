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

### go
```
# download go
wget https://golang.org/dl/go1.15.8.linux-amd64.tar.gz
tar -C /usr/local -xzf go1.15.8.linux-amd64.tar.gz
export PATH=$PATH:/usr/local/go/bin

# check version
go version
```

### cmake

```bash
# apt
sudo apt-get install cmake

# build from source
sudo apt install libssl-dev
git clone https://github.com/Kitware/CMake.git
cd CMake
./bootstrap && make && sudo make install
```

### bazel

```bash
sudo apt install g++ unzip zip
wget --no-check-certificate https://github.com/bazelbuild/bazel/releases/download/3.7.0/bazel-3.7.0-installer-linux-x86_64.sh
chmod +x bazel-3.7.0-installer-linux-x86_64.sh
./bazel-3.7.0-installer-linux-x86_64.sh --user
export PATH="$PATH:$HOME/bin"
```

### openssl
```bash
git clone https://github.com/openssl/openssl.git
cd openssl
./Configure --prefix=/usr/local/ssl --openssldir=/usr/local/ssl '-Wl,-rpath,$(LIBRPATH)'
make -j
make install
```

### curl
```bash
git clone https://github.com/curl/curl.git
cd curl
./buildconf
./configure --prefix=/usr/local --with-openssl=/usr/local/ssl/
make -j
make install
```


### gflags

```bash
#git clone git@github.com:gflags/gflags.git
git clone https://github.com/gflags/gflags.git
cd gflags
mkdir build && cd build
cmake .. -DBUILD_SHARED_LIBS=ON
make
sudo make install
```

### glog

```bash
#git clone git@github.com:google/glog.git
git clone https://github.com/google/glog.git
cd glog
mkdir build && cd build
cmake .. -DBUILD_SHARED_LIBS=ON
make -j4
sudo make install

# apt
sudo apt-get install -y libgoogle-glog-dev
```

### googletest

```shell
#git clone git@github.com:google/googletest.git
git clone https://github.com/google/googletest.git
cd googletest
git checkout release-1.10.0
mkdir build && cd build
cmake -DBUILD_SHARED_LIBS=ON -Dgtest_build_samples=ON ..
make -j4
sudo make install
```

### Anaconda

```bash
# download anaconda3 install shell script from offical website
wget https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh
bash Anaconda3-2020.11-Linux-x86_64.sh

# To activate conda’s base environment in your shell session
eval "$(~/anaconda3/bin/conda shell.bash hook)"
# To install conda’s shell functions for easier access, first activate, then:
conda init
# If you’d prefer that conda’s base environment not be activated on startup, set the auto_activate_base parameter to false:
conda config --set auto_activate_base false
```

### Python

```bash
sudo apt-get install libffi-dev libssl-dev libsqlite3-dev libbz2-dev libncursesw5-dev liblzma-dev tk-dev uuid-dev libreadline-dev libgdbm-dev

# download python3
wget https://www.python.org/ftp/python/3.7.9/Python-3.7.9.tgz
tar xzf Python-3.7.9.tgz && cd Python-3.7.9
./configure --prefix=$HOME/python37 --enable-shared
make -j4
make install
export PATH=$HOME/python37/bin:$PATH
export LD_LIBRARY_PATH=$HOME/python37/lib:$LD_LIBRARY_PATH
```

### LLVM

```bash
sudo apt-get install clang lld
git clone https://github.com/llvm/llvm-project.git
cd llvm-project
git checkout llvmorg-14.0.0-rc1
mkdir build && cd build
# cmake -G 'Unix Makefiles' -DLLVM_ENABLE_PROJECTS=all -DLLVM_TARGETS_TO_BUILD=all ../llvm
# cmake -G 'Unix Makefiles' -DLLVM_ENABLE_PROJECTS="clang;clang-tools-extra;compiler-rt;debuginfo-tests;libclc;libcxx;libcxxabi;libunwind;lld;lldb;mlir;openmp;parallel-libs;polly;pstl" -DLLVM_TARGETS_TO_BUILD=all ../llvm/
# make
# sudo make install
cmake -G Ninja ../llvm -DLLVM_ENABLE_PROJECTS="clang;clang-tools-extra;mlir" -DLLVM_BUILD_EXAMPLES=ON -DLLVM_TARGETS_TO_BUILD="X86;NVPTX;AMDGPU" -DCMAKE_BUILD_TYPE=Release -DLLVM_ENABLE_ASSERTIONS=ON -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++ -DLLVM_ENABLE_LLD=ON
ninja
```

### ShadowSocks

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

# build from source
https://jiweibo.github.io/VIM/
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

```bash
wget https://dl.bintray.com/boostorg/release/1.74.0/source/boost_1_74_0.tar.gz
tar -xzf boost_1_74_0.tar.gz
cd boost_1_74_0
./bootstrap.sh --with-python=$HOME/anaconda3/bin/python
./b2
sudo ./b2 install
```

### ProtoBuf

```bash
sudo apt-get install libprotobuf-dev protobuf-compiler
```

```bash
sudo apt install autoconf automake libtool
git clone https://github.com/protocolbuffers/protobuf.git
cd protobuf
git submodule update --init --recursive
git checkout v3.14.0
./autogen.sh
./configure
make
make check
sudo make install
sudo ldconfig

# for python protobuf
cd python
python setup.py build
python setup.py test
python setup.py install
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
git checkout 4.5.0
cd ..
git clone https://github.com/opencv/opencv.git
cd opencv
git checkout 4.5.0

mkdir build_4.5.0 && cd build_4.5.0
ANACONDA_HOME=$HOME/anaconda3
cmake -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/usr/local \
      -DINSTALL_C_EXAMPLES=OFF \
      -DINSTALL_PYTHON_EXAMPLES=OFF \
      -DOPENCV_GENERATE_PKGCONFIG=ON \
      -DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
      -DBUILD_EXAMPLES=OFF \
      -DWITH_CUDA=OFF \
      -DBUILD_TIFF=ON \
      -DOPENCV_FORCE_3RDPARTY_BUILD=ON \
      -DOPENCV_ENABLE_NONFREE=ON \
      -DPYTHON_DEFAULT_EXECUTABLE=$(which python) \
      -DPYTHON_EXECUTABLE=$ANACONDA_HOME/bin/python3 \
      -DPYTHON_INCLUDE_DIR=$ANACONDA_HOME/include/python3.8 \
      -DPYTHON_LIBRARY=$ANACONDA_HOME/lib/libpython3.8.so \
      -DPYTHON_NUMPY_INCLUDE_DIRS=$ANACONDA_HOME/lib/python3.8/site-packages/numpy/core/include \
      ..
make -j
sudo make install

# sudo vim /etc/ld.so.conf.d/opencv.conf
# add /usr/local/lib to opencv.conf
sudo ldconfig
pkg-config --modversion opencv4

# for python
# cp /usr/local/lib/python3.8/site-packages/cv2/python-3.8/cv2.cpython-38-x86_64-linux-gnu.so $ANACONDA_HOME/lib/python3.8/site-packages/
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

#### CBLAS

```bash
wget http://www.netlib.org/blas/blas-3.8.0.tgz
wget http://www.netlib.org/blas/blast-forum/cblas.tgz
tar xzf blas-3.8.0.tgz
tar xzf cblas.tgz

pushd BLAS-3.8.0
make
popd
pushd CBLAS
# vim Makefile.in
# Modify BLLIB to the blas_LINUX.a obtained by compiling BLAS before
make
# If you wang to get the dynamic library, you need to make the following modifications:
# ARCH = gcc
# ARCHFLAGS = -shared -o
popd
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
# compile and install
# cmake
# boost
# gflags
# glog
# protobuf
# openblas
# install python numpy and scipy

sudo apt-get install -y build-essential git pkg-config
sudo apt-get install -y libleveldb-dev libsnappy-dev libhdf5-serial-dev protobuf-compiler liblmdb-dev

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
git clone https://github.com/libunwind/libunwind.git
cd libunwind
./autogen.sh
./configure
make
sudo make install

cd ..

git clone https://github.com/gperftools/gperftools.git
cd gperftools
./autogen.sh
./configure
make
make check
sudo make install
```

### gRPC

```bash
sudo apt install build-essential autoconf libtool pkg-config
git clone https://github.com/grpc/grpc.git
cd gtpc
git checkout v1.34.0
git submodule update --init --recursive
mkdir build && cd build
cmake -DgRPC_INSTALL=ON \
      -DgRPC_BUILD_TESTS=OFF \
      -DgRPC_PROTOBUF_PROVIDER=package \
      ..
make -j
sudo make install
```

### LTTng

```
# https://lttng.org/docs/v2.13

sudo apt install pkg-config liburcu‑dev libpopt‑dev libxml2‑dev babeltrace2 liblttng‑ust‑dev
wget https://lttng.org/files/lttng-tools/lttng-tools-latest-2.13.tar.bz2 &&
tar -xf lttng-tools-latest-2.13.tar.bz2 &&
cd lttng-tools-2.13.* &&
./configure &&
make -j8 &&
sudo make install &&
sudo ldconfig

sudo apt-add-repository ppa:lttng/ppa
sudo apt-get update
sudo apt-get install babeltrace2
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

### typora

```bash
# or run:
# sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys BA300B7755AFCFAE
wget -qO - https://typora.io/linux/public-key.asc | sudo apt-key add -
# add Typora's repository
sudo add-apt-repository 'deb https://typora.io/linux ./'
sudo apt-get update
# install typora
sudo apt-get install typora
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


### cc library

#### jsoncpp

```bash
sudo apt-get install libjsoncpp-dev

# demo:https://github.com/jiweibo/ProjectTest/tree/master/jsoncpp_test
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


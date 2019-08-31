---
layout: post
title:  "Softwares"
date:   2019-08-04 19:52:00 +0800
description: Softwares
categories: Softwares
tags: [Softwares]
location: Beijing,China
img: 
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
# download deb or tgz cudnn file.

sudo dpkg -i libcudnn7_7.6.1.34-1+cuda10.1_amd64.deb

sudo dpkg -i libcudnn7-dev_7.6.1.34-1+cuda10.1_amd64.deb

sudo dpkg -i libcudnn7-doc_7.6.1.34-1+cuda10.1_amd64.deb

sudo ldconfig

# Verifying cudnn

cp -r /usr/src/cudnn_samples_v7/ $HOME

cd  $HOME/cudnn_samples_v7/mnistCUDNN

make clean && make

./mnistCUDNN
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

sudo usermod -aG docker your-user
```

### nvidia-docker

```bash
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | \
  sudo apt-key add -

distribution=$(. /etc/os-release;echo $ID$VERSION_ID)

curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
 sudo tee /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get update

sudo apt-get install nvidia-docker2

sudo pkill -SIGHUP dockerd

docker run --runtime=nvidia --rm nvidia/cuda nvidia-smi
```

### boost

```bash
sudo apt-get install libboost-all-dev
```

### ProtoBuf

```bash
sudo apt-get install libprotobuf-dev protobuf-compiler
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

# in the project run the command
clang-format -style=google -dump-config > .clang-format

# modify DerivePointerAlignment false to enable PointerAlignment
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
```

### BLAS

#### OpenBLAS

```bash
```

#### ATLAS BLAS

```bash
sudo apt-get install libatlas-base-dev liblapack-dev libblas-dev
```

#### Intel MKL && MKLDNN

```bash
```

#### cuBLAS

```bash
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

---

# Windows Softwares


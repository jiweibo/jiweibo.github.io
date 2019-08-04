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


### pre-commit

### cpplint

### clang-flormat

## Life

### variety

```bash
sudo apt update && sudo apt install variety
```

---

# Windows Softwares


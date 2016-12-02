# Using SugarTensor through Docker

SugarTensor supports `Docker` to make it easy to running through [Docker](http://www.docker.com/).

## Installing and using Docker

For general information about Docker, see [the Docker site](https://docs.docker.com/installation/).
If you use GPUs, you'll need [Nvidia-docker](https://github.com/NVIDIA/nvidia-docker) as well

## Download SugarTensor docker images

Get sugartensor docker image for GPU support

```
docker pull sugartensor/sugartensor:latest
```

Get sugartensor docker image for CPU only support

```
docker pull sugartensor/sugartensor:latest-cpu
```

## Running sugartensor container

Run sugartensor container with GPU support

```
nvidia-docker run -it sugartensor/sugartensor:latest /bin/bash 
```

Run sugartensor container with CPU only support

```
docker run -it sugartensor/sugartensor:latest-cpu /bin/bash 
```

## Testing sugartensor container

Run MNIST example.

```
python mnist_conv.py
```

# Author

Namju Kim (buriburisuri@gmail.com) at Jamonglabs Co., Ltd.





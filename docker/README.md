# Using SugarTensor through Docker

SugarTensor supports `Docker` to make it easy to running through [Docker](http://www.docker.com/).

## Installing and using Docker

For general information about Docker, see [the Docker site](https://docs.docker.com/installation/).
If you use GPUs, you'll need [Nvidia-docker](https://github.com/NVIDIA/nvidia-docker) as well.

## Download SugarTensor docker images

Get sugartensor docker image for GPU support

```
docker pull sugartensor/sugartensor
```

Get sugartensor docker image for CPU only support

```
docker pull sugartensor/sugartensor:latest-cpu
```

## Running sugartensor container for shell console

Run sugartensor container with GPU support

```
nvidia-docker run -it sugartensor/sugartensor 
```

Run sugartensor container with CPU only support

```
docker run -it sugartensor/sugartensor:latest-cpu 
```

## Setting docker to enable SSH connection to Docker container

In Ubuntu 14.04, add following line to the end of '/etc/default/docker' file.

```
DOCKER_OPTS="-H tcp://0.0.0.0:2376 -H unix:///var/run/docker.sock"
```

and restart docker service.

```
sudo service docker restart
```

## Running sugartensor container for SSH server mode

Run sugartensor container with GPU support

```
nvidia-docker run -p 2222:22 -d sugartensor/sugartensor
```

Run sugartensor container with CPU only support

```
docker run -p 2222:22 -d sugartensor/sugartensor:latest-cpu 
```

## Connect to the sugartensor SSH server container using ssh

Connect using the following command. ( pwd is 'ubuntu' )
Now, you can use your favorite IDE's remote debugging feature if your IDE supports remote debugging. 
( To my knowledge, PyCharm professional and VisualStudio support remote python debugging )

```
ssh root@your_docker_host -p 2222
```

## Testing sugartensor container

Run MNIST example in the '/root/sugartensor/example' directory.

```
python mnist_conv.py
```

# Author

Namju Kim (namju.kim@kakaocorp.com) at KakaoBrain Corp.



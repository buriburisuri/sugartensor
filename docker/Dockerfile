FROM tensorflow/tensorflow:0.12.0-gpu
MAINTAINER Namju Kim buriburisuri@gmail.com

# requirements
RUN pip install --upgrade pip
RUN pip install sugartensor==0.0.2.4

# make example directory
RUN mkdir -p /sugartensor/example
ADD sugartensor/example/* /sugartensor/example/

# set default directory to sugar tensor's example directory
WORKDIR /sugartensor/example



FROM tensorflow/tensorflow:0.12.0-rc1-gpu
MAINTAINER Namju Kim buriburisuri@gmail.com

# requirements
RUN pip install --upgrade pip
RUN pip install sugartensor==0.0.2.3
RUN pip3 install --upgrade pip
RUN pip3 install sugartensor==0.0.2.3

# make example directory
RUN mkdir -p /sugartensor/example
ADD sugartensor/example/* /sugartensor/example/

# set default directory to sugar tensor's example directory
WORKDIR /sugartensor/example



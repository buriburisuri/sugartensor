FROM tensorflow/tensorflow:0.12.0-rc0-gpu
MAINTAINER Namju Kim buriburisuri@gmail.com

# requirements
RUN pip install --upgrade pip
RUN pip install tqdm==4.8.4
RUN pip install sugartensor==0.0.2.1

# make example directory
RUN mkdir /sugartensor
RUN copy -R ../sugartensor/example /sugartensor

# set default directory to sugar tensor's example directory
WORKDIR /sugartensor/example

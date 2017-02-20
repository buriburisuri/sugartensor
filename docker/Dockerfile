FROM tensorflow/tensorflow:1.0.0-gpu
MAINTAINER Namju Kim buriburisuri@gmail.com

# requirements
RUN pip install --upgrade pip
RUN pip install sugartensor==0.0.2.5

# SSH support
RUN apt-get update && apt-get install -y openssh-server
RUN mkdir /var/run/sshd
RUN echo 'root:ubuntu' | chpasswd
RUN sed -ie 's/PermitRootLogin without-password/PermitRootLogin yes/g' /etc/ssh/sshd_config

# make example directory
RUN mkdir -p /sugartensor/example
ADD sugartensor/example/* /sugartensor/example/

# set default directory to sugar tensor's example directory
WORKDIR /sugartensor/example

# port export
EXPOSE 22 8888



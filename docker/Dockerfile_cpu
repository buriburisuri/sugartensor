FROM tensorflow/tensorflow:1.0.0
MAINTAINER Namju Kim namju.kim@kakaobrain.com

# requirements
RUN pip install --upgrade pip
RUN pip install sugartensor==1.0.0.2

# SSH support
RUN apt-get update && apt-get install -y openssh-server
RUN mkdir /var/run/sshd
RUN echo 'root:ubuntu' | chpasswd
RUN sed -ie 's/PermitRootLogin without-password/PermitRootLogin yes/g' /etc/ssh/sshd_config

# make example directory
RUN mkdir -p /root/sugartensor/example
ADD sugartensor/example/* /root/sugartensor/example/

# expose SSH port
EXPOSE 22

# set default directory to sugar tensor's example directory
WORKDIR /root/sugartensor/example

# default entry point
ENTRYPOINT service ssh restart && bash

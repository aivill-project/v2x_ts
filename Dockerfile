FROM ubuntu:18.04

WORKDIR /home/

RUN git clone https://github.com/a22106/v2x_ts.git 50-1_v2x

# COPY . ./

RUN apt-get update
RUN apt-get install wget git -y

# - Geographic area:  설정하기
RUN sudo apt -y install software-properties-common dirmngr apt-transport-https lsb-release ca-certificates
RUN add-apt-repository ppa:graphics-drivers/ppa
RUN apt-get install nvidia-driver-470
RUN apt-get install nvidia-cuda-toolkit

RUN apt-get install python3-pip

# pip 패키지 설치
RUN pip install --upgrade pip
RUN pip install tsai
RUN pip install numpy==1.23 IPython ipykernel



WORKDIR /home/50-1_v2x
COPY ./data/X_sum_all.npy ./data/X_sum_all.npy
COPY ./data/y_sum_all.npy ./data/y_sum_all.npy
COPY ./models/turn_20221226_0955 ./models/turn_20221226_0955
COPY ./models/speed_20221226_1202 ./models/speed_20221226_1202
COPY ./models/hazard_20221226_1809 ./models/hazard_20221226_1809
FROM ubuntu:22.04
ARG DEBIAN_FRONTEND=noninteractive

RUN apt-get -qq update -y \
  && apt-get -qq install -y python3-pip curl wget \
  && python3 -m pip install --upgrade pip

RUN apt-get -qq install -y python3-numpy python3-opencv

RUN curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | tee /etc/apt/sources.list.d/nvidia-container-toolkit.list \
  && apt-get update \
  && apt-get install -y nvidia-container-toolkit

RUN pip3 install \
  pytest coverage pytest-mock

WORKDIR /workdir
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

CMD bash
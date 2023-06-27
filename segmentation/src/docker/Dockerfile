FROM python:3

ENV WORK_PATH /workspace

RUN apt-get update && \
    apt-get install -y \
    sudo curl git build-essential tmux python3-pip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    rm -rf /var/cache/apk/

ENV PIP_OPTIONS "--no-cache-dir --progress-bar off"
RUN pip3 install -U pip && \
    pip3 install poetry==1.5.0 urllib3==1.26.9 && \
    poetry config virtualenvs.create false

# Install OpenCV and Mediapipe
RUN python3 get-pip.py && \
    pip install opencv-python && \
    pip install mediapipe

# ユーザーの追加
ARG USER_NAME=duser
ARG GROUP_NAME=build
ARG USER_ID=1000
ARG GROUP_ID=1000
RUN groupadd --gid $GROUP_ID -o $GROUP_NAME && \
    adduser --disabled-password --gecos '' --uid $USER_ID --gid $GROUP_ID $USER_NAME && \
    adduser $USER_NAME sudo && \
    echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER $USER_NAME

WORKDIR /workspace
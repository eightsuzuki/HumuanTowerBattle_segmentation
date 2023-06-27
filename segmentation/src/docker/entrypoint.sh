#!/bin/bash
USER_NAME=duser
# sync uid & gid
sudo groupadd --gid $GROUP_ID -o $USER_NAME
if [ `id -u` -ne $USER_ID ]; then
    if [ `id -g` -ne $GROUP_ID ]; then
        sudo usermod -u $USER_ID -g $GROUP_ID -o $USER_NAME
    else
        sudo usermod -u $USER_ID -o $USER_NAME
    fi
else
    if [ `id -g` -ne $GROUP_ID ]; then
        sudo usermod -g $GROUP_ID $USER_NAME
    fi
fi
$@
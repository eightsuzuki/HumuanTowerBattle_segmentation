.DEFAULT_GOAL := help

# const
IMAGE_NAME=tower-battle
CONTAINER_NAME=tower-battle
JUPYTER_PORT=8890

CURRENT_UID := $(shell id -u)
CURRENT_GID := $(shell id -g)
export CURRENT_UID
export CURRENT_GID

## build docker image
.PHONY: build
build:
	docker build -t ${IMAGE_NAME} --force-rm=true --no-cache -f docker/Dockerfile ./docker

## run docker container
.PHONY: up
up:
	docker run --gpus all --rm --name ${CONTAINER_NAME} --shm-size=8g --env USER_ID=${CURRENT_UID} --env GROUP_ID=${CURRENT_GID} -v ${PWD}/mmsegmentation:/mmsegmentation -w /mmsegmentation -dit ${IMAGE_NAME}

## run bash in container
.PHONY: exec
exec:
	docker exec -w /workspace -it ${CONTAINER_NAME} bash

## down the container
.PHONY: down
down:
	docker stop ${CONTAINER_NAME}

## reboot the container
.PHONY: reboot
reboot: down up

## help
.PHONY: help
help:
	@printf "\nusage : make <commands> \n\nthe following commands are available : \n\n"
	@cat $(MAKEFILE_LIST) | awk '1;/help:/{exit}' | awk '!/^.PHONY:.*/' | awk '/##/ { print; getline; print; }' | awk '{ getline x; print x; }1' | awk '{ key=$$0; getline; printf "\033[36m%-30s\033[0m %s\n", key, $$0;}' | sed -e 's/##//' | sed -e 's/://'
	@printf "\n"
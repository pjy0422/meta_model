#! /bin/bash
echo "configuring container compositions..."

IMAGE_NAME=pjy0422_env:311
CONTAINER_USER=pjy0422
CONTAINER_PORT=3800
IMAGE_NAME_1=${IMAGE_NAME/:/''}
CONTAINER_NAME=${IMAGE_NAME_1/./''}_${CONTAINER_USER}
CONTAINER_MEMORY=100G

echo "building container..."

# Do not change these variables if unnecessary
CPU_DEVICE=16
CPU_ID_LIST="1-32"

# Change the last row to bind individual workspace to docker-container
docker run -it -d \
    -p ${CONTAINER_PORT}:${CONTAINER_PORT} \
    --user ${CONTAINER_USER} \
    --cpus=${CPU_DEVICE} \
    -m ${CONTAINER_MEMORY} \
    --memory-swap -1 \
    -v /hdd1/${CONTAINER_USER}:/home/workspace:rw \
    --workdir /home/workspace \
    --name ${CONTAINER_NAME} ${IMAGE_NAME} \
    /bin/bash

# Start container and run mlflow server in background
docker start $CONTAINER_NAME

# Give necessary permissions to the user
docker exec -it --user root $CONTAINER_NAME bash -c "adduser $CONTAINER_USER sudo && exit"
docker exec -it --user root $CONTAINER_NAME bash -c "echo '$CONTAINER_USER ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers && exit"
docker exec -it --user root $CONTAINER_NAME bash -c "passwd $CONTAINER_USER"

# Start mlflow server in the background
docker exec -d --user root $CONTAINER_NAME bash -c "mlflow server --host 127.0.0.1 --port 3060"

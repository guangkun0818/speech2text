# speech2text

## Environment set up
Build training runtime as below.
```bash
# Build docker image
docker build -t training_env:version . -f Dockerfile.build
# Start your container with docker 
docker run -itd \
    --gpus=all \
    --ipc=host \
    --name=training-runtime \
    -v /mnt:/mnt \
    training_env:0.1 /bin/bash
```

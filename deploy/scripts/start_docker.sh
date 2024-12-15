#!/bin/bash
# Log everything to start_docker.log
exec > /home/ubuntu/start_docker.log 2>&1

echo "Logging in to ECR..."
aws ecr get-login-password --region ap-southeast-2 | docker login --username AWS --password-stdin 225989338339.dkr.ecr.ap-southeast-2.amazonaws.com

echo "Pulling Docker image..."
docker pull 225989338339.dkr.ecr.ap-southeast-2.amazonaws.com/yt-chrome-plugin:latest

echo "Checking for existing container..."
if [ "$(docker ps -q -f name=yt-chrome-app)" ]; then
    echo "Stopping existing container..."
    docker stop yt-chrome-app
fi

if [ "$(docker ps -aq -f name=yt-chrome-app)" ]; then
    echo "Removing existing container..."
    docker rm yt-chrome-app
fi

echo "Starting new container..."
docker run -d -p 80:5000 -e GROQ_API_KEY=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 --name yt-chrome-app  225989338339.dkr.ecr.ap-southeast-2.amazonaws.com/yt-chrome-plugin:latest

echo "Container started successfully."
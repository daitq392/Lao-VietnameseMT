#!/bin/bash

container_id=$(docker run -d --name tts66laovi -p 7080:7080 tts66laovi_image)
sleep 10
docker logs -f "$container_id" | grep "MODEL_LOG - Processed" &

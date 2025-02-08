#!/bin/bash

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <host:port> <input_file> <output_file>"
    exit 1
fi


host_port="$1"
input_file="$2"
output_file="$3"


docker cp "$input_file" tts66laovi:/home/model-server/input.txt


docker logs -f tts66laovi | grep "MODEL_LOG - Processed" &


docker exec -i tts66laovi curl -s -X POST "http://$host_port/predictions/tts66laovi" -T /home/model-server/input.txt -o /home/model-server/output.txt


docker exec -i tts66laovi python /home/model-server/process_output.py "/home/model-server/output.txt" "/home/model-server/output_processed.txt"

docker cp tts66laovi:/home/model-server/output_processed.txt "$output_file"

docker stop tts66laovi
docker rm tts66laovi

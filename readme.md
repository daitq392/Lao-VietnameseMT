# Lao-Vietnamese Translation System

Download the docker image here: https://drive.google.com/file/d/12R4WDf6zylqs50cqjP57EdiTy7ljAyJO/view?usp=sharing
## Load the Docker Image

Run the following command inside the project directory:

```bash
sudo docker load -i tts66laovi_image_latest.tar
```

## Run the Docker Container

```bash
sudo bash deploy.sh
```

## Translate a Test Set
```bash
./test.sh localhost:7080 <input_file> <output_file>
```
If the system is running correctly, logs will be generated after processing every 5 sentences.


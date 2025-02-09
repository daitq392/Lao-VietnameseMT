# Lao-Vietnamese Translation System

Download the docker image here: [link](https://drive.google.com/file/d/12R4WDf6zylqs50cqjP57EdiTy7ljAyJO/view?usp=sharing)
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

## Acknowledgement

Code adapted from [Harvard NLP's Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/).

The model was trained on the dataset provided in the VLSP 2023 challenge on Vietnamese-Lao machine translation. After preprocessing, approximately 34,000 sentences remained. We used 30,248 sentences for training, 2,018 for validation, and 2,018 for the test set.

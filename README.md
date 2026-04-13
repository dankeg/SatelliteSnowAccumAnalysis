# SatelliteSnowAccumAnalysis
DS4520 Project for Predicting Changes in Snow Accumulation from Satellite Imagery!

## Initial Project Structure
The core training and inference code currently lives in:
- Bayesian_Predictor.py
- ImageDataset.py
- cnn_segmentation.py
- cnn_training.py

A docker container is provided to abstract out environment setup.
To launch any file in docker, you can run:

`docker compose run --rm app python [file_path]`

Alternatively, the requirements file can be used for a local installation.

For the Vercel demo and local inference:

`pip install -r requirements.txt`

For training and dataset work:

`pip install -r requirements-training.txt`

## Fetching Data
Data fetching occurs from the Sentinel Satellite Data Catalog. Metadata such as pre-existing weather data, image city/location, date ranges, and image sizing and chunking settings are used to target images that are most likely to have snowfall, and retrieve them for training and testing the CNN model.

The API provided by Sentinel is very complex. The following preset fetches a reasonable dataset for training and using the CNN model, and downstream training and using the bayesian model:

``

## Running CNN
The CNN uses a UNet style architecture, that has been slimmed down in size.
Research shows that UNets are ideal for pixel classification / masking tasks.

To run the CNN, run the following from the root of the repo:
`python SatelliteSnowAccumAnalysis/Training/cnn_training.py`

It will train the model, run validation, export performance metrics over epochs, and save the model to disk:

```

python SatelliteSnowAccumAnalysis/Fetch/sentinel_data_fetch.py \                                              
  --cities boston new_york buffalo chicago \
  --years 2015 2016 2017 2018 2019 2020 2021 2022 2023 2024 \
  --start-month-day 11-15 \
  --end-month-day 02-28 \
  --max-items-per-window 5 \
  --max-cloud 25 \
  --patch-size 256 \
  --stride 256 \
  --min-labeled-ratio 0.50 \
  --min-clear-ratio 0.50 \
  --background-keep-prob 0.10 \
  --min-snow-ratio-positive 0.01 \
  --split-unit scene \
  --outdir snow_dataset_small

```

This preset focuses on U.S cities snowfall, during winter date ranges where snowfall is most likely to occur. The max cloud setting serves to prevent the image from being unhelpfully obfuscated by cloud cover. The ratios ensure that the images we select actually contain snowfall: with images with higher amounts of snowfall being collected more often.

This ensures we still collect images without snowfall (as this is a useful edge-case for training , as otherwise we may not be able to properly handle sparse cases), but ensures we aren't saturated with these images alone. 

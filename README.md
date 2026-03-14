# SatelliteSnowAccumAnalysis
DS4520 Project for Predicting Changes in Snow Accumulation from Satellite Imagery!

## Initial Project Structure
The project currently has 3 modules:
- Bayesian_Predictor.py
- CNN_Predictor.py
- ImageDataset.py

A docker container is provided to abstract out environment setup.
To launch any file in docker, you can run:

`docker compose run --rm app python [file_path]`
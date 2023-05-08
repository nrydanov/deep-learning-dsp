# Guitar effects emulation

Repository overview:

1. `model` — all `.py` project sources
2. `config` — configuration files for models
3. `data` — folder with train/inference data

    NOTE: This folder will be mounted into Docker container.

4. `logs` — training logs information
5. `checkpoints` — model training checkpoints
6. `pyproject.toml` — poetry dependency file
7. `train_example.sh` — training example 
8. `inference_example.sh` — inference example

## Build

This application can be built locally or in Docker

### Locally

To build locally:

1. Install Python 3.10 via any package manager
2. Install Poetry 
3. Run `poetry install` in project root

### Docker

To build in Docker:

1. Run `docker-compose --build --build-arg PLATFORM=<PLATFORM> --build-arg DEVICE=<DEVICE>`

- `PLATFORM` — target platform of docker image. Expected values: `arm64/amd64/etc`
- `DEVICE` — inferencing device. Expected values: `cpu/cuda/mps`

Currently GPU computations is only available on `amd64` platform with `cuda` support.

## Prepare data

Of course, as every neural network, this one needs some data to work.

You can find it on your own and place in `data` folder or download original dataset with `download.sh` script in `data` folder.

NOTE: `wget` is required for script to work, so you will probably need to install it. You can also download data from browser using link provided in script.

## Run

To start container just run

```docker-compose up```

## Training

To train model run `train.py` script in `model` folder with following arguments:

1. `--model_type` — one of defined in `models.py` models.
2. `--model_config` — path to config for this model
   
    Predefined configs are available in `config` folder.
3. `--data_config` — path to config for `Dataset` object.
   
    This works only if model has constructor based on `pydantic` configuration file.
4. `--learning_rate` — model `learning_rate`
5. `--epochs` — number of epochs
6. `--batch_size` — batch size
7. `--attempt_name` — name of current attempt
   
    Based on this argument checkpoints and will be saved in `checkpoints/<attempt_name>.pt` and `logs/<attempt_name>.csv`.
8.  `--device` — computation device [Optional]

    Default: GPU if available else CPU
9.  `--restore_state` — path to `.pt` file with checkpoint [Optional]

    Default: `None`
10. `--level` — logging level

    Default: `INFO`
11. [Optional]

Example is available in `entrypoint.sh` file.

## Inference 

To inference use `inference.py` script located in `model` folder with following arguments:

1. `--model_type` — one of defined in `models.py` models
2. `--model_config` — path to config for this model
   
    Predefined configs are available in `config` folder.
3. `--checkpoint` — path to pre-trained checkpoint using `train.py` script or any other checkpoint with same JSON structure
4.  `--output_path` — path where should script save output of a model
5.  `--device` — computation device [Optional]

    Default: GPU if available else CPU
6.  `--batch_size` — batch size [Optional]


    Default: 65536
7.  `--duration` — how much time of input file should be inferred (in seconds) [Optional]

    Default: audio duration
8.  `--sr` — sample rate

    Default: 44100

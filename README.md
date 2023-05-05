# Guitar effects emulation

Repository overview:

1. `model` — all `.py` project sources
2. `config` — configuration files for models
3. `data` — folder with training/inferencing data

    NOTE: This folder will be mounted into Docker container.

4. `logs` — training logs information
5. `checkpoints` — model training checkpoints
6. `pyproject.toml` — poetry dependency file
7. `entrypoint.sh` — training example 

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

Currently GPU inferencing is only available on `amd64` platform with `cuda` support.

## Run

To start container just run

```docker-compose up```

## Training

To train model run `train.py` script with following arguments:

1. `--model_type` — one of defined in `models.py` models.
2. `--model_config` — path to config for this model.
   
    Predefined configs are available in `config` folder
3. `--data_config` — path to config for `Dataset` object.
   
    This works only if model has constructor based on `pydantic` configuration file
4. `--learning_rate` — model `learning_rate`.
5. `--epochs` — number of epochs.
6. `--batch_size` — batch size.
7. `--attempt_name` — name of current attempt. 
   
    Based on this argument checkpoints and will be saved in `checkpoints/<attempt_name>.pt` and `logs/<attempt_name>.csv`
8.  `--device` — inferencing device, defaults to GPU if available else CPU [Optional]
9.  `--restore_state` — path to `.pt` file with checkpoint [Optional] 
10. `--level` — logging level, defaults to `INFO` [Optional]

Example is available in `entrypoint.sh` file.


#!/bin/sh

poetry run python3 nn/train.py \
        --model_type=BaselineLSTM \
        --model_config=config/example/model.cfg \
        --data_config=config/example/data.cfg \
        --attempt_name=example

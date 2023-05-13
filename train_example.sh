#!/bin/sh

poetry run python3 nn/train.py \
        --model_type=BaselineLSTM \
        --epochs=100 \
        --model_config=config/example/model.cfg \
        --data_config=config/example/data.cfg \
        --learning_rate=0.001 \
        --batch_size=64 \
        --attempt_name=example

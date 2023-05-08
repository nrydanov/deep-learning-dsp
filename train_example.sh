#!/bin/sh

poetry run python3 model/train.py \
        --model_type=BaselineRNN \
        --epochs=100 \
        --model_config=config/lstm64/model.cfg \
        --data_config=config/lstm64/data.cfg \
        --learning_rate=0.001 \
        --batch_size=64 \
        --attempt_name=lstm64
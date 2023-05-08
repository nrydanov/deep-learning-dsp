#!/bin/sh

python3 model/train.py \
        --model_type=BaselineRNN \
        --epochs=100 \
        --model_config=config/example/model.cfg \
        --data_config=config/example/data.cfg \
        --learning_rate=0.001 \
        --batch_size=64 \
        --attempt_name=example
#!/bin/sh

poetry run python3 model/train.py \
        --model_type=BaselineRNN \
        --epochs=10 \
        --model_config=config/baseline/model.cfg \
        --data_config=config/baseline/data.cfg \
        --learning_rate=0.001 \
        --batch_size=64 \
        --save_path=./checkpoints/baseline.pt \
        --device=$TORCH_DEVICE
#!/bin/sh

poetry run python3 nn/inference.py \
    --model_type=BaselineLSTM \
    --model_config=config/lstm64/model.cfg \
    --checkpoint=checkpoints/lstm64.pt \
    --input=data/float32/test_in.wav \
    --output=output.wav \
    --duration=10

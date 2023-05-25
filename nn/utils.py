import argparse
import logging
import os
from argparse import ArgumentParser
from dataclasses import fields
from enum import Enum
from typing import Dict

import numpy as np
import pandas as pd
import torch
from auraloss.time import DCLoss, ESRLoss
from librosa import istft
from torch.utils.tensorboard import SummaryWriter


def get_required_for(cls, kwargs: Dict[str, object]) -> Dict[str, object]:
    return {
        k: v for k, v in kwargs.items() if k in list(map(lambda x: x.name, fields(cls)))
    }


def output_to_audio(data: np.ndarray, **kwargs) -> np.ndarray[np.float32]:
    data = data.swapaxes(1, 2)
    return np.stack([istft(x, **kwargs)] for x in data)


class ParserType(Enum):
    TRAIN = "train"
    INFERENCE = "inference"


def init_parser(type: ParserType) -> ArgumentParser:
    logging.info("Initializing parser")
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--device", type=str, required=False)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--loss", type=str, required=False, default="mse")
    if type == ParserType.TRAIN:
        parser.add_argument("--epochs", type=int, required=False, default=1000)
        parser.add_argument("--learning_rate", type=float, required=False, default=0.004)
        parser.add_argument("--attempt_name", type=str, required=True)
        parser.add_argument("--restore_state", action=argparse.BooleanOptionalAction)
        parser.add_argument("--level", type=str, required=False)
        parser.add_argument("--overwrite", type=bool, required=False, default=False)
        parser.add_argument("--scheduler", type=str, required=True)
        parser.add_argument("--batch_size", type=int, required=False, default=26)
    else:
        parser.add_argument("--batch_size", type=int, required=False, default=65536)
        parser.add_argument("--checkpoint", type=str, required=True)
        parser.add_argument("--input", type=str, required=True)
        parser.add_argument("--output", type=str, required=False)
        parser.add_argument("--duration", type=int, required=False, default=None)
        parser.add_argument("--sr", type=int, required=False, default=44100)
        parser.add_argument("--test", type=str, required=False)
    return parser


def init_device(device: str) -> torch.device:
    if device is not None:
        logging.info(f'Using device from command line: "{device}"')
        return torch.device(device)
    else:
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            name = "mps"
        elif torch.cuda.is_available():
            name = "cuda"
        else:
            name = "cpu"
        logging.info(
            f'Device is not provided, selecting "{name}" based on current environment'
        )
        return torch.device(name)


def init_loss(loss: str) -> torch.nn.Module:
    match loss:
        case "mse":
            return torch.nn.MSELoss(reduction='sum')
        case "esr":
            return ESRLoss()
        case _:
            raise ValueError("Got an unexpected loss function")



def init_logger(args) -> None:
    # TODO Add proper logging configuration
    FORMAT = "[%(levelname)s] [%(asctime)s] %(message)s"
    logging.basicConfig(format=FORMAT, datefmt="%m-%d %H:%M:%S", force=True)
    logging.getLogger().setLevel(logging.INFO)


def empty_cache(device) -> None:
    match device.type:
        case "cuda":
            torch.cuda.empty_cache()
        case "mps":
            torch.mps.empty_cache()
        case "cpu":
            pass
        case _:
            raise ValueError("Got an unexpected device")


def save_history(writer: str, attempt_name: str, history: dict):
    log_dir = f"tensorboard/{attempt_name}"

    for key, value in history.items():
        if key == "epoch":
            continue
        writer.add_scalar(key, value, history["epoch"])

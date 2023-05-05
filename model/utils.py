import numpy as np
import pandas as pd
import torch
import logging
import os

from dataclasses import fields
from librosa import istft
from argparse import ArgumentParser
from typing import Dict


def get_required_for(cls, kwargs: Dict[str, object]) -> Dict[str, object]:
    return {
        k: v for k, v in kwargs.items() if k in list(map(lambda x: x.name, fields(cls)))
    }


def output_to_audio(data: np.ndarray, **kwargs) -> np.ndarray[np.float32]:
    data = data.swapaxes(1, 2)
    return np.stack([istft(x, **kwargs)] for x in data)


def init_parser(type: str) -> ArgumentParser:
    logging.info("Initializing parser")
    parser = ArgumentParser()
    if type == "train":
        parser.add_argument("--model_type", type=str, required=True)
        parser.add_argument("--epochs", type=int, required=True)
        parser.add_argument("--model_config", type=str, required=True)
        parser.add_argument("--data_config", type=str, required=True)
        parser.add_argument("--learning_rate", type=float, required=True)
        parser.add_argument("--batch_size", type=int, required=True)
        parser.add_argument("--attempt_name", type=str, required=True)
        parser.add_argument("--device", type=str, required=False)
        parser.add_argument("--restore_state", type=bool, required=False)
        parser.add_argument("--level", type=str, required=False)
    else:
        parser.add_argument("--model_type", type=str, required=True)
        parser.add_argument("--model_config", type=str, required=True)
        parser.add_argument("--checkpoint", type=str, required=True)
        parser.add_argument("--device", type=str, required=True)
        parser.add_argument("--input", type=str, required=True)
        parser.add_argument("--output_path", type=str, required=True)
        parser.add_argument("--batch_size", type=int, required=True)
        parser.add_argument("--duration", type=int, required=False, default=None)
    return parser


def init_device(device: str) -> torch.device:
    if device is not None:
        logging.info(f"Using device from command line: {device}")
        return torch.device(device)
    else:
        if torch.backends.mps.is_available() and torch.backends.mps.is_built():
            name = "mps"
        elif torch.cuda.is_available():
            name = "cuda"
        else:
            name = "cpu"
        logging.info(f"Device is not provided, selecting automatically: {device}")
        return torch.device(name)


def init_logger(args) -> None:
    # TODO Add proper logging configuration
    FORMAT = "[%(asctime)s] %(message)s"
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


def save_history(attempt_name: str, history: dict):
    os.makedirs("logs", exist_ok=True)

    path = f"logs/{attempt_name}.csv"
    try:
        logs = pd.read_csv(path, index_col="index")
    except OSError:
        logs = pd.DataFrame(columns=history.keys())

    logs.loc[logs.shape[0]] = history.values()

    logs.to_csv(path, index_label="index")

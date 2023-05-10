import inspect
import logging
import sys

import torch
from pydantic import BaseSettings
from torch.nn import (
    LSTM,
    BatchNorm1d,
    Conv1d,
    Dropout,
    Linear,
    Module,
    Sequential,
)

from data import BaseDataset, WaveformDataset


class BaseModel(Module):
    def get_provider(self, **kwargs) -> BaseDataset:
        return NotImplementedError()


class BaselineRNN(Module):
    def __init__(self, config) -> None:
        super().__init__()

        hidden_size = config.hidden_size

        self.lstm = LSTM(1, hidden_size, batch_first=True)
        self.linear = Linear(hidden_size, 1)
        self.h_0 = None
        self.c_0 = None

    class Settings(BaseSettings):
        hidden_size: int

        class Config:
            pass

    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        if self.training:
            x, (h_n, c_n) = self.lstm(x0)
        else:
            if self.h_0 is None:
                x, (h_n, c_n) = self.lstm(x0)
            else:
                batch_size = x0.size(dim=0)
                x, (h_n, c_n) = self.lstm(
                    x0, (self.h_0[:, :batch_size, :], self.c_0[:, :batch_size, :])
                )

        self.h_0, self.c_0 = (h_n.detach(), c_n.detach())
        x = self.linear(x)
        return torch.add(x, x0)

    def train(self, mode=True):
        self.h_0, self.c_0 = None, None
        return super().train(mode)

    def eval(self):
        self.h_0, self.c_0 = None, None
        return super().eval()

    def get_provider(self):
        return WaveformDataset


class BaselineRNN2(Module):
    def __init__(self, config) -> None:
        super().__init__()

        hidden_size = config.hidden_size

        self.seq = Sequential(
            Conv1d(1, 32, 5, 3, padding_mode="zeros"),
            Conv1d(32, 32, 5, 3, padding_mode="zeros"),
        )

        self.seq2 = Sequential(
            LSTM(1, hidden_size, batch_first=True),
            Dropout(p=0.2, inplace=True),
            BatchNorm1d(hidden_size),
            Linear(hidden_size, 1),
        )

    class Settings(BaseSettings):
        hidden_size: int

        class Config:
            pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        result = self.seq(x)

        result = torch.transpose(result, 1, 2)

        return result


def get_model(name: str):
    logging.info(f"Selecting model: {name}")
    is_class_member = (
        lambda member: inspect.isclass(member) and member.__module__ == __name__
    )
    models = inspect.getmembers(sys.modules[__name__], is_class_member)

    for entry in models:
        if entry[0] == name:
            return entry[1]

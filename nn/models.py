import inspect
import logging
import sys
from typing import Optional, Type

import torch
from datasets import BaseDataset, STFTDataset, WaveformDataset
from pydantic import BaseSettings
from torch.nn import LSTM, Linear, Module, BatchNorm1d


class BaseModel(Module):
    def get_provider(self, **kwargs) -> Type[BaseDataset]:
        return NotImplementedError()


class BaselineLSTM(Module):
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

    def get_provider(self) -> Type[BaseDataset]:
        return WaveformDataset


class FourierLSTM(Module):
    def __init__(self, config) -> None:
        super().__init__()

        input_size = 1 + config.n_fft // 2
        hidden_size = config.hidden_size

        self.lstm = LSTM(input_size, hidden_size, batch_first=True)
        self.linear = Linear(hidden_size, input_size)
        self.h_0 = None
        self.c_0 = None

    class Settings(BaseSettings):
        hidden_size: int
        n_fft: int

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

    def get_provider(self) -> Type[BaseDataset]:
        return STFTDataset


def get_model(name: str) -> Optional[Type[BaseModel]]:
    logging.info(f"Selecting model: {name}")
    is_class_member = (
        lambda member: inspect.isclass(member) and member.__module__ == __name__
    )
    models = inspect.getmembers(sys.modules[__name__], is_class_member)

    for entry in models:
        if entry[0] == name:
            return entry[1]

    return None

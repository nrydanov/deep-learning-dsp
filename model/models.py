import inspect
import logging
import sys

import torch
from pydantic import BaseSettings
from torch.nn import (
    LSTM,
    BatchNorm1d,
    Conv1d,
    Conv2d,
    ConvTranspose2d,
    Dropout,
    Linear,
    MaxPool2d,
    MaxUnpool2d,
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


class UNetSTFT(Module):
    def __init__(self) -> None:
        super().__init__()

        self.conv1 = Conv2d(1, 32, 8, 4)
        self.maxpool1 = MaxPool2d(2, return_indices=True)
        self.conv2 = Conv2d(32, 64, 4, 2)
        self.maxpool2 = MaxPool2d(2, return_indices=True)
        self.conv3 = Conv2d(64, 128, 2, 1)

        self.tconv3 = ConvTranspose2d(128, 64, 2)
        self.unpool2 = MaxUnpool2d(2)
        self.tconv2 = ConvTranspose2d(64, 32, 4, 2)
        self.unpool1 = MaxUnpool2d(2)
        self.tconv1 = ConvTranspose2d(32, 1, 8, 4)

    def forward(self, x0: torch.Tensor) -> torch.Tensor:
        x1 = self.conv1(x0)
        x2, ind2 = self.maxpool1(x1)
        x3 = self.conv2(x2)
        x4, ind4 = self.maxpool2(x3)
        x5 = self.conv3(x4)

        x4b = torch.add(self.tconv3(x5), x4)
        x3b = torch.add(self.unpool2(x4b, indices=ind4, output_size=x3.size()), x3)
        x2b = torch.add(self.tconv2(x3b, output_size=x2.size()), x2)
        x1b = torch.add(self.unpool1(x2b, indices=ind2, output_size=x1.size()), x1)
        x0b = torch.add(self.tconv1(x1b, output_size=x0.size()), x0)

        return x0b


def get_model(name: str):
    logging.info(f"Selecting model: {name}")
    is_class_member = (
        lambda member: inspect.isclass(member) and member.__module__ == __name__
    )
    models = inspect.getmembers(sys.modules[__name__], is_class_member)

    for entry in models:
        if entry[0] == name:
            return entry[1]

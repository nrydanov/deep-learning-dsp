import torch
import sys
import inspect
import numpy as np

from torchaudio.transforms import Spectrogram, InverseSpectrogram
from torch.nn import Module, LSTM, Linear, Conv2d, \
    Dropout, BatchNorm1d, Sequential, MaxPool2d, ConvTranspose2d, MaxUnpool2d
from data import WaveformDataset, STFTDataset, BaseDataset
from pydantic import BaseSettings
    
class BaseModel(Module):
  
  def get_provider(self, **kwargs) -> BaseDataset:
    return NotImplementedError()

class BaselineRNN(Module):
  
  def __init__(self, config) -> None:
    super().__init__()
    
    hidden_size = config.hidden_size
    
    self.lstm = LSTM(1, hidden_size)
    self.linear = Linear(hidden_size, 1)
    
  class Settings(BaseSettings):
    hidden_size: int
    
    class Config:
      pass
    
  def forward(self, x0: torch.Tensor) -> torch.Tensor:
    x, _ = self.lstm(x0)
    x = self.linear(x)
    return torch.add(x, x0)
  
  def get_provider(self):
    return WaveformDataset
  
class BaselineRNN2(Module):
  
  def __init__(self, hidden_size) -> None:
    super().__init__()
    
    self.seq = Sequential(
      Conv2d(1, 32, 5, 3, padding_mode='zeros'),
      Conv2d(32, 32, 5, 3, padding_mode='zeros'),
      LSTM(32, hidden_size),
      Dropout(p=0.2, inplace=True),
      BatchNorm1d(hidden_size),
      Linear(hidden_size, 1)
    )
    
  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.seq(x)
  
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
  is_class_member = lambda member: inspect.isclass(member) \
      and member.__module__ == __name__
  models = inspect.getmembers(sys.modules[__name__], is_class_member)
  
  for entry in models:
    if entry[0] == name:
      return entry[1]
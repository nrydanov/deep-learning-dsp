import keras
import librosa
import numpy as np

from typing import Tuple

from dataclasses import dataclass
from utils import get_required_for

    
@dataclass
class STFTLoader:
  n_fft: int = 2048
  hop_length: int = 512
  freq_bins = 1 + n_fft // 2
  
  def transform(self, data: np.ndarray) -> np.ndarray[np.float32]:
    spec = librosa.stft(y=data, n_fft=self.n_fft,
                        hop_length=self.hop_length)
    return spec
  
  def process(self, data: np.ndarray)-> np.ndarray[np.float32]:
    return self.transform(data).swapaxes(0, 1)

@dataclass
class BaseLoader(keras.utils.Sequence):
  input_path: str = None
  output_path: str = None
  batch_size: int = 32
  input_size: int = 4410
  duration: int = None
  offset: int = 0
  preemphasis: bool = True
    
  def load_data(self) -> Tuple[np.ndarray[np.float32], np.ndarray[np.float32]]:
    in_data, _ = librosa.load(path=self.input_path,
      sr=44100,
      mono=True,
      offset=self.offset,
      duration=self.duration)
    out_data, _ = librosa.load(path=self.output_path,
      sr=44100,
      mono=True,
      offset=self.offset,
      duration=self.duration)

    if self.preemphasis:
      in_data = librosa.effects.preemphasis(in_data)
      out_data = librosa.effects.preemphasis(out_data)

    self.len = in_data.shape[0]

    return in_data, out_data
  
  def __post_init__(self) -> None:
    self.x, self.y = self.load_data()


class WindowWaveformLoader(BaseLoader):
  
  def load_data(self) -> Tuple[np.ndarray[np.float32], np.ndarray[np.float32]]:
    
    in_data, out_data = super().load_data()
    out_data = out_data[self.input_size - 1:]
    
    return in_data, out_data
    
  def __getitem__(self, index: int) -> Tuple[np.ndarray[np.float32], 
                                             np.ndarray[np.float32]]:
    
    start = index * self.batch_size
    end = (index + 1) * self.batch_size
    
    x_cur = np.stack([self.x[i:i + self.input_size] for i in range(start, end)])
    y_cur = self.y[start:end]
    
    x_cur = x_cur.reshape(self.batch_size, self.input_size, 1)
    y_cur = y_cur.reshape(self.batch_size, 1)
    
    return x_cur, y_cur 
  
  def __len__(self) -> int:
    return (self.len - self.input_size + 1) // self.batch_size


class DirectWaveformLoader(BaseLoader):
    
  def __getitem__(self, index: int) -> Tuple[np.ndarray[np.float32], 
                                             np.ndarray[np.float32]]:
    
    start = index * self.batch_size * self.input_size
    end = (index + 1) * self.batch_size * self.input_size
    
    x_cur = np.stack([self.x[i:i + self.input_size] 
                      for i in range(start, end, self.input_size)])
    y_cur = np.stack([self.y[i:i + self.input_size] 
                      for i in range(start, end, self.input_size)])
    
    x_cur = x_cur.reshape(self.batch_size, self.input_size, 1)
    y_cur = y_cur.reshape(self.batch_size, self.input_size, 1)
    
    return x_cur, y_cur 
  
  def __len__(self) -> int:
    return self.len // (self.batch_size * self.input_size)


class DirectSTFTLoader(DirectWaveformLoader, STFTLoader):
  
  def __init__(self, **kwargs) -> None:
    DirectWaveformLoader.__init__(self, **get_required_for(DirectWaveformLoader,
                                                           kwargs))
    STFTLoader.__init__(self, **get_required_for(STFTLoader, kwargs))
  
  def __getitem__(self, index: int) -> Tuple[np.ndarray[np.float32], 
                                             np.ndarray[np.float32]]:
    x_cur, y_cur = super().__getitem__(index)
    x_cur = np.apply_along_axis(self.process, 1, x_cur) \
      .reshape(self.batch_size, -1, self.freq_bins)
    y_cur = np.apply_along_axis(self.process, 1, y_cur) \
      .reshape(self.batch_size, -1, self.freq_bins)
    
    return np.abs(x_cur), np.abs(y_cur)


class WindowSTFTLoader(WindowWaveformLoader, STFTLoader):
  
  def __init__(self, **kwargs) -> None:
    WindowWaveformLoader.__init__(self, **get_required_for(WindowWaveformLoader, 
                                                           kwargs))
    STFTLoader.__init__(self, **get_required_for(STFTLoader, kwargs))
  
  def __getitem__(self, index: int) -> Tuple[np.ndarray[np.float32],
                                             np.ndarray[np.float32]]:
    x_cur, y_cur = super().__getitem__(index)

    x_cur = np.apply_along_axis(self.process, 1, x_cur) \
      .reshape(self.batch_size, -1, self.freq_bins)
    
    return np.abs(x_cur), y_cur


@dataclass
class DataSplitter:
  input_path: str
  output_path: str
  duration: int = None
  train_size: float = 0.7
  val_size: float = 0.2
  test_size: float = 0.0
  batch_size: int = 32
  input_size: int = 150
  preemphasis: bool = True
  valid_only: bool = True

  def split(self, cls, **kwargs) -> Tuple[BaseLoader]:
    if self.duration is None:
      length = librosa.get_duration(path=self.input_path)
    else:
      length = self.duration

    train_duration = int(length * self.train_size)
    val_duration = int(length * self.val_size)
    test_duration = int(length * self.val_size)
    
    train = cls(input_path=self.input_path, output_path=self.output_path,
                batch_size=self.batch_size, input_size=self.input_size,
                duration=train_duration,
                offset=val_duration + test_duration,
                preemphasis=self.preemphasis, **kwargs)
    
    valid = cls(input_path=self.input_path, output_path=self.output_path,
                batch_size=self.batch_size, input_size=self.input_size,
                duration=val_duration,
                offset=0,
                preemphasis=self.preemphasis, **kwargs)
    
    if not self.valid_only:
      test = cls(input_path=self.input_path, output_path=self.output_path,
                 batch_size=self.batch_size, input_size=self.input_size,
                 duration=test_duration,
                 offset=val_duration,
                 preemphasis=self.preemphasis, **kwargs)
      return train, valid, test
    else:
      return train, valid
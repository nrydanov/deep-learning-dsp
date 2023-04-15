import keras
import librosa
import numpy as np
from dataclasses import dataclass

@dataclass
class DataSplitter:
  input_path: str
  output_path: str
  duration: int = None
  train_size: float = 0.7
  val_size: float = 0.2
  test_size: float = 0.0
  batch_size: int = 2 ** 12
  input_size: int = 150
  preemphasis: bool = True
  valid_only: bool = True
    
  def split(self, cls) -> tuple:
    if self.duration is None:
      length = librosa.get_duration(path=self.input_path)
    else:
      length = self.duration
    
    train_duration = int(length * self.train_size)
    val_duration = int(length * self.val_size)
    test_duration = int(length * self.val_size)
    
    train = cls(self.input_path, self.output_path, 
                self.batch_size, self.input_size, 
                duration=train_duration,
                offset=val_duration + test_duration,
                preemphasis=self.preemphasis)
    
    valid = cls(self.input_path, self.output_path, 
                self.batch_size, self.input_size, 
                duration=val_duration,
                offset=0,
                preemphasis=self.preemphasis)
    
    if not self.valid_only:
      test = cls(self.input_path, self.output_path, 
                  self.batch_size, self.input_size, 
                  duration=test_duration,
                  offset=val_duration,
                  preemphasis=self.preemphasis)
      return train, valid, test
    else:
      return train, valid

@dataclass
class BaseLoader(keras.utils.Sequence):
  input_path: str
  output_path: str
  batch_size: int 
  input_size: int
  duration: int
  offset: int = 0
  preemphasis: bool = True
  
  def __post_init__(self) -> None:
        
    self.x, self.y = self.load_data()
    
    def load_data(self):
      raise NotImplementedError()
    
class WindowWaveformLoader(BaseLoader):
  
  def load_data(self):
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
    
    out_data = out_data[self.input_size - 1:]
    
    return in_data, out_data
    
  def __getitem__(self, index: int) -> tuple:
    
    start = index * self.batch_size
    end = (index + 1) * self.batch_size
    
    x_cur = np.stack([self.x[i:i + self.input_size] for i in range(start, end)])
    y_cur = self.y[start:end]
    
    x_cur = x_cur.reshape(self.batch_size, self.input_size, 1)
    y_cur = y_cur.reshape(self.batch_size, 1)
    
    return x_cur, y_cur 
  
  def __len__(self) -> int:
    return (self.len - self.input_size + 1) // self.batch_size

class RawWaveformLoader(BaseLoader):
  
  def load_data(self):
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
    
  def __getitem__(self, index: int) -> tuple:
    
    start = index * self.batch_size * self.input_size
    end = (index + 1) * self.batch_size * self.input_size
    
    x_cur = np.stack([self.x[i:i + self.input_size] for i in range(start, end, self.input_size)])
    y_cur = np.stack([self.y[i:i + self.input_size] for i in range(start, end, self.input_size)])
    
    x_cur = x_cur.reshape(self.batch_size, self.input_size, 1)
    y_cur = y_cur.reshape(self.batch_size, self.input_size, 1)
    
    return x_cur, y_cur 
  
  def __len__(self) -> int:
    return self.len // (self.batch_size * self.input_size)
  
# class STFTWaveformLoader(BaseLoader):
#   n_fft: int
#   hop_length: int
  
#   def load_data(self):
#     in_data, _ = librosa.load(path=self.input_path,
#         sr=44100,
#         mono=True,
#         offset=self.offset,
#         duration=self.duration)
    
#     out_data, _ = librosa.load(path=self.output_path,
#         sr=44100,
#         mono=True,
#         offset=self.offset,
#         duration=self.duration)
    
#     if self.preemphasis:
#       in_data = librosa.effects.preemphasis(in_data)
#       out_data = librosa.effects.preemphasis(out_data)
    
#     self.len = in_data.shape[0]
    
#     return in_data, out_data
    
#   def __getitem__(self, index: int) -> tuple:
    
#     start = index * self.batch_size * self.input_size
#     end = (index + 1) * self.batch_size * self.input_size
    
#     x_cur = np.stack([self.x[i:i + self.input_size] for i in range(start, end, self.input_size)])
#     y_cur = np.stack([self.y[i:i + self.input_size] for i in range(start, end, self.input_size)])
    
#     x_cur = x_cur.reshape(self.batch_size, self.input_size, 1)
#     y_cur = y_cur.reshape(self.batch_size, self.input_size, 1)
    
#     return x_cur, y_cur 
  
#   def __len__(self) -> int:
#     return self.len // (self.batch_size * self.input_size)
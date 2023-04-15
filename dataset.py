import keras
import librosa
import numpy as np
from dataclasses import dataclass
from utils import get_required_for


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

  def split(self, cls, **kwargs) -> tuple:
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
    
@dataclass
class MelLoader:
  n_fft: int = 2048
  sr: int = 44100
  hop_length: int = 512
  n_mels: int = 128
  fmax: int = 8000
  
  def convert_to_mel(self, data: np.ndarray):
    spec = librosa.feature.melspectrogram(y=data, sr=self.sr, n_fft=self.n_fft,
                                          hop_length=self.hop_length, 
                                          n_mels=self.n_mels, fmax=self.fmax)

    spec_db = librosa.power_to_db(spec, ref=np.max)
    return spec_db
  
  def process(self, data: np.ndarray):
    return self.convert_to_mel(data).transpose()

@dataclass
class BaseLoader(keras.utils.Sequence):
  input_path: str = None
  output_path: str = None
  batch_size: int = 32
  input_size: int = 4410
  duration: int = None
  offset: int = 0
  preemphasis: bool = True
    
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
  
  def __post_init__(self) -> None:
        
    self.x, self.y = self.load_data()
    
class WindowWaveformLoader(BaseLoader):
  
  def load_data(self):
    
    in_data, out_data = super().load_data()
    
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

class DirectWaveformLoader(BaseLoader):
    
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
  
class DirectMelLoader(DirectWaveformLoader, MelLoader):
  def __init__(self, **kwargs):

    DirectWaveformLoader.__init__(self, **get_required_for(DirectWaveformLoader, kwargs))
    MelLoader.__init__(self, **get_required_for(MelLoader, kwargs))
  
  def __getitem__(self, index: int) -> tuple:
    
    start = index * self.batch_size * self.input_size
    end = (index + 1) * self.batch_size * self.input_size
    
    x_cur = np.stack([self.x[i:i + self.input_size] for i in range(start, end, self.input_size)])
    y_cur = np.stack([self.y[i:i + self.input_size] for i in range(start, end, self.input_size)])
    
    x_cur = x_cur.reshape(self.batch_size, self.input_size, 1)
    y_cur = y_cur.reshape(self.batch_size, self.input_size, 1)
    
    x_cur = np.apply_along_axis(self.process, 1, x_cur).reshape(self.batch_size, self.n_mels, -1)
    y_cur = np.apply_along_axis(self.process, 1, y_cur).reshape(self.batch_size, self.n_mels, -1)
    
    return x_cur, y_cur
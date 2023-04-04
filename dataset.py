import keras
import librosa
import numpy as np

class BaseLoader(keras.utils.Sequence):
   def __init__(self, 
               input_path: str, 
               output_path: str, 
               batch_size: int, 
               input_size: int,
               duration: int,
               offset=0,
               preemphasis=True) -> None:
    self.input_path = input_path
    self.output_path = output_path
    self.batch_size = batch_size
    self.input_size = input_size
    self.offset = offset
    self.preemphasis = preemphasis
    self.duration = duration
        
    self.x, self.y = self.load_interval()
    
    self.y = self.y[input_size - 1:]
    
    def load_interval(self):
      raise NotImplementedError()
    
class RawWaveformLoader(BaseLoader):
  
  def load_interval(self):
      
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
    
    start = index * self.batch_size
    end = (index + 1) * self.batch_size
    
    x_cur = np.stack([self.x[i:i + self.input_size] for i in range(start, end)])
    y_cur = self.y[start:end]
    
    x_cur = x_cur.reshape(self.batch_size, self.input_size, 1)
    y_cur = y_cur.reshape(self.batch_size, 1)
    
    return x_cur, y_cur 
  
  def __len__(self) -> int:
    return (self.len - self.input_size + 1) // self.batch_size
  
  class DataSplitter:
    def __init__(self,
                input_path: str,
                output_path: str,
                train_time: int, # in seconds 
                valid_time: int, # in seconds
                test_time = 0, # in seconds
                batch_size = 32,
                input_size = 44,
                preemphasis=True,
                valid_only=True) -> None:
      self.input_path = input_path
      self.output_path = output_path
      
      self.test_time = test_time
      self.valid_time = valid_time
      self.train_time = train_time
      
      self.batch_size = batch_size
      self.input_size = input_size
      
      self.preemphasis = preemphasis
      self.valid_only = valid_only
      
    def split(self) -> tuple:
      train = RawWaveformLoader(self.input_path, self.output_path, 
                            self.batch_size, self.input_size, 
                            duration=self.train_time,
                            offset=self.valid_time + self.test_time,
                            preemphasis=self.preemphasis)
      valid = RawWaveformLoader(self.input_path, self.output_path, 
                            self.batch_size, self.input_size, 
                            duration=self.valid_time,
                            offset=0,
                            preemphasis=self.preemphasis)
      if not self.valid_only:
        test = RawWaveformLoader(self.input_path, self.output_path, 
                              self.batch_size, self.input_size, 
                              duration=self.test_time,
                              offset=self.valid_time,
                              preemphasis=self.preemphasis)
        return train, valid, test
      else:
        return train, valid
      
class STFTLoader(BaseLoader):
  pass
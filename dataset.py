import keras
import librosa
import numpy as np
import random


class BaseLoader(keras.utils.Sequence):
   def __init__(self, 
               input_path: str, 
               output_path: str, 
               batch_size: int, 
               input_size: int,
               interval_duration=10 * 60,
               start_offset=0,
               end_offset=None,
               preemphasis=True,
               dynamic=False) -> None:
    self.input_path = input_path
    self.output_path = output_path
    self.batch_size = batch_size
    self.input_size = input_size
    self.interval_duration = interval_duration
    self.start_offset = start_offset
    self.batch_count = end_offset
    self.preemphasis = preemphasis
    self.dynamic = dynamic
    
    if end_offset is None:
      self.end_offset = librosa.get_duration(path=input_path, sr=44100)
    else:
      self.end_offset = end_offset
        
    self.x, self.y = self.__select_interval__()
    
    def __select_interval__(self):
      raise NotImplementedError()
    

class RawWaveformLoader(BaseLoader):
  
  def __select_interval__(self):
    
    if self.dynamic:
      current_offset = random.randint(self.start_offset, int(self.end_offset) - self.interval_duration)
    else:
      current_offset = self.start_offset
      
    in_data, _ = librosa.load(path=self.input_path,
        sr=44100,
        mono=True,
        offset=current_offset,
        duration=self.interval_duration)
    out_data, _ = librosa.load(path=self.output_path,
        sr=44100,
        mono=True,
        offset=current_offset,
        duration=self.interval_duration)
    
    if self.preemphasis:
      in_data = librosa.effects.preemphasis(in_data)
      out_data = librosa.effects.preemphasis(out_data)
    
    self.len = in_data.shape[0]
    
    return in_data, out_data
    
  def __getitem__(self, index: int) -> tuple:
    
    start = index * self.batch_size * self.input_size
    end = (index + 1) * self.batch_size * self.input_size
    
    x_cur = np.array(np.stack([self.x[i:i + self.input_size] for i in range(start, end, self.input_size)]))
    y_cur = np.array(np.stack([self.y[i:i + self.input_size] for i in range(start, end, self.input_size)]))
    
    return x_cur, y_cur
  
  def __len__(self) -> int:
    return self.len // (self.batch_size * self.input_size)
  
  def on_epoch_end(self) -> None:
    if self.dynamic:
      self.x, self.y = self.__select_interval__()
  
  class DataSplitter:
    def __init__(self,
                input_path: str,
                output_path: str,
                train_per_epoch: int, # in seconds 
                valid_time: int, # in seconds
                test_time = 0, # in seconds
                batch_size = 32,
                input_size = 441,
                preemphasis=True,
                dynamic_train=True,
                valid_only=True) -> None:
      self.input_path = input_path
      self.output_path = output_path
      self.test_time = test_time
      self.valid_time = valid_time
      self.batch_size = batch_size
      self.input_size = input_size
      self.preemphasis = preemphasis
      self.train_per_epoch = train_per_epoch
      self.dynamic_train = dynamic_train
      self.valid_only = valid_only
      
    def split(self) -> tuple:
      train = RawWaveformLoader(self.input_path, self.output_path, 
                            self.batch_size, self.input_size, 
                            interval_duration=self.train_per_epoch,
                            start_offset=self.valid_time + self.test_time, end_offset=None,
                            preemphasis=self.preemphasis, dynamic=self.dynamic_train)
      valid = RawWaveformLoader(self.input_path, self.output_path, 
                            self.batch_size, self.input_size, 
                            interval_duration=self.valid_time,
                            start_offset=0,
                            preemphasis=self.preemphasis)
      if not self.valid_only:
        test = RawWaveformLoader(self.input_path, self.output_path, 
                              self.batch_size, self.input_size, 
                              interval_duration=self.test_time,
                              start_offset=self.valid_time,
                              preemphasis=self.preemphasis)
        return train, valid, test
      else:
        return train, valid
      
class STFTLoader(BaseLoader):
  pass
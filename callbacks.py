from keras.callbacks import ModelCheckpoint, Callback
from scipy.io import wavfile
import keras
import os
import math
import keras.backend as K


class CheckpointAndSave(ModelCheckpoint):
  cached = False
  test_result = None
  
  def __init__(self, filepath, demo_data, model: keras.Model, batch_size: int, monitor: str = "val_loss", verbose: int = 0, 
               save_best_only: bool = False, save_weights_only: bool = False, 
               mode: str = "auto", save_freq="epoch", options=None, 
               initial_value_threshold=None, **kwargs):
    
    super().__init__(filepath, monitor, verbose, save_best_only, 
                     save_weights_only, mode, save_freq, options, 
                     initial_value_threshold, **kwargs)
    self.demo_data = demo_data
    self.model = model
    self.batch_size = batch_size
    
  def save_demo(self, data, save_path):
    if not CheckpointAndSave.cached:
      CheckpointAndSave.test_result = self.model.predict(data, batch_size=self.batch_size).reshape(-1, 1)
      
    wavfile.write(save_path, 44100, CheckpointAndSave.test_result)
    CheckpointAndSave.cached = True
  
  def on_epoch_end(self, epoch, logs=None):

    folder = f"{self.filepath}/demo/"
    if not os.path.exists(folder):
      os.makedirs(folder)
      
    if self.save_best_only:
      current = logs.get(self.monitor)
      if self.monitor_op(current, self.best):
        self.save_demo(self.demo_data, f"{folder}/{epoch + 1}.wav")
      
    super().on_epoch_end(epoch, logs)
    
  def on_epoch_begin(self, epoch, logs=None):
    CheckpointAndSave.cached = False

class CosineAnnealingScheduler(Callback):
    """Cosine annealing scheduler.
    """

    def __init__(self, T_max, eta_max, eta_min=0, verbose=0):
        super(CosineAnnealingScheduler, self).__init__()
        self.T_max = T_max
        self.eta_max = eta_max
        self.eta_min = eta_min
        self.verbose = verbose

    def on_epoch_begin(self, epoch, logs=None):
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        lr = self.eta_min + (self.eta_max - self.eta_min) * (1 + math.cos(math.pi * epoch / self.T_max)) / 2
        K.set_value(self.model.optimizer.lr, lr)
        if self.verbose > 0:
            print('\nEpoch %05d: CosineAnnealingScheduler setting learning '
                  'rate to %s.' % (epoch + 1, lr))

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        logs['lr'] = K.get_value(self.model.optimizer.lr)
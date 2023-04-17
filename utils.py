import json
import git
import os
from keras import Model
from dataclasses import fields
import numpy as np
from librosa import istft

from typing import Dict

def save_attempt_data(path: str, model: Model) -> None:
  if not os.path.exists(path):
      os.makedirs(path)

  repo = git.Repo(search_parent_directories=True)
  data = {'commit': repo.head.object.hexsha}
  
  with open(f"{path}/history.json", 'w') as f:
    json.dump(data, f)

def get_required_for(cls, kwargs) -> Dict[str, object]:
  return {k : v for k, v in kwargs.items() 
          if k in list(map(lambda x: x.name, fields(cls)))}

def output_to_audio(data: np.ndarray, **kwargs) -> np.ndarray[np.float32]:
  data = data.swapaxes(1, 2)
  return np.stack([istft(x, **kwargs)] for x in data)
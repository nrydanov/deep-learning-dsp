import json
import git
import os
from keras import Model
from dataclasses import fields

def save_attempt_data(path: str, model: Model):
  if not os.path.exists(path):
      os.makedirs(path)

  repo = git.Repo(search_parent_directories=True)

  data = {'commit': repo.head.object.hexsha}

  f = open(f"{path}/history.json", 'w')

  json.dump(data, f)
  
  f.close()  
def get_required_for(cls, kwargs):
  return {k : v for k, v in kwargs.items() if k in list(map(lambda x: x.name, fields(cls)))}
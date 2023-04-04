import json
import git
import os
from keras import Model

def save_attempt_data(path: str, model: Model):
  if not os.path.exists(path):
      os.makedirs(path)
      
  repo = git.Repo(search_parent_directories=True)
  
  data = {'commit': repo.head.object.hexsha}
  
  f = open(f"{path}/history.json", 'w')
  
  json.dump(data, f)
  
  f.close()
  
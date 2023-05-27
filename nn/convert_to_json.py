from models import get_model
import argparse
import json
from json import JSONEncoder
from torch.utils.data import Dataset
import torch

class EncodeTensor(JSONEncoder,Dataset):
    def default(self, obj):
        if isinstance(obj, torch.Tensor):
            return obj.cpu().detach().numpy().tolist()
        return super(json.NpEncoder, self).default(obj)

def main(args):
    model = get_model(args.model)
    model_config = model.Settings(_env_file=f"{args.config}/model.cfg")
    model = model(model_config)
    
    model.load_state_dict(torch.load(args.checkpoint)['model'])

    with open(args.output, 'w') as json_file:
        json.dump(model.state_dict(), json_file, cls=EncodeTensor)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()
    main(args)

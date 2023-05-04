import torch
from utils import init_parser, init_device, init_logger
from models import get_model

import librosa


def main():
    parser = init_parser("inference")
    args = parser.parse_args()
    init_logger(args)
    device: torch.device = init_device(args.device)

    model = get_model(args.model_type)

    model.load_state_dict(torch.load(args.checkpoint))
    model.to(device)

    model.eval()
    with torch.no_grad():
        inputs = torch.tensor(librosa.load(args.input))
        inputs.to(device)

        outputs = model(inputs)

        print(outputs.size())

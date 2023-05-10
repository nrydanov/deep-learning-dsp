import logging

import librosa
import numpy as np
import torch
from models import get_model
from scipy.io import wavfile
from torch.nn import MSELoss
from tqdm import tqdm
from utils import ParserType, init_device, init_logger, init_parser


def main():
    parser = init_parser(ParserType.INFERENCE)
    args = parser.parse_args()
    init_logger(args)
    device: torch.device = init_device(args.device)

    model = get_model(args.model_type)

    model_config = model.Settings(_env_file=args.model_config)
    model: torch.Module = model(model_config)

    model.load_state_dict(torch.load(args.checkpoint)["best_model"])
    model.to(device)

    logging.info("Loading input")
    data, _ = librosa.load(args.input, sr=args.sr, duration=args.duration)

    model.eval()
    logging.info("Starting inference loop")
    with torch.no_grad():
        result = np.array([], dtype=np.float32)
        for i in tqdm(range(0, data.shape[0], args.batch_size)):
            inputs = torch.tensor(data[i : i + args.batch_size].reshape(1, -1, 1)).to(
                device
            )
            outputs = model(inputs)

            result = np.append(result, outputs.cpu().numpy())

    if args.test is not None:
        test, _ = librosa.load(args.test, sr=args.sr, duration=args.duration)
        
        loss = MSELoss()
        total_loss = loss(torch.tensor(result).cpu(), torch.tensor(test).cpu())
        print(f"Test loss: {total_loss}")
        
    logging.info("Writing model output to file")
    wavfile.write(args.output, args.sr, result.reshape(-1, 1))


if __name__ == "__main__":
    main()

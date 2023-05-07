import torch
import librosa
import numpy as np

from tqdm import tqdm
from utils import init_parser, init_device, init_logger, ParserType
from models import get_model
from scipy.io import wavfile


def main():
    parser = init_parser(ParserType.INFERENCE)
    args = parser.parse_args()
    init_logger(args)
    device: torch.device = init_device(args.device)

    model = get_model(args.model_type)

    model_config = model.Settings(_env_file=args.model_config)
    model: torch.Module = model(model_config)

    model.load_state_dict(torch.load(args.checkpoint)["model"])
    model.to(device)

    data, _ = librosa.load(args.input, sr=args.sr, duration=args.duration)

    model.eval()
    with torch.no_grad():
        result = np.array([], dtype=np.float32)
        for i in tqdm(range(0, data.shape[0], args.batch_size)):
            inputs = torch.tensor(data[i: i + args.batch_size].reshape(-1, 1)).to(device)
            outputs  = model(inputs)
            
            result = np.append(result, outputs.cpu().numpy())
        wavfile.write(args.output_path, args.sr, result.reshape(-1, 1))


if __name__ == "__main__":
    main()

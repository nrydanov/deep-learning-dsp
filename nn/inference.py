import logging

import librosa
import numpy as np
import torch
from models import get_model
from scipy.io import wavfile
from torch.nn import MSELoss
from tqdm import tqdm
from utils import ParserType, init_device, init_logger, init_loss, init_parser


def main():
    parser = init_parser(ParserType.INFERENCE)
    args = parser.parse_args()
    init_logger(args)
    device = init_device(args.device)

    model = get_model(args.model)

    model_config = model.Settings(_env_file=f"{args.config}/model.cfg")
    model = model(model_config)

    model.load_state_dict(torch.load(args.checkpoint)["best_model"])
    model.to(device)

    provider = model.get_provider()
    config = provider.Settings(f"{args.config}/data.cfg")

    converter = provider.Converter(config)

    logging.info("Loading input")
    data, _ = librosa.load(args.input, sr=args.sr, duration=args.duration)

    model.eval()
    logging.info("Starting inference loop")
    with torch.no_grad():
        result = torch.tensor([], dtype=torch.float32).to(device)
        for i in tqdm(range(0, data.shape[0], args.batch_size)):
            encoded = converter.encode(data[i : i + args.batch_size].reshape(1, -1, 1))
            inputs = torch.tensor(encoded).to(device)
            outputs = model(inputs)
            result = torch.cat((result, outputs), 1)

        if args.test is not None:
            logging.info("Starting train data inference")
            loss = init_loss(args.loss)
            test_data, _ = librosa.load(args.test, sr=args.sr, duration=args.duration)
            expected = torch.tensor([], dtype=torch.float32).to(device)
            for i in tqdm(range(0, data.shape[0], args.batch_size)):
                encoded = converter.encode(
                    test_data[i : i + args.batch_size].reshape(1, -1, 1)
                )
                expected = torch.cat(
                    (expected, torch.tensor(encoded, dtype=torch.float32).to(device)), 1
                )
            total_loss = loss(expected, result).item()
            print(f"Test loss: {total_loss}")

    result = converter.decode(result).reshape(-1)

    logging.info("Writing model output to file")

    if args.output is None:
        output_path = f"outputs/{args.checkpoint.split('/')[-1][:-3]}.wav"
    else:
        output_path = args.output

    wavfile.write(output_path, args.sr, result.cpu().numpy().reshape(-1, 1))


if __name__ == "__main__":
    main()

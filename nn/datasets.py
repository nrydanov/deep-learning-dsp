from typing import Optional, Tuple

import librosa
import numpy as np
import torch
from pydantic import BaseSettings
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self, config: BaseSettings) -> None:
        raise NotImplementedError()

    class Settings(BaseSettings):
        in_path: str
        out_path: str
        sample_rate: int
        offset: int
        duration: Optional[int] = None
        preemphasis: bool = False

        class Config:
            pass


class WaveformDataset(BaseDataset):
    input: np.ndarray
    output: np.ndarray

    def __init__(self, config: BaseSettings) -> None:
        self.input, _ = librosa.load(
            path=config.in_path,
            sr=config.sample_rate,
            offset=config.offset,
            duration=config.duration,
        )

        self.output, _ = librosa.load(
            path=config.out_path,
            sr=config.sample_rate,
            offset=config.offset,
            duration=config.duration,
        )

        if config.preemphasis:
            self.input = librosa.effects.preemphasis(self.input)
            self.output = librosa.effects.preemphasis(self.output)

        self.config = config

        self.converter = WaveformDataset.Converter(self.config)

        np.random.seed(69)
        self.x, self.y = self.__generate_samples__()

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, i) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[i], self.y[i]

    def __generate_samples__(self) -> Tuple[np.ndarray, np.ndarray]:
        x, y = [], []
        step = self.config.sample_rate // 2
        for i in range(0, len(self.input), step):
            x_cur = self.input[i : i + step]
            y_cur = self.output[i : i + step]
            x.append(x_cur.reshape(-1, 1))
            y.append(y_cur.reshape(-1, 1))

        return np.array(x, dtype=np.float32), np.array(y, dtype=np.float32)

    class Converter:
        def __init__(self, config):
            pass

        def encode(self, input: torch.Tensor):
            return input

        def decode(self, output: torch.Tensor):
            return output


class STFTDataset(WaveformDataset):
    class Converter:
        def __init__(self, config):
            self.config = config

        def encode(self, inputs: np.ndarray) -> np.ndarray:
            def conversion(sample) -> np.ndarray:
                return np.abs(
                    librosa.stft(
                        sample,
                        n_fft=self.config.n_fft,
                        hop_length=self.config.hop_length,
                        win_length=self.config.win_length,
                    )
                )

            result = np.array(
                list(
                    map(
                        conversion,
                        inputs.reshape(inputs.shape[0], -1),
                    )
                )
            ).swapaxes(1, 2)

            return result

        def decode(self, outputs: torch.Tensor) -> np.ndarray:
            outputs = outputs.cpu().numpy().swapaxes(1, 2)

            def conversion(sample) -> np.ndarray:
                return librosa.griffinlim(
                    sample,
                    n_fft=self.config.n_fft,
                    hop_length=self.config.hop_length,
                    win_length=self.config.win_length,
                    n_iter=128,
                )

            result = torch.tensor(conversion(outputs), dtype=torch.float32)

            return result

    def __init__(self, config):
        super().__init__(config)

        self.config = config

        self.converter = STFTDataset.Converter(self.config)

        self.x = self.converter.encode(self.x)
        self.y = self.converter.encode(self.y)

    class Settings(WaveformDataset.Settings):
        n_fft: int
        hop_length: int
        win_length: int

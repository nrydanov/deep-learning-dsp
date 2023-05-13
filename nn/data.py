import librosa
import numpy as np
from pydantic import BaseSettings
from torch.utils.data import Dataset
import torch

from typing import Tuple, Optional


class BaseDataset(Dataset):
    def __init__(self, config: BaseSettings) -> None:
        raise NotImplementedError()

    class Settings(BaseSettings):
        in_path: str
        out_path: str
        sample_rate: int
        offset: int
        duration: Optional[int] = None

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

        self.config = config

        np.random.seed(69)
        self.x, self.y = self.__generate_samples__()

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, i) -> Tuple[torch.tensor, torch.tensor]:
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


class STFTDataset(WaveformDataset):
    def __init__(self, config):
        super().__init__(config)

        conversion = lambda sample: np.abs(
            librosa.stft(
                sample,
                n_fft=config.n_fft,
                hop_length=config.hop_length,
                win_length=config.win_length,
            )
        )

        self.x = np.array(
            list(
                map(
                    conversion,
                    self.x.reshape(5000, -1),
                )
            )
        )

        self.y = np.array(
            list(
                map(
                    conversion,
                    self.y.reshape(5000, -1),
                )
            )
        )

        self.x = self.x.swapaxes(1, 2)
        self.y = self.y.swapaxes(1, 2)

    class Settings(WaveformDataset.Settings):
        n_fft: int
        hop_length: int
        win_length: int

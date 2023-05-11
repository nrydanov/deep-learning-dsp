import librosa
import numpy as np
from pydantic import BaseSettings
from torch.utils.data import Dataset
import torch

from typing import Tuple


class BaseDataset(Dataset):
    def __init__(self, config: BaseSettings) -> None:
        raise NotImplementedError()

    class Settings(BaseSettings):
        in_path: str
        out_path: str
        sample_rate: int
        total_samples: int
        offset: int
        duration: int = None

        class Config:
            pass


class WaveformDataset(BaseDataset):
    input: str = None
    output: str = None

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

        self.x, self.y = [], []
        np.random.seed(69)
        for _ in range(config.total_samples):
            x_cur, y_cur = self.__generate_sample__()

            self.x.append(x_cur)
            self.y.append(y_cur)

        self.x = np.array(self.x, dtype=np.float32)
        self.y = np.array(self.y, dtype=np.float32)
        
    def decode(x, y) -> Tuple[torch.tensor, torch.tensor]:
        return x, y

    def __len__(self) -> int:
        return self.config.total_samples

    def __getitem__(self, i) -> Tuple[torch.tensor, torch.tensor]:
        return self.x[i], self.y[i]

    def __generate_sample__(self) -> Tuple[torch.tensor, torch.tensor]:
        time = np.random.randint(self.input.shape[0] - self.config.sample_rate // 2)

        x_cur = self.input[time : time + self.config.sample_rate // 2]
        y_cur = self.output[time : time + self.config.sample_rate // 2]

        return np.array(x_cur).reshape(-1, 1), np.array(y_cur).reshape(-1, 1)
    
    


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

import librosa
import torch
import numpy as np

from torch.utils.data import TensorDataset
from torchaudio.transforms import Spectrogram

from pydantic import BaseSettings


class BaseDataset(TensorDataset):
    def __init__(self, config) -> None:
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

    def __init__(self, config):
        self.input, _ = librosa.load(
            config.in_path,
            sr=config.sample_rate,
            offset=config.offset,
            duration=config.duration,
        )

        self.output, _ = librosa.load(
            config.out_path,
            sr=config.sample_rate,
            offset=config.offset,
            duration=config.duration,
        )

        self.config = config

    def __len__(self):
        return self.config.total_samples

    def __getitem__(self, _):
        return self.__generate_sample__()

    def __generate_sample__(self):
        time = np.random.randint(self.input.shape[0] - self.config.sample_rate // 2)

        x_cur = torch.tensor(self.input[time : time + self.config.sample_rate // 2])
        y_cur = torch.tensor(self.output[time : time + self.config.sample_rate // 2])

        return x_cur.unsqueeze(1), y_cur.unsqueeze(1)


class STFTDataset(WaveformDataset):
    def __init__(self, config, **kwargs):
        super().__init__(config)

        self.kwargs = kwargs

    def __generate_sample__(self):
        x_cur, y_cur = super().__generate_sample__()

        x_cur = (Spectrogram(**self.kwargs)(x_cur),)
        y_cur = Spectrogram(**self.kwargs)(y_cur)

        return x_cur[None, :], y_cur[None, :]

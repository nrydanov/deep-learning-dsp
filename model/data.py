import librosa
import numpy as np
from pydantic import BaseSettings
from torch.utils.data import Dataset
from torchaudio.transforms import Spectrogram


class BaseDataset(Dataset):
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

    def __len__(self):
        return self.config.total_samples

    def __getitem__(self, i):
        return self.x[i], self.y[i]

    def __generate_sample__(self):
        time = np.random.randint(self.input.shape[0] - self.config.sample_rate // 2)

        x_cur = self.input[time : time + self.config.sample_rate // 2]
        y_cur = self.output[time : time + self.config.sample_rate // 2]

        return np.array(x_cur).reshape(-1, 1), np.array(y_cur).reshape(-1, 1)


class STFTDataset(WaveformDataset):
    def __init__(self, config, **kwargs):
        super().__init__(config)

        self.kwargs = kwargs

    def __generate_sample__(self):
        x_cur, y_cur = super().__generate_sample__()

        x_cur = (Spectrogram(**self.kwargs)(x_cur),)
        y_cur = Spectrogram(**self.kwargs)(y_cur)

        return x_cur[None, :], y_cur[None, :]

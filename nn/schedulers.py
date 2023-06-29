from torch.optim.lr_scheduler import CosineAnnealingLR, ConstantLR
from pydantic import BaseSettings
from types import MethodType
import logging
import numpy as np


class DecayingCosineAnnealingLR(CosineAnnealingLR):
    def get_lr(self) -> float:
        lr = super().get_lr()
        decay_factor = np.exp([-0.05 * (self.last_epoch // (2 * self.T_max))])[0]
        lr[0] *= decay_factor
        return lr

    def __init__(self, optimizer, **settings) -> float:
        super().__init__(optimizer, **settings)

    class Settings(BaseSettings):
        T_max: int
        eta_min: float


class CosineAnnealingLR(CosineAnnealingLR):
    class Settings(BaseSettings):
        T_max: int
        eta_min: float


class ConstantLR(ConstantLR):
    class Settings(BaseSettings):
        pass


def get_scheduler(name: str):
    logging.info(f"Selecting scheduler: {name}")

    match name:
        case "dca":
            return DecayingCosineAnnealingLR
        case "constant":
            return ConstantLR
        case "ca":
            return CosineAnnealingLR

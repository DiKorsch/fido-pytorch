from dataclasses import dataclass
from fido.infill import Infill

@dataclass
class FIDOConfig:
    learning_rate: float = 1e1
    batch_size: int = 8
    iterations: int = 100
    approx_steps: int = None

    l1: float = 1e-3
    l2: float = 0.0
    tv: float = 1e-2

    reg_warmup: int = 5

    thresh: float = 0.5

    @property
    def reg_params(self) -> dict:
        return dict(l1=self.l1, l2=self.l2, tv=self.tv, warmup=self.reg_warmup)

@dataclass
class MaskConfig:
    optimized: bool = True
    mask_size: int = None
    infill_strategy: Infill = Infill.BLUR

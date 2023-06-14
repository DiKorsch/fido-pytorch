import numpy as np
import torch as th

def concrete_dropout(dropout_rate: th.Tensor, temp: float = 0.1, eps: float = np.finfo(float).eps):
    u = th.zeros_like(dropout_rate).uniform_()

    approx = (
        (dropout_rate + eps).log() -
        (1 - dropout_rate + eps).log() +
        (u + eps).log() -
        (1 - u + eps).log()

    )

    return 1 - (approx / temp).sigmoid()


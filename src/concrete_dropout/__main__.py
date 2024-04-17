import torch as th
import sys

from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from concrete_dropout.default import concrete_dropout
from concrete_dropout.improved import sigmoid_concrete_dropout

def main():
    size = 32
    X = th.zeros((size, size), requires_grad=True, dtype=th.float32)
    u = th.zeros_like(X).uniform_()

    y0 = concrete_dropout(X.sigmoid(), u=u)
    y1 = sigmoid_concrete_dropout(X, u=u)
    assert th.allclose(y0, y1)

main()

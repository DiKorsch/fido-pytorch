import numpy as np
import torch as th

EPS = np.finfo(float).eps
class SigmoidConcreteDropout(th.autograd.Function):

    @staticmethod
    def forward(ctx, x, temp: float = 1.0, u = None):
        global EPS
        temp = th.scalar_tensor(temp)

        if u is None:
            u = th.zeros_like(x).uniform_()

        noise = ((u + EPS) / (1 - u  + EPS)).log()
        xt_exp = ((x + noise) / temp).exp()

        ctx.save_for_backward(xt_exp, temp)
        return 1 / (xt_exp + 1)

    @staticmethod
    def backward(ctx, grad_output):
        xt_exp, temp = ctx.saved_tensors
        grad = -(xt_exp) / (xt_exp + 1).pow(2) / temp
        return grad * grad_output, None, None

    @classmethod
    def _check_grad(clf, temp: float = 1.0, size: int = 16, *, dtype = th.float64):
        X = th.zeros((size, size), requires_grad=True, dtype=dtype)
        u = th.zeros_like(X).uniform_()
        return th.autograd.gradcheck(sigmoid_concrete_dropout, (X, temp, u))

def sigmoid_concrete_dropout(X, temp: float = 1.0, u = None):
    return SigmoidConcreteDropout.apply(X, temp, u)

if __name__ == '__main__':
    from tqdm.auto import tqdm
    print("Checking gradients")
    for temp in tqdm(np.logspace(-2, 2, 10)):
        assert SigmoidConcreteDropout._check_grad(temp=temp)
    print("Gradient checks passed")

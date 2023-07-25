import numpy as np
import torch as th

EPS = 1e-7
class SigmoidConcreteDropout(th.autograd.Function):

    @staticmethod
    def forward(ctx, logit_p, temp: float = 1.0, u = None):
        """
            our proposed simplification
        """
        global EPS
        temp = th.scalar_tensor(temp)

        if u is None:
            u = th.zeros_like(logit_p).uniform_()

        noise = ((u + EPS) / (1 - u  + EPS)).log()
        logit_p_temp = (logit_p + noise) / temp
        res = logit_p_temp.sigmoid()

        ctx.save_for_backward(res, logit_p_temp, temp)
        return res

    @staticmethod
    def backward(ctx, grad_output):
        """
            Gradient of random_tensor w.r.t logit_p is simply
            1/temp * sigmoid(logit_p_temp)^2 * exp(-logit_p_temp)
        """
        res, logit_p_temp, temp = ctx.saved_tensors
        grad = th.zeros_like(res)
        mask = res != 0
        grad[mask] = res[mask]**2 * (-logit_p_temp[mask]).exp() / temp

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

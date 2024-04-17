import numpy as np
import torch as th

class SigmoidConcreteDropout(th.autograd.Function):

    @staticmethod
    def forward(ctx, logit_p, temp: float = 1.0, u = None, *, eps = np.finfo(float).eps):
        """
            our proposed simplification
        """
        temp = th.scalar_tensor(temp)

        if u is None:
            u = th.zeros_like(logit_p).uniform_()

        noise = ((u + eps) / (1 - u  + eps)).log()
        logit_p_temp = (logit_p + noise) / temp
        keep_rate = logit_p_temp.sigmoid()

        ctx.save_for_backward(keep_rate, logit_p_temp, temp)
        return 1 - keep_rate

    @staticmethod
    def backward(ctx, grad_output):
        """
            Gradient of random_tensor w.r.t logit_p is simply
            1/temp * sigmoid(logit_p_temp)^2 * exp(-logit_p_temp)
        """
        keep_rate, logit_p_temp, temp = ctx.saved_tensors
        grad = th.zeros_like(keep_rate)
        mask = keep_rate != 0
        grad[mask] = keep_rate[mask]**2 * (-logit_p_temp[mask]).exp() / temp

        return -grad * grad_output, None, None

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

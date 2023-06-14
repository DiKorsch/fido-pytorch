import numpy as np
import torch as th
import torch.nn.functional as F
import torchmetrics as tm

from matplotlib import pyplot as plt
from pathlib import Path

from concrete_dropout import concrete_dropout
from concrete_dropout import sigmoid_concrete_dropout
from fido.configs import FIDOConfig
from fido.configs import MaskConfig
from fido.infill import new_infill
from fido.metrics import Metrics

def reg_scale(i, max_value):
    if i >= max_value:
        return 1.0
    return i / max_value

class FIDO(th.nn.Module):

    @classmethod
    def new(cls, im, params: MaskConfig, *, device):

        # de-normalize first
        infill = new_infill(im * 0.5 + 0.5, params.infill_strategy)

        C, H, W = im.shape
        size = (params.mask_size, params.mask_size)
        if params.mask_size is None:
            size = (H, W)

        fido = cls(size, infill=infill.unsqueeze(0), device=device, optimized=params.optimized)

        return fido

    def __init__(self, size: tuple, *,
        infill: th.Tensor,
        optimized: bool = True,
        device = None
    ):
        super().__init__()

        self.ssr_logit_p = th.zeros(size, device=device, requires_grad=True)
        self.sdr_logit_p = th.zeros(size, device=device, requires_grad=True)

        self._optimized = optimized
        self._infill = infill


    @property
    def params(self):
        return [self.ssr_logit_p, self.sdr_logit_p]

    @property
    def sdr_dropout_rate(self):
        return self.sdr_logit_p.sigmoid()

    @property
    def ssr_dropout_rate(self):
        return self.ssr_logit_p.sigmoid()

    @property
    def joint_dropout_rate(self):
        return ((1 - self.ssr_dropout_rate) * self.sdr_dropout_rate)

    def tv_loss(self, *, reduction="mean"):
        # p = th.stack([self.ssr_logit_p, self.sdr_logit_p], axis=0)
        p = th.stack([self.ssr_dropout_rate, self.sdr_dropout_rate], axis=0)
        # p = self.joint_dropout_rate[None]
        return tm.TotalVariation(reduction=reduction)(p[None])

    def l1_norm(self):
        # return self.joint_dropout_rate.sum()
        ssr_l1 = (1 - self.ssr_dropout_rate).sum()
        sdr_l1 = self.sdr_dropout_rate.sum()
        return (ssr_l1 + sdr_l1) / 2

    def _keep_rate(self, logit_p, *, batch_size: int = 1, deterministic: bool = False):

        if deterministic:
            return th.stack([1 - logit_p.sigmoid()], axis=0)
        else:
            # bernouli sampling
            if self._optimized:
                return th.stack([sigmoid_concrete_dropout(logit_p) for _ in range(batch_size)], axis=0)
            else:
                return th.stack([concrete_dropout(logit_p.sigmoid()) for _ in range(batch_size)], axis=0)

    def _upsample(self, X, mask, *, upsample_mode: str = "bilinear"):
        *_, H, W = X.shape
        n, h, w = mask.shape

        if (h, w) == (H, W):
            return mask

        return F.interpolate(mask.unsqueeze(1), (H, W), mode=upsample_mode)[:, 0]


    def _blend(self, X, logit_p, **kwargs):
        keep_rate = self._keep_rate(logit_p, **kwargs)

        keep_rate = self._upsample(X, keep_rate)

        keep_rate = keep_rate.unsqueeze(1)
        X = X.unsqueeze(0)

        return keep_rate * X + (1 - keep_rate) * self._infill

    def ssr(self, X, **kwargs):
        return self._blend(X, self.ssr_logit_p, **kwargs)

    def sdr(self, X, **kwargs):
        return self._blend(X, self.sdr_logit_p, **kwargs)

    def forward(self, X, **kwargs):

        ssr = self.ssr(X, **kwargs)
        sdr = self.sdr(X, **kwargs)
        return ssr, sdr

    def objective(self, X, y, clf, *, batch_size: int, deterministic: bool = False):
        ssr, sdr = self(X, batch_size=batch_size, deterministic=deterministic)
        n = len(ssr)
        _x = th.concatenate([ssr, sdr])
        prob, odds = clf.log_odds(_x, c=y)
        ssr_prob, sdr_prob = prob[:n], prob[n:]
        ssr_odds, sdr_odds = odds[:n], odds[n:]
        # sdr - ssr <== minimizing sdr and maximizing ssr probabilities
        loss = (sdr_odds - ssr_odds).mean()
        return (ssr, sdr), (ssr_prob.mean(), sdr_prob.mean()), loss, self.l1_norm(), self.tv_loss()

    def fit(self, im, y, clf, *, config: FIDOConfig, metrics: Metrics = None, update_callback = None):

        opt = th.optim.AdamW(self.params, lr=config.learning_rate, eps=0.1, weight_decay=config.l2)
        opt.zero_grad()

        ssr_grad, sdr_grad = 1, 1
        for i in range(config.iterations):

            if config.approx_steps is None:
                # no approximation
                masks, probs, loss, l1_norm, tvl = self.objective(im, y, clf, batch_size=config.batch_size)

            else:
                if i % config.approx_steps == 0:
                    (ssr, sdr), probs, loss, l1_norm, tvl = self.objective(im, y, clf, batch_size=config.batch_size)
                    ssr.retain_grad()
                    sdr.retain_grad()

                else:
                    ssr, sdr = self(im, batch_size=config.batch_size)
                    order = th.randperm(len(ssr))

                    loss = ((ssr * ssr_grad.clone()[order]) + (sdr * sdr_grad.clone()[order])).sum()
                    l1_norm = self.l1_norm()
                    tvl = self.tv_loss()


            (
                # log-odds loss
                loss +
                # regularization
                reg_scale(i, config.reg_warmup) * (config.l1 * l1_norm + config.tv * tvl)
            ).backward()


            if config.approx_steps is not None and i % config.approx_steps == 0:
                ssr_grad = ssr.grad.detach()
                sdr_grad = sdr.grad.detach()

            if metrics is not None:
                metrics.append(
                    ssr=probs[0], sdr=probs[1],
                    loss=loss, l1loss=l1_norm, tvloss=tvl,
                )

            if update_callback is not None:
                update_callback(i+1)

            opt.step()
            opt.zero_grad()
        update_callback(i, is_last_step=True)


    def plot(self, im, pred, clf, *, output=None, metrics: dict = None):
        ssr_keep_rate = self._upsample(im, self._keep_rate(self.ssr_logit_p, deterministic=True))
        ssr_keep_rate = ssr_keep_rate.detach().cpu().squeeze(0).numpy()
        # ssr_bernouli = self._upsample(im, self._keep_rate(self.ssr_logit_p, deterministic=False))

        sdr_keep_rate = self._upsample(im, self._keep_rate(self.sdr_logit_p, deterministic=True))
        sdr_keep_rate = sdr_keep_rate.detach().cpu().squeeze(0).numpy()
        # sdr_bernouli = self._upsample(im, self._keep_rate(self.sdr_logit_p, deterministic=False))

        joint_keep_rate = np.sqrt(ssr_keep_rate * (1-sdr_keep_rate))

        cls_id = pred.argmax()
        orig_im = (im * 0.5 + 0.5).permute(1, 2, 0).numpy()

        filled = self(im, deterministic=True)
        filled = th.concatenate(filled, axis=0)

        filled_pred, *_ = clf.predict(filled)
        filled_pred = filled_pred.softmax(axis=1)
        ssr_prob, sdr_prob = filled_pred[:, cls_id]

        ssr_filled, sdr_filled = (filled * 0.5 + 0.5).permute(0, 2, 3, 1).detach().numpy()

        fig, axs = plt.subplots(2, 3, figsize=(16,9), squeeze=False)

        ims = [
            (
                f"Predicted class #{cls_id}: {pred[cls_id]:.3%}",
                orig_im
            ),
            (
                f"[SSR] Predicted class #{cls_id}: {ssr_prob:.3%}",
                ssr_filled
            ),
            (
                f"[SDR] Predicted class #{cls_id}: {sdr_prob:.3%}",
                sdr_filled
            ),
            (
                "[$SSR \\times (1 - SDR)$ mask]",
                [joint_keep_rate, orig_im]
            ),
            (
                f"[SSR mask] min: {float(ssr_keep_rate.min()):.3f} | max: {float(ssr_keep_rate.max()):.3f}",
                ssr_keep_rate
            ),
            (
                f"[1-SDR mask] min: {float(sdr_keep_rate.min()):.3f} | max: {float(sdr_keep_rate.max()):.3f}",
                1-sdr_keep_rate
            ),
        ]

        for i, (title, _im) in enumerate(ims):
            ax = axs[np.unravel_index(i, axs.shape)]

            ax.axis("off")
            if _im is None:
                continue

            if isinstance(_im, list):
                for i in _im:
                    alpha = 0.3 if i.ndim == 3 else 1.0
                    ax.imshow(i, alpha=alpha)
            else:
                ax.imshow(_im)
            ax.set_title(title)

        plt.tight_layout()
        if output is None:
            plt.show()
        else:
            plt.savefig(output)


        if metrics is not None:
            fig, axs = plt.subplots(len(metrics), 1, figsize=(16,9), squeeze=False)
            for i, (key, values) in enumerate(metrics.items()):
                ax = axs[np.unravel_index(i, axs.shape)]

                ax.set_title(key)
                ax.plot(np.arange(len(values)), values)

            plt.tight_layout()
            if output is None:
                plt.show()
            else:
                output = Path(output)
                loss_graphs = output.with_suffix(f".losses{output.suffix}")
                plt.savefig(loss_graphs)

        plt.close()

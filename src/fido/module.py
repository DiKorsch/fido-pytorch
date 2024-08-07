import numpy as np
import torch as th
import torch.nn.functional as F
import torchmetrics as tm
import typing as T

from matplotlib import pyplot as plt
from pathlib import Path

from concrete_dropout import concrete_dropout
from concrete_dropout import sigmoid_concrete_dropout
from fido.configs import FIDOConfig
from fido.configs import MaskConfig
from fido.metrics import Metrics

def reg_scale(i, max_value):
    if i >= max_value:
        return 1.0
    return i / max_value

def sigm_inv(x):
    return th.log(x / (1 - x))

class FIDO(th.nn.Module):

    @classmethod
    def new(cls, im, params: MaskConfig, *, device: th.device, init: th.Tensor = None, track_metrics: bool = False):

        # de-normalize first
        infill = params.infill_strategy.new(im * 0.5 + 0.5, device=device)

        C, H, W = im.shape
        size = (params.mask_size, params.mask_size)
        if params.mask_size is None:
            size = (H, W)

        metrics = Metrics() if track_metrics else None

        fido = cls(size,
                   infill=infill.unsqueeze(0),
                   optimized=params.optimized,
                   metrics=metrics,
                   init=init,
                )

        return fido.to(device)

    def __init__(self, size: tuple, *,
        infill: th.Tensor,
        init: T.Optional[th.Tensor] = None,
        optimized: bool = True,
        metrics: Metrics = None
    ):
        super().__init__()
        if init is None:
            self.ssr_logit_p = th.nn.Parameter(th.zeros(size, requires_grad=True))
            self.sdr_logit_p = th.nn.Parameter(th.zeros(size, requires_grad=True))
        else:
            assert init.shape == size, f"Expected shape {size} but got {init.shape}"
            ssr_init = sigm_inv(1-init) # init = 1 - ssr_init.sigmoid()
            sdr_init = sigm_inv(init) # init = sdr_init.sigmoid()

            self.ssr_logit_p = th.nn.Parameter(ssr_init, requires_grad=True)
            self.sdr_logit_p = th.nn.Parameter(sdr_init, requires_grad=True)

        self.register_buffer("optimized", th.as_tensor(optimized))
        self.register_buffer("infill", infill)
        self.metrics = metrics

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
        return tm.image.TotalVariation(reduction=reduction).to(p.device)(p[None])

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
            if self.optimized:
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

        return keep_rate * X + (1 - keep_rate) * self.infill

    def ssr(self, X, **kwargs):
        return self._blend(X, self.ssr_logit_p, **kwargs)

    def sdr(self, X, **kwargs):
        return self._blend(X, self.sdr_logit_p, **kwargs)

    def forward(self, X, **kwargs):

        ssr = self.ssr(X, **kwargs)
        sdr = self.sdr(X, **kwargs)
        return ssr, sdr

    def objective(self, X, y, clf, *, batch_size: int, deterministic: bool = False, add_deterministic: bool = False):
        ssr, sdr = self(X, batch_size=batch_size, deterministic=deterministic)
        n = len(ssr)
        _x = th.concatenate([ssr, sdr])
        if hasattr(clf, "log_odds"):
            prob, odds = clf.log_odds(_x, c=y)
        else:
            logits = clf(_x)
            prob, odds = log_odds(logits, c=y)

        ssr_prob, sdr_prob = prob[:n], prob[n:]
        ssr_odds, sdr_odds = odds[:n], odds[n:]
        # sdr - ssr <== minimizing sdr and maximizing ssr probabilities
        loss = (sdr_odds - ssr_odds).mean()

        if not deterministic and add_deterministic:
            _, _, det_loss, _, _ = self.objective(X, y, clf, batch_size=1, deterministic=True, add_deterministic=False)
            loss += det_loss

        return (ssr, sdr), (ssr_prob.mean(), sdr_prob.mean()), loss, self.l1_norm(), self.tv_loss()

    def fit(self, im, y, clf, *, config: FIDOConfig, metrics: Metrics = None, update_callback = None):

        opt = th.optim.AdamW(self.parameters(), lr=config.learning_rate, eps=0.1, weight_decay=config.l2)
        # opt = th.optim.SGD(self.params, lr=config.learning_rate, momentum=0.9, weight_decay=config.l2)
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
        if update_callback is not None:
            update_callback(i, is_last_step=True)


    def plot(self, im, pred, clf, *, output=None, thresh: float = None):
        ssr_keep_rate = self._upsample(im, self._keep_rate(self.ssr_logit_p, deterministic=True))
        ssr_keep_rate = ssr_keep_rate.detach().cpu().squeeze(0).numpy()
        # ssr_bernouli = self._upsample(im, self._keep_rate(self.ssr_logit_p, deterministic=False))

        sdr_keep_rate = self._upsample(im, self._keep_rate(self.sdr_logit_p, deterministic=True))
        sdr_keep_rate = sdr_keep_rate.detach().cpu().squeeze(0).numpy()
        # sdr_bernouli = self._upsample(im, self._keep_rate(self.sdr_logit_p, deterministic=False))

        joint_keep_rate = np.sqrt(ssr_keep_rate * (1-sdr_keep_rate))

        if thresh is not None:
            joint_keep_rate[joint_keep_rate < thresh] = np.nan

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
                f"[1-SDR mask] min: {float((1-sdr_keep_rate).min()):.3f} | max: {float((1-sdr_keep_rate).max()):.3f}",
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
        plt.close(fig=fig)

        if self.metrics is not None:
            metrics = self.metrics.as_dict()
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

            plt.close(fig=fig)


def log_odds(logits, c):
    # normalized log-probabilities
    log_probs = F.log_softmax(logits, dim=1)
    mask = th.ones_like(log_probs[:1])
    mask[:, c] = 0
    odds = log_probs[:, c] - th.logsumexp(log_probs * mask, dim=1)
    return log_probs[:, c].exp(), odds

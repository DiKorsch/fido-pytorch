import cv2
import numpy as np

from dataclasses import dataclass
from dataclasses import field
from scipy.ndimage import gaussian_filter


@dataclass
class Metrics:
    ssr_probs: list[float] = field(default_factory=list)
    sdr_probs: list[float] = field(default_factory=list)
    losses: list[float] = field(default_factory=list)
    l1_losses: list[float] = field(default_factory=list)
    tv_losses: list[float] = field(default_factory=list)

    def as_dict(self):
        return {
            "SSR prob": list(self.ssr_probs),
            "SDR prob": list(self.sdr_probs),
            "loss": list(self.losses),
            "l1 norm": list(self.l1_losses),
            "total variation": list(self.tv_losses),
        }

    def append(self, *, ssr, sdr, loss, l1loss, tvloss):
        self.ssr_probs.append(float(ssr))
        self.sdr_probs.append(float(sdr))
        self.losses.append(float(loss))
        self.l1_losses.append(float(l1loss))
        self.tv_losses.append(float(tvloss))

    def reset(self):
        for values in [self.ssr_probs, self.sdr_probs, self.losses, self.l1_losses, self.tv_losses]:
            values.clear()

def _thresh(sal, thresh: float, *, normalize: bool = False, sigma: float = None,
    supress_value: float = None):

    sal = sal.copy()

    if sigma is not None and sigma >= 1:
        sal = gaussian_filter(sal, sigma=sigma)

    if normalize:
        sal -= sal.min()
        max_val = sal.max()
        if max_val != 0:
            sal /= max_val

    if thresh == "mean" or not (sal > thresh).any():
        thresh = sal.mean()

    sal[sal < thresh] = np.nan if supress_value is None else supress_value

    return sal


def _bbox_from_mask(mask, *, rotated: bool = False):

    cont, *_ = cv2.findContours((mask * 255).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pts = np.concatenate(cont)
    hull = cv2.convexHull(pts)

    if rotated:
        return cv2.minAreaRect(hull)

    x0, y0 = pts.min(axis=0)[0]
    x1, y1 = pts.max(axis=0)[0]


    center = cx, cy = (x1 + x0) // 2, (y1 + y0) // 2
    size = sx, sy = max((x1 - x0), 32), max((y1 - y0), 32)

    return center, size, 0


def iou(sal, bbox, thresh: float, *, normalize: bool = False):

    sal = _thresh(sal, thresh, normalize=normalize, sigma=3.0)
    if bbox is None:
        return 0, sal

    (cx, cy), (sx, sy), angle = _bbox_from_mask(np.isfinite(sal))
    assert angle == 0

    x0, y0 = cx - sx//2, cy - sy//2

    pred_arr = np.zeros(sal.shape, dtype=bool)
    pred_arr[y0:y0+sy, x0:x0+sx] = True

    gt_arr = np.zeros(sal.shape, dtype=bool)
    x0, y0, w, h = bbox
    gt_arr[y0:y0+h, x0:x0+w] = True

    union = np.logical_or(gt_arr, pred_arr)
    intersect = np.logical_and(gt_arr, pred_arr)

    return intersect.sum() / union.sum(), sal


def fdr(sal, bbox, thresh: float, *, normalize: bool = False):
    """ return a percentage of saliency pixels outsite of the bounding box """
    #assert bbox is not None, "For the evaluation, bounding box annotations are required!"

    sal = _thresh(sal, thresh, normalize=normalize)

    if bbox is None:
        return 0, sal

    n_overall = np.isfinite(sal).sum()
    if n_overall == 0:
        return 0, sal

    H, W, *_ = sal.shape
    x0, y0, w, h = bbox
    non_bbox_area = max(0, H * W - (h * w))
    if non_bbox_area == 0:
        return 0, sal

    _sal = sal.copy()
    _sal[y0:y0+h, x0:x0+w] = np.nan
    n_non_bbox = np.isfinite(_sal).sum()

    return n_non_bbox / non_bbox_area, sal

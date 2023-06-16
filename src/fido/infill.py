import cv2
import numpy as np
import torch as th

from numpy.lib.stride_tricks import sliding_window_view
from skimage.measure import regionprops
from torchvision.transforms import functional as tr

from fido.metrics import _thresh


def patches(im, size):
    window_shape = (size, size, im.shape[-1])
    return sliding_window_view(im, window_shape)[::size, ::size]

def new_infill(im: th.Tensor, strategy: str, *, mean: float = 0.5, std: float = 0.5, size: int = 9, device=None):

    if strategy == "original":
        infill = im.clone()

    elif strategy == "zeros":
        infill = th.zeros_like(im, device=device)

    elif strategy == "uniform":
        infill = th.zeros_like(im, device=device).uniform_(0, 1)

    elif strategy == "normal":
        infill = th.zeros_like(im, device=device).normal_(std=0.2)

    elif strategy == "mean":
        mean_pixel = im.mean(axis=(1,2), keepdim=True)
        infill = mean_pixel.expand(im.shape)

    elif strategy == "blur":
        # should result in std=10, same as in the paper
        # de-normalize first
        infill = tr.gaussian_blur(im.to(device), kernel_size=75).to(im.device)

    elif strategy == "local":
        pass

    elif strategy == "knockoff":
        infill = np.zeros_like(im)
        windows = patches(im.detach().cpu().numpy().transpose(1,2,0), size=size)
        rows, cols, *_ = windows.shape
        for i in range(rows*cols):
            row, col, _chan = np.unravel_index(i, (rows, cols, 1))
            patch = windows[row, col, _chan]

            x0, y0 = col*size, row*size
            x1, y1 = (col+1)*size, (row+1)*size

            idxs = np.random.permutation(size**2)
            idxs = np.unravel_index(idxs, (size, size))

            infill[:, y0:y1, x0:x1] = patch[idxs].reshape(patch.shape).transpose(2,0,1)

        infill = th.tensor(infill)
    else:
        raise ValueError(f"Unknown in-fill strategy: {strategy}")

    return (infill - mean) / std




def _calc_bbox(mask, min_size, *, pad: int = 10, squared: bool = True):
    props = regionprops(mask)
    for prop in props:
        if prop.label == 1:
            y0, x0, y1, x1 = prop.bbox
            break

    w, h = max(x1-x0+pad, min_size), max(y1-y0+pad, min_size)
    cx, cy = (x1+x0)/2, (y1+y0)/2

    if squared:
        w = h = max(w, h)

    x0, y0 = max(cx - w//2, 0), max(cy - h//2, 0)

    return int(x0), int(y0), int(w), int(h)

def enhance(im: np.ndarray, ssr: np.ndarray, sdr: np.ndarray, *,
    infill_strategy: str = "blur",
    mask_to_use: str = "joint",
    cropped: bool = True,
    threshold: float = 0.5,
    sigma: float = 3.0,

    device = None,
    ):

    assert mask_to_use in ["ssr", "sdr", "joint"]
    assert infill_strategy == "blur", \
        "only gaussian_blur infill is currently supported!"

    if mask_to_use == "ssr":
        sal = ssr.clone()
    elif mask_to_use == "sdr":
        sal = sdr.clone()
    else:
        sal = np.sqrt(ssr * sdr)

    *size, c = im.shape
    sal = cv2.resize(sal, size)
    infill = cv2.GaussianBlur(im, (0, 0), sigmaX=10)

    A = im.astype(np.float32)
    B = infill.astype(np.float32)
    alpha = sal[:, :, None]
    enhanced = (A * alpha + B * (1-alpha)).astype(im.dtype)

    if not cropped:
        return enhanced, sal

    mask = _thresh(sal, threshold, sigma=sigma, supress_value=0.0)
    x0, y0, w, h = _calc_bbox((mask != 0.0).astype(np.int32), min_size=64)

    return enhanced[y0:y0+h, x0:x0+w], sal

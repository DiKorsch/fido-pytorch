import torch as th
import numpy as np

from numpy.lib.stride_tricks import sliding_window_view
from torchvision.transforms import functional as tr


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
        infill = tr.gaussian_blur(im, kernel_size=75)

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


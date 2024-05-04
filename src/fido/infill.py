import cv2
import enum
import numpy as np
import torch as th

from functools import partial
from numpy.lib.stride_tricks import sliding_window_view
from skimage.measure import regionprops
from torchvision.transforms import functional as tr
from pathlib import Path
from fido.metrics import _thresh


def patches(im, size):
    window_shape = (size, size, im.shape[-1])
    return sliding_window_view(im, window_shape)[::size, ::size]

gan = None

class Infill(enum.Enum):
    ORIGINIAL = enum.auto()
    ZEROS = enum.auto()
    UNIFORM = enum.auto()
    NORMAL = enum.auto()
    MEAN = enum.auto()
    BLUR = enum.auto()
    GAN = enum.auto()
    KNOCKOFF = enum.auto()

    def normalize(self, arr: th.Tensor, mean: float, std: float) -> th.Tensor:
        return (arr - mean) / std

    def new(self, im: th.Tensor,  *, mean: float = 0.5, std: float = 0.5, size: int = 9, device: th.device = None):
        global gan
        _normalize = partial(self.normalize, mean=mean, std=std)
        if self == Infill.BLUR:
            # should result in std=10, same as in the paper
            # de-normalize first
            return _normalize(tr.gaussian_blur(im.to(device), kernel_size=75).to(im.device))

        if self == Infill.ORIGINIAL:
            return _normalize(im.clone())

        if self == Infill.ZEROS:
            return _normalize(th.zeros_like(im, device=device))

        if self == Infill.UNIFORM:
            return _normalize(th.zeros_like(im, device=device).uniform_(0, 1))

        if self == Infill.NORMAL:
            return _normalize(th.zeros_like(im, device=device).normal_(std=0.2))

        if self == Infill.MEAN:
            mean_pixel = im.mean(axis=(1,2), keepdim=True)
            return _normalize(mean_pixel.expand(im.shape))

        if self == Infill.KNOCKOFF:

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

            return _normalize(th.tensor(infill))

        if self == Infill.GAN:
            if gan is None:
                gan = GAN(device=im.device)
            infill = gan(im)
            return infill

        raise NotImplementedError(f"Strategy is not implemented yet: {self}!")

import dmfn  # noqa: E402
from dmfn.models.networks import define_G  # noqa: E402

class GAN:

    DEFAULT_WEIGHTS: Path = Path(dmfn.__file__).resolve().parent.parent.parent / "outputs/cub/checkpoints/latest_G.pth"

    @staticmethod
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            th.nn.init.orthogonal_(m.weight.data, 1.0)
        elif classname.find('BatchNorm') != -1:
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)
        elif classname.find('Linear') != -1:
            th.nn.init.orthogonal_(m.weight.data, 1.0)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def __init__(self, weights: Path = None, device: th.device = th.device("cpu")):

        if weights is None:
            weights = GAN.DEFAULT_WEIGHTS
        assert Path(weights).exists(), \
            f"Could not find weights: {weights}"

        opt = dict(network_G=dict( which_model_G='DMFN', in_nc=4, out_nc=3, nf=64, n_res=8),is_train=False)
        self.generator = define_G(opt).to(device)
        self.generator.load_state_dict(th.load(weights), strict=True)
        self.generator.eval()

    def __call__(self, im: th.Tensor, *, grid_size: int = 16, size: tuple = (256, 256)):
        mask = th.randint(0, 2, size=(1, 1, grid_size, grid_size), dtype=im.dtype, device=im.device)
        orig_size = im.size()[-2:]
        im = tr.resize(im, size)
        mask = tr.resize(mask, size, interpolation=tr.InterpolationMode.NEAREST)

        X1 = th.cat([im * mask, 1-mask], dim=1)
        X2 = th.cat([im * (1-mask), mask], dim=1)

        X = th.cat([X1, X2], dim=0)
        out = self.generator(X).detach()
        # combine the result from the generated images
        res = out[0] * (1 - mask[0]) + out[1] * mask[0]
        return tr.resize(res, orig_size)

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
    assert infill_strategy in ["blur", "original"], \
        "only 'gaussian_blur' or 'original' infills are currently supported!"

    if mask_to_use == "ssr":
        sal = ssr.clone()
    elif mask_to_use == "sdr":
        sal = sdr.clone()
    else:
        sal = np.sqrt(ssr * sdr)

    *size, c = im.shape
    sal = cv2.resize(sal, size)
    if infill_strategy == "blur":
        infill = cv2.GaussianBlur(im, (0, 0), sigmaX=10)
    else:
        infill = im.copy()

    A = im.astype(np.float32)
    B = infill.astype(np.float32)
    alpha = sal[:, :, None]
    enhanced = (A * alpha + B * (1-alpha)).astype(im.dtype)

    if not cropped:
        return enhanced, sal

    mask = _thresh(sal, threshold, sigma=sigma, supress_value=0.0)
    x0, y0, w, h = _calc_bbox((mask != 0.0).astype(np.int32), min_size=64)

    return enhanced[y0:y0+h, x0:x0+w], sal

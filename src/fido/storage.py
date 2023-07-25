from __future__ import annotations

import logging
import numpy as np

from pathlib import Path
from dataclasses import dataclass

@dataclass
class Entry:
    array: np.ndarray
    min_value: float
    max_value: float

    @classmethod
    def new(cls, arr: np.ndarray) -> Entry:
        minv, maxv = arr.min(), arr.max()

        arr = (arr - minv) / (maxv - minv)
        arr = (arr * 255).astype(np.uint8)

        return cls(arr, minv, maxv)

    def unpack(self, *, dtype=np.float32):
        arr = self.array.astype(dtype) / 255.0
        minv, maxv = self.min_value, self.max_value
        return arr * (maxv - minv) + minv

class LazyLoadMaps:

    def __init__(self, maps_file, *, map_type: str):
        super().__init__()
        self.maps_file = maps_file
        self._maptype = map_type

        self._cache = {}
        self._npzfile = None

    @property
    def npzfile(self):
        if self._npzfile is None:
            self._npzfile = np.load(self.maps_file)

        return self._npzfile

    def __getitem__(self, i):
        key = f"{self._maptype}/{i}"
        if key not in self._cache:
            self._cache[key] = Entry.new(self.npzfile[key])

        return self._cache[key].unpack()

    def __len__(self) -> int:
        return len(self._npzfile.files) // 2

def _maps_file_name(subset, *, lazy_load: bool = True):
    suffix = ".lazy" if lazy_load else ""
    return f"masks.{subset}{suffix}.npz"

def load_maps(maps_folder: Path, subset: str, lazy_load: bool = True):

    maps_file = Path(maps_folder) / _maps_file_name(subset, lazy_load=lazy_load)

    maps = np.load(maps_file)
    logging.info(f"Loading {subset} maps from {maps_file} ...")
    if lazy_load:
        ssrs, sdrs = [LazyLoadMaps(maps_file, map_type=key) for key in ["ssr", "sdr"]]
        logging.info("Loaded LazyLoadMaps")

    else:
        ssrs, sdrs = [maps[key] for key in ["ssr", "sdr"]]
        logging.info(f"Maps loaded: {ssrs.shape} | {sdrs.shape}")

    return ssrs, sdrs


def dump_maps(maps_folder, subset, ssr, sdr, lazy_load: bool = True):

    maps_file = Path(maps_folder) / _maps_file_name(subset, lazy_load=lazy_load)

    if lazy_load:
        ssrs = {f"ssr/{i}": arr for i, arr in enumerate(ssr)}
        sdrs = {f"sdr/{i}": arr for i, arr in enumerate(sdr)}
        np.savez(maps_file, **ssrs, **sdrs)
    else:
        np.savez(maps_file, ssr=np.array(ssr), sdr=np.array(sdr))


def _convert_to_lazy(folder, subsets = ["train", "val"]):
    for subset in subsets:
        ssrs, sdrs = load_maps(folder, subset, lazy_load=False)
        dump_maps(folder, subset, ssrs, sdrs, lazy_load=True)

if __name__ == '__main__':
    from tqdm.auto import tqdm
    from cvargparse import Arg
    from cvargparse import BaseParser
    parser = BaseParser([Arg("dest")])
    args = parser.parse_args()

    _convert_to_lazy(args.dest, subsets=tqdm(["train", "val"]))

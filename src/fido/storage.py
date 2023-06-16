import logging
import numpy as np

from pathlib import Path

def load_maps(maps_folder: Path, subset: str):

    maps_file = Path(maps_folder) / f"masks.{subset}.npz"

    maps = np.load(maps_file)
    logging.info(f"Loading {subset} maps from {maps_file} ...")
    ssrs, sdrs = [maps[key] for key in ["ssr", "sdr"]]

    logging.info(f"Maps loaded: {ssrs.shape} | {sdrs.shape}")
    return ssrs, sdrs

def dump_maps(dest, ssr, sdr):
    np.savez(dest, ssr=np.array(ssr), sdr=np.array(sdr))

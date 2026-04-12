# Phase 1 - dataset setup, HDF5 loading, and PCamDataset

import os
import sys
import h5py
import numpy as np
import pandas as pd
import yaml
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Tuple, Optional


# Config loader

ROOT = Path(__file__).resolve().parent.parent   # d:/Aakanksha/WSI

def load_config() -> dict:
    cfg_path = ROOT / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# File verification

REQUIRED_FILES = [
    "pcam_train_x", "pcam_valid_x", "pcam_test_x",
    "train_y",      "valid_y",       "test_y",
    "train_mask",
    "train_meta",   "valid_meta",    "test_meta",
]

def verify_files(cfg: dict) -> bool:
    """Check all required dataset files exist. Returns True if all present."""
    paths = cfg["paths"]
    all_ok = True
    print("\n-- File Verification ---------------------------------------")
    for key in REQUIRED_FILES:
        fpath = ROOT / paths[key]
        status = "[OK]" if fpath.exists() else "[MISSING]"
        size   = f"({fpath.stat().st_size / 1e9:.2f} GB)" if fpath.exists() else ""
        print(f"  [{status}] {key:20s}  {size}")
        if not fpath.exists():
            all_ok = False
    print()
    return all_ok


# Raw data loaders (returns numpy arrays)

def load_split(cfg: dict, split: str = "train") -> Tuple[np.ndarray, np.ndarray]:
    """
    Load images and labels for a given split.

    Returns:
        X : np.ndarray  shape (N, 96, 96, 3)  uint8
        y : np.ndarray  shape (N,)             uint8  {0, 1}
    """
    assert split in ("train", "valid", "test"), f"Unknown split: {split}"
    paths = cfg["paths"]

    x_key = {"train": "pcam_train_x", "valid": "pcam_valid_x", "test": "pcam_test_x"}[split]
    y_key = {"train": "train_y",       "valid": "valid_y",       "test": "test_y"}[split]

    with h5py.File(ROOT / paths[x_key], "r") as f:
        X = f["x"][:]                       # (N, 96, 96, 3)

    with h5py.File(ROOT / paths[y_key], "r") as f:
        y = f["y"][:].squeeze().astype(np.uint8)   # (N,)

    assert len(X) == len(y), "X/y length mismatch!"
    return X, y


def load_masks(cfg: dict) -> np.ndarray:
    """
    Load pixel-level tumor masks for the training split.

    Returns:
        masks : np.ndarray  shape (262144, 96, 96, 1)  bool
    """
    path = ROOT / cfg["paths"]["train_mask"]
    with h5py.File(path, "r") as f:
        masks = f["mask"][:]
    return masks


def load_metadata(
    cfg: dict,
    split: str = "train",
    indices: Optional[np.ndarray] = None,
) -> pd.DataFrame:
    """
    Load WSI provenance metadata.

    Columns: coord_y, coord_x, tumor_patch, center_tumor_patch, wsi
    """
    key = {"train": "train_meta", "valid": "valid_meta", "test": "test_meta"}[split]
    df = pd.read_csv(ROOT / cfg["paths"][key], index_col=0)
    if indices is not None:
        idx = np.asarray(indices, dtype=np.int64)
        df = df.iloc[idx].reset_index(drop=True)
    return df


# PyTorch Dataset

class PCamDataset(Dataset):
    """
    Lazy-loading PCam dataset backed by HDF5 files.

    Args:
        split       : "train" | "valid" | "test"
        transform   : albumentations transform (optional)
        cfg         : config dict (if None, loaded from config.yaml)
        indices     : optional subset indices (for train/val fold splitting)
        return_mask : if True, also returns the pixel mask (train only)
    """

    def __init__(
        self,
        split: str = "train",
        transform=None,
        cfg: Optional[dict] = None,
        indices: Optional[np.ndarray] = None,
        return_mask: bool = False,
    ):
        self.cfg        = cfg or load_config()
        self.split      = split
        self.transform  = transform
        self.return_mask = return_mask and split == "train"

        paths = self.cfg["paths"]
        x_key = {"train": "pcam_train_x", "valid": "pcam_valid_x", "test": "pcam_test_x"}[split]
        y_key = {"train": "train_y",       "valid": "valid_y",       "test": "test_y"}[split]

        self.x_path = str(ROOT / paths[x_key])
        self.y_path = str(ROOT / paths[y_key])
        self.m_path = str(ROOT / paths["train_mask"]) if self.return_mask else None

        # Load labels fully (small -- 262K × 1 float = ~2 MB)
        with h5py.File(self.y_path, "r") as f:
            self._labels = f["y"][:].squeeze().astype(np.float32)   # (N,)

        if indices is not None:
            self.indices = indices
        else:
            all_indices = np.arange(len(self._labels))
            # Apply data limits from config if set
            data_cfg  = self.cfg.get("data", {})
            limit_key = {"train": "max_train_samples",
                         "valid": "max_valid_samples",
                         "test" : "max_test_samples"}[split]
            max_n = data_cfg.get(limit_key, None)
            if max_n and max_n < len(self._labels):
                # Balanced subsample: equal tumor/normal
                rng       = np.random.default_rng(42)
                half      = max_n // 2
                tumor_idx  = all_indices[self._labels == 1]
                normal_idx = all_indices[self._labels == 0]
                tumor_sel  = rng.choice(tumor_idx,  min(half, len(tumor_idx)),  replace=False)
                normal_sel = rng.choice(normal_idx, min(half, len(normal_idx)), replace=False)
                self.indices = np.sort(np.concatenate([tumor_sel, normal_sel]))
            else:
                self.indices = all_indices

        # Open HDF5 file handles lazily (set in __getitem__ on first call)
        self._x_file = None
        self._m_file = None

    def __len__(self) -> int:
        return len(self.indices)

    def _open_files(self):
        """Open HDF5 file handles (one per worker process)."""
        if self._x_file is None:
            self._x_file = h5py.File(self.x_path, "r")
        if self.return_mask and self._m_file is None:
            self._m_file = h5py.File(self.m_path, "r")

    def __getitem__(self, idx: int):
        self._open_files()
        real_idx = int(self.indices[idx])

        image = self._x_file["x"][real_idx]          # (96, 96, 3)  uint8
        label = self._labels[real_idx]               # float32  {0.0, 1.0}

        if self.transform is not None:
            augmented = self.transform(image=image)
            image = augmented["image"]               # tensor (3, H, W) after ToTensorV2
        else:
            # Default: HWC uint8 → CHW float32 in [0, 1]
            image = torch.from_numpy(image.transpose(2, 0, 1)).float() / 255.0

        label = torch.tensor(label, dtype=torch.float32)

        if self.return_mask:
            mask = self._m_file["mask"][real_idx]    # (96, 96, 1) bool
            mask = torch.from_numpy(mask.squeeze()).float()
            return image, label, mask

        return image, label

    def __del__(self):
        if self._x_file is not None:
            try:
                self._x_file.close()
            except Exception:
                pass
        if self._m_file is not None:
            try:
                self._m_file.close()
            except Exception:
                pass


# Dataset summary

def print_dataset_summary(cfg: dict):
    print("=" * 60)
    print("  PCam / Camelyon16 Dataset Summary")
    print("=" * 60)

    for split in ("train", "valid", "test"):
        X, y = load_split(cfg, split)
        n_tumor  = int(y.sum())
        n_normal = len(y) - n_tumor
        print(f"\n  [{split.upper()}]")
        print(f"    Images : {X.shape}  dtype={X.dtype}")
        print(f"    Labels : {y.shape}  dtype={y.dtype}")
        print(f"    Tumor  : {n_tumor:,}  ({100*n_tumor/len(y):.1f}%)")
        print(f"    Normal : {n_normal:,}  ({100*n_normal/len(y):.1f}%)")

    # Masks
    masks = load_masks(cfg)
    print(f"\n  [TRAIN MASKS]")
    print(f"    Shape  : {masks.shape}  dtype={masks.dtype}")
    print(f"    Coverage: pixel-level tumor annotations for all training patches")

    # Metadata
    meta = load_metadata(cfg, "train")
    print(f"\n  [METADATA -- TRAIN]")
    print(f"    Shape         : {meta.shape}")
    print(f"    Columns       : {meta.columns.tolist()}")
    print(f"    Unique WSIs   : {meta['wsi'].nunique()}")
    print(f"    WSI types     : tumor={meta['wsi'].str.contains('tumor').sum():,}  "
          f"normal={meta['wsi'].str.contains('normal').sum():,}")

    print()
    print("  [DONE] Dataset verified. Ready for preprocessing (Phase 2).")
    print("=" * 60)


# Entry point

if __name__ == "__main__":
    cfg = load_config()

    print("\nVerifying dataset files ...")
    ok = verify_files(cfg)
    if not ok:
        print("ERROR: One or more required files are missing. Check paths in config.yaml.")
        sys.exit(1)

    print_dataset_summary(cfg)

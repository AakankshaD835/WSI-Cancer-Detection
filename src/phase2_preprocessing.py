# Phase 2 - Macenko normalization, tissue detection, and augmentation

import numpy as np
import cv2
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Tuple
import albumentations as A
from albumentations.pytorch import ToTensorV2


# 1. Macenko Stain Normalization (pure numpy)

class MacenkoNormalizer:
    """
    Macenko stain normalization for H&E histopathology images.

    Reference:
        Macenko et al., "A method for normalizing histology slides for
        quantitative analysis", ISBI 2009.

    Usage:
        normalizer = MacenkoNormalizer()
        normalizer.fit(target_image)        # fit on a reference patch
        out = normalizer.transform(patch)   # normalize any patch
    """

    def __init__(self, alpha: float = 1.0, beta: float = 0.15):
        self.alpha = alpha
        self.beta  = beta
        # Default H&E stain matrix (pre-fitted on a canonical H&E slide)
        self.HERef = np.array([
            [0.5626, 0.2159],
            [0.7201, 0.8012],
            [0.4062, 0.5581],
        ], dtype=np.float64)
        self.maxCRef = np.array([1.9705, 1.0308], dtype=np.float64)

    @staticmethod
    def _rgb2od(img: np.ndarray) -> np.ndarray:
        """Convert RGB image [0,255] to optical density (OD) space."""
        img = img.astype(np.float64)
        img = np.clip(img, 1, 255)
        return -np.log(img / 255.0)

    @staticmethod
    def _od2rgb(od: np.ndarray) -> np.ndarray:
        """Convert OD back to RGB [0,255]."""
        return np.clip(255.0 * np.exp(-od), 0, 255).astype(np.uint8)

    def fit(self, target: np.ndarray):
        """
        Fit normalizer to a target/reference image.

        Args:
            target : np.ndarray  shape (H, W, 3)  uint8
        """
        OD = self._rgb2od(target)
        OD_flat = OD.reshape(-1, 3)

        # Remove near-transparent pixels
        ODhat = OD_flat[np.all(OD_flat > self.beta, axis=1)]
        if len(ODhat) < 10:
            return  # not enough tissue — keep defaults

        # SVD to get stain directions
        _, _, Vt = np.linalg.svd(ODhat, full_matrices=False)
        V = Vt[:2].T                     # (3, 2)

        # Project onto the plane spanned by first two singular vectors
        That = ODhat @ V                 # (N, 2)

        # Find angle for each projection
        phi = np.arctan2(That[:, 1], That[:, 0])
        minphi = np.percentile(phi, self.alpha)
        maxphi = np.percentile(phi, 100 - self.alpha)

        # Stain vectors
        vmin = V @ np.array([np.cos(minphi), np.sin(minphi)])
        vmax = V @ np.array([np.cos(maxphi), np.sin(maxphi)])

        # Make sure H is first column (more blue than E)
        if vmin[0] > vmax[0]:
            vmin, vmax = vmax, vmin

        self.HERef = np.stack([vmin, vmax], axis=1)

        # Solve for concentrations to get max
        Y = OD_flat.T                    # (3, N)
        C = np.linalg.lstsq(self.HERef, Y, rcond=None)[0]  # (2, N)
        self.maxCRef = np.percentile(C, 99, axis=1)

    def transform(self, img: np.ndarray) -> np.ndarray:
        """
        Normalize a single patch.

        Args:
            img : np.ndarray  shape (H, W, 3)  uint8

        Returns:
            normalized : np.ndarray  shape (H, W, 3)  uint8
        """
        h, w = img.shape[:2]
        OD = self._rgb2od(img)
        OD_flat = OD.reshape(-1, 3).T           # (3, N)

        # Deconvolve
        C = np.linalg.lstsq(self.HERef, OD_flat, rcond=None)[0]    # (2, N)

        # Normalize concentrations
        maxC = np.percentile(C, 99, axis=1, keepdims=True)
        maxC = np.where(maxC == 0, 1e-6, maxC)
        C = C / maxC * self.maxCRef[:, None]

        # Reconstruct
        OD_norm = self.HERef @ C                # (3, N)
        rgb = self._od2rgb(OD_norm.T.reshape(h, w, 3))
        return rgb


# Singleton normalizer pre-fitted with canonical H&E reference
# (fitted lazily on first use with a representative patch)
_GLOBAL_NORMALIZER: Optional[MacenkoNormalizer] = None

def get_normalizer(fit_image: Optional[np.ndarray] = None) -> MacenkoNormalizer:
    """Return the global Macenko normalizer, fitting it if an image is provided."""
    global _GLOBAL_NORMALIZER
    if _GLOBAL_NORMALIZER is None:
        _GLOBAL_NORMALIZER = MacenkoNormalizer()
    if fit_image is not None:
        _GLOBAL_NORMALIZER.fit(fit_image)
    return _GLOBAL_NORMALIZER


# 2. Tissue Detection

def is_tissue(
    patch: np.ndarray,
    sat_threshold: int = 15,
    tissue_fraction: float = 0.5,
) -> bool:
    """
    Return True if the patch contains enough tissue.

    Strategy (two independent checks, both must pass):
      1. HSV saturation: tissue has high saturation; background is nearly white.
      2. Otsu threshold on grayscale: tissue pixels are darker than background.

    Args:
        patch            : (H, W, 3) uint8 RGB
        sat_threshold    : minimum mean HSV saturation to consider as tissue
        tissue_fraction  : minimum fraction of non-background pixels

    Returns:
        bool
    """
    # HSV saturation check
    hsv = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1].astype(np.float32)
    if sat.mean() < sat_threshold:
        return False

    # Otsu threshold on grayscale
    gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    tissue_pct = binary.mean() / 255.0
    return tissue_pct >= tissue_fraction


def filter_tissue_patches(
    X: np.ndarray,
    y: np.ndarray,
    sat_threshold: int = 15,
    tissue_fraction: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Filter out background patches from arrays.

    Returns:
        X_filtered, y_filtered, kept_indices
    """
    print(f"  Tissue filtering: {len(X):,} patches …", end=" ", flush=True)
    keep = []
    for i, patch in enumerate(X):
        if is_tissue(patch, sat_threshold, tissue_fraction):
            keep.append(i)
    keep = np.array(keep, dtype=np.int64)
    print(f"kept {len(keep):,} / {len(X):,}  ({100*len(keep)/len(X):.1f}%)")
    return X[keep], y[keep], keep


# 3. Augmentation Pipelines

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


def get_train_transforms(img_size: int = 96) -> A.Compose:
    """
    Training augmentation pipeline — medically appropriate for H&E patches.

    Includes:
      - Spatial: rotations, flips (H&E is orientation-invariant)
      - Color/stain: hue/sat/brightness jitter to simulate stain variation
      - Blur/noise: simulate scanner variation
      - Elastic deformation: tissue morphology variation
    """
    return A.Compose([
        # Spatial
        A.RandomRotate90(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Transpose(p=0.3),
        A.ShiftScaleRotate(
            shift_limit=0.05, scale_limit=0.1, rotate_limit=15,
            border_mode=cv2.BORDER_REFLECT, p=0.4
        ),
        # Stain / Color
        A.ColorJitter(
            brightness=0.15, contrast=0.15,
            saturation=0.15, hue=0.05, p=0.5
        ),
        A.HueSaturationValue(
            hue_shift_limit=10, sat_shift_limit=20, val_shift_limit=15, p=0.4
        ),
        # Microscope artifacts
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
        A.GaussNoise(std_range=(0.01, 0.05), p=0.2),
        A.RandomBrightnessContrast(p=0.3),
        # Morphological variation
        A.ElasticTransform(alpha=30, sigma=5, p=0.2),
        # Normalize + to tensor
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def get_val_transforms() -> A.Compose:
    """Validation / test transforms — normalize only, no augmentation."""
    return A.Compose([
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


# 4. Full preprocessing function (called once before training)

def preprocess_split(
    X: np.ndarray,
    y: np.ndarray,
    split: str = "train",
    normalizer: Optional[MacenkoNormalizer] = None,
    sat_threshold: int = 15,
    apply_tissue_filter: bool = False,    # PCam is pre-extracted, mostly tissue
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Full preprocessing pipeline for one dataset split.

    Steps:
      1. (optional) Tissue filter
      2. Macenko normalization

    Note: augmentation is applied on-the-fly inside the Dataset/DataLoader.

    Returns:
        X_proc : uint8 normalized patches
        y_proc : labels
    """
    print(f"\n[Preprocessing — {split.upper()}]")
    print(f"  Input: {X.shape}  labels: {y.shape}")

    # Step 1: tissue filter (optional for PCam — already curated)
    kept_idx = np.arange(len(X))
    if apply_tissue_filter:
        X, y, kept_idx = filter_tissue_patches(X, y, sat_threshold)

    # Step 2: Macenko normalization
    if normalizer is not None:
        print(f"  Macenko normalization …", end=" ", flush=True)
        X_norm = np.empty_like(X)
        for i, patch in enumerate(X):
            try:
                X_norm[i] = normalizer.transform(patch)
            except Exception:
                X_norm[i] = patch    # keep original if normalization fails
            if (i + 1) % 50000 == 0:
                print(f"{i+1:,}/{len(X):,}", end=" ", flush=True)
        X = X_norm
        print("done.")
    else:
        print("  Macenko normalization: SKIPPED (no normalizer provided)")

    print(f"  Output: {X.shape}  labels: {y.shape}")
    return X, y


# Visualization helpers

def visualize_normalization(
    patches: np.ndarray,
    normalizer: MacenkoNormalizer,
    n: int = 4,
    save_path: Optional[str] = None,
):
    """Side-by-side: original vs Macenko normalized patches."""
    fig, axes = plt.subplots(n, 2, figsize=(6, n * 3))
    fig.suptitle("Macenko Stain Normalization", fontsize=14, fontweight="bold")

    for i in range(n):
        orig = patches[i]
        norm = normalizer.transform(orig)

        axes[i, 0].imshow(orig)
        axes[i, 0].set_title("Original", fontsize=9)
        axes[i, 0].axis("off")

        axes[i, 1].imshow(norm)
        axes[i, 1].set_title("Normalized", fontsize=9)
        axes[i, 1].axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {save_path}")
    plt.show()


def visualize_tissue_detection(
    patches: np.ndarray,
    n: int = 8,
    sat_threshold: int = 15,
    save_path: Optional[str] = None,
):
    """Show patches colored green (tissue) / red (background)."""
    fig, axes = plt.subplots(2, n // 2, figsize=(n * 2, 5))
    axes = axes.ravel()
    fig.suptitle("Tissue Detection (green=tissue, red=background)", fontsize=13)

    for i, ax in enumerate(axes[:n]):
        patch = patches[i]
        label = "TISSUE" if is_tissue(patch, sat_threshold) else "BACKGROUND"
        color = "green" if label == "TISSUE" else "red"
        ax.imshow(patch)
        ax.set_title(label, color=color, fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# Entry point — preview normalization on a few training patches

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from phase1_setup import load_config, load_split

    cfg = load_config()
    print("Loading a small sample of training patches …")
    X, y = load_split(cfg, "train")

    # Use first tumor patch as Macenko reference
    tumor_idx = np.where(y == 1)[0]
    reference_patch = X[tumor_idx[0]]

    normalizer = MacenkoNormalizer()
    normalizer.fit(reference_patch)
    print("  Fitted Macenko normalizer on reference patch.")

    # Visualize
    sample = X[tumor_idx[:8]]
    visualize_normalization(sample, normalizer, n=4,
                             save_path="reports/macenko_preview.png")
    visualize_tissue_detection(X[:16], n=8,
                                save_path="reports/tissue_detection_preview.png")

    print("\nPreprocessing pipeline verified.")
    print("Transforms ready:")
    print("  Train :", get_train_transforms())
    print("  Val   :", get_val_transforms())

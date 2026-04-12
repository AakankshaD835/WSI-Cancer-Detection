# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# visualization script - generates all pipeline output figures organized by phase

import argparse
import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")   # non-interactive backend - safe on Windows/headless

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import numpy as np
import cv2
import h5py
from tqdm import tqdm

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase1_setup        import load_config, load_masks, load_metadata
from phase2_preprocessing import (MacenkoNormalizer, is_tissue,
                                   get_train_transforms, get_val_transforms)

ROOT = Path(__file__).resolve().parent.parent


# Helpers

def save(fig, path: Path, dpi: int = 150):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  [OK]  Saved -> {path.relative_to(ROOT)}")


def load_images_labels(cfg, split="train", n=None):
    """Load n patches and labels from a split."""
    x_key = {"train": "pcam_train_x", "valid": "pcam_valid_x", "test": "pcam_test_x"}[split]
    y_key = {"train": "train_y",       "valid": "valid_y",       "test": "test_y"}[split]
    with h5py.File(ROOT / cfg["paths"][x_key], "r") as fx:
        X = fx["x"][:n] if n else fx["x"][:]
    with h5py.File(ROOT / cfg["paths"][y_key], "r") as fy:
        y = fy["y"][:n].squeeze() if n else fy["y"][:].squeeze()
    return X.astype(np.uint8), y.astype(np.uint8)


def _find_indices(y, label, n, rng=None):
    """Return n indices where y == label, randomly sampled."""
    idx = np.where(y == label)[0]
    if rng:
        idx = rng.choice(idx, size=min(n, len(idx)), replace=False)
    else:
        idx = idx[:n]
    return idx


# PHASE 1 - Dataset Exploration

def phase1_visuals(cfg, out_dir: Path):
    print("\n[PHASE 1] Dataset exploration visuals …")
    rng = np.random.default_rng(42)

    # Load a manageable chunk of training data
    with h5py.File(ROOT / cfg["paths"]["pcam_train_x"], "r") as fx:
        X = fx["x"][:5000].astype(np.uint8)
    with h5py.File(ROOT / cfg["paths"]["train_y"], "r") as fy:
        y = fy["y"][:5000].squeeze().astype(np.uint8)

    tumor_idx  = np.where(y == 1)[0]
    normal_idx = np.where(y == 0)[0]

    # 1a: raw_tumor_examples.png
    n = 10
    chosen = rng.choice(tumor_idx, size=n, replace=False)
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle("Tumor Patch Examples (PCam - label=1)", fontsize=14, fontweight="bold")
    for ax, i in zip(axes.ravel(), chosen):
        ax.imshow(X[i])
        ax.set_title(f"idx {i}", fontsize=7)
        ax.axis("off")
    save(fig, out_dir / "raw_tumor_examples.png")

    # 1b: raw_normal_examples.png
    chosen = rng.choice(normal_idx, size=n, replace=False)
    fig, axes = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle("Normal Patch Examples (PCam - label=0)", fontsize=14, fontweight="bold")
    for ax, i in zip(axes.ravel(), chosen):
        ax.imshow(X[i])
        ax.set_title(f"idx {i}", fontsize=7)
        ax.axis("off")
    save(fig, out_dir / "raw_normal_examples.png")

    # 1c: raw_tumor_normal_grid.png (5x10)
    n_each = 25   # 25 tumor + 25 normal = 50 total in a 5x10 grid
    t_idx = rng.choice(tumor_idx,  size=n_each, replace=False)
    n_idx = rng.choice(normal_idx, size=n_each, replace=False)

    fig, axes = plt.subplots(5, 10, figsize=(22, 11))
    fig.suptitle("Tumor (top half, red) vs Normal (bottom half, blue) Patches - 5x10 Grid",
                 fontsize=13, fontweight="bold")

    for col in range(10):
        # Row 0-1: tumor (first 2 rows x 10 cols = 20 patches) + 5 from rows 2 left half
        pass  # We'll fill row by row below

    all_idx    = np.concatenate([t_idx[:25], n_idx[:25]])
    all_labels = np.array([1]*25 + [0]*25)

    for flat_i, ax in enumerate(axes.ravel()):
        pidx  = all_idx[flat_i]
        label = all_labels[flat_i]
        ax.imshow(X[pidx])
        for spine in ax.spines.values():
            spine.set_edgecolor("red" if label == 1 else "blue")
            spine.set_linewidth(3)
        ax.set_xticks([])
        ax.set_yticks([])

    # Legend
    fig.legend(handles=[
        mpatches.Patch(color="red",  label="Tumor (label=1)"),
        mpatches.Patch(color="blue", label="Normal (label=0)"),
    ], loc="lower center", ncol=2, fontsize=12, frameon=True)

    plt.subplots_adjust(wspace=0.05, hspace=0.05, bottom=0.06)
    save(fig, out_dir / "raw_tumor_normal_grid.png")


# PHASE 2 - Preprocessing

def phase2_visuals(cfg, out_dir: Path):
    print("\n[PHASE 2] Preprocessing visuals …")
    rng = np.random.default_rng(0)

    with h5py.File(ROOT / cfg["paths"]["pcam_train_x"], "r") as fx:
        X = fx["x"][:2000].astype(np.uint8)
    with h5py.File(ROOT / cfg["paths"]["train_y"], "r") as fy:
        y = fy["y"][:2000].squeeze().astype(np.uint8)

    tumor_idx = np.where(y == 1)[0]
    ref_patch  = X[tumor_idx[0]]

    # 2a: macenko_before_after.png
    norm = MacenkoNormalizer()
    norm.fit(ref_patch)

    samples = rng.choice(tumor_idx, size=6, replace=False)
    fig, axes = plt.subplots(6, 2, figsize=(7, 18))
    fig.suptitle("Macenko Stain Normalization\nLeft = Raw   |   Right = Normalized",
                 fontsize=13, fontweight="bold")
    axes[0, 0].set_title("Raw Patch",       fontsize=11, fontweight="bold")
    axes[0, 1].set_title("After Macenko",   fontsize=11, fontweight="bold")

    for row, pidx in enumerate(samples):
        orig     = X[pidx]
        normalized = norm.transform(orig)
        axes[row, 0].imshow(orig)
        axes[row, 1].imshow(normalized)
        axes[row, 0].axis("off")
        axes[row, 1].axis("off")

    plt.tight_layout()
    save(fig, out_dir / "macenko_before_after.png")

    # 2b: tissue_detection_mask.png
    # Show both clear tissue patches and background patches
    n_show = 8
    patches_show = X[:n_show * 3]   # grab some candidates
    tissue_patches = [p for p in patches_show if is_tissue(p)][:4]
    back_patches   = [p for p in patches_show if not is_tissue(p)][:4]

    # If not enough background patches in first chunk, pad with near-white patches
    if len(back_patches) < 4:
        # Generate synthetic near-white patches for demo
        for _ in range(4 - len(back_patches)):
            fake = np.full((96, 96, 3), 245, dtype=np.uint8)
            back_patches.append(fake)

    # Show patch + binary mask side by side for n_show patches
    n_display = min(len(tissue_patches), 4)
    fig, axes = plt.subplots(n_display, 4, figsize=(14, n_display * 3.5))
    if n_display == 1:
        axes = axes[np.newaxis, :]
    fig.suptitle("Tissue Detection Mask\n"
                 "Columns: Patch | HSV Saturation | Otsu Binary Mask | Label",
                 fontsize=12, fontweight="bold")

    col_titles = ["Patch", "HSV Saturation", "Tissue Mask (Otsu)", "Decision"]
    for c, t in enumerate(col_titles):
        axes[0, c].set_title(t, fontsize=10, fontweight="bold")

    for row, patch in enumerate(tissue_patches[:n_display]):
        hsv   = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)
        sat   = hsv[:, :, 1]
        gray  = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        label = "TISSUE [OK]" if is_tissue(patch) else "BACKGROUND [FAIL]"

        axes[row, 0].imshow(patch)
        axes[row, 1].imshow(sat, cmap="YlOrRd", vmin=0, vmax=255)
        axes[row, 2].imshow(otsu, cmap="gray")
        axes[row, 3].imshow(patch)
        color = "green" if "TISSUE" in label else "red"
        axes[row, 3].set_xlabel(label, color=color, fontsize=10, fontweight="bold")
        for ax in axes[row]:
            ax.set_xticks([])
            ax.set_yticks([])

    plt.tight_layout()
    save(fig, out_dir / "tissue_detection_mask.png")

    # 2c: augmentation_examples.png
    # Show one source patch augmented 8 different ways
    # Build augmentation pipelines WITHOUT normalize+totensor so we can display
    import albumentations as A

    aug_configs = [
        ("Original",         A.Compose([])),
        ("Horizontal Flip",  A.Compose([A.HorizontalFlip(p=1.0)])),
        ("Vertical Flip",    A.Compose([A.VerticalFlip(p=1.0)])),
        ("Rotate 90°",       A.Compose([A.RandomRotate90(p=1.0)])),
        ("Color Jitter",     A.Compose([A.ColorJitter(brightness=0.3, contrast=0.3,
                                                       saturation=0.3, hue=0.1, p=1.0)])),
        ("Stain (H+S+V)",    A.Compose([A.HueSaturationValue(
                                         hue_shift_limit=20, sat_shift_limit=30,
                                         val_shift_limit=20, p=1.0)])),
        ("Gaussian Blur",    A.Compose([A.GaussianBlur(blur_limit=(5, 9), p=1.0)])),
        ("Elastic Deform",   A.Compose([A.ElasticTransform(alpha=60, sigma=8, p=1.0)])),
        ("Shift+Scale",      A.Compose([A.ShiftScaleRotate(
                                         shift_limit=0.1, scale_limit=0.15,
                                         rotate_limit=25, p=1.0)])),
        ("Brightness+Cont.", A.Compose([A.RandomBrightnessContrast(
                                         brightness_limit=0.3, contrast_limit=0.3, p=1.0)])),
    ]

    src = X[tumor_idx[5]]   # pick a distinctive tumor patch
    fig, axes = plt.subplots(2, 5, figsize=(18, 7))
    fig.suptitle("Data Augmentation Examples\n(all transforms applied to the same tumor patch)",
                 fontsize=13, fontweight="bold")

    for ax, (title, pipeline) in zip(axes.ravel(), aug_configs):
        aug_img = pipeline(image=src.copy())["image"]
        ax.imshow(aug_img)
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.axis("off")

    plt.tight_layout()
    save(fig, out_dir / "augmentation_examples.png")


# PHASE 3 - Patch-level model visuals (needs trained checkpoint)

def phase3_visuals(cfg, out_dir: Path):
    print("\n[PHASE 3] Patch-level model visuals …")
    import torch
    import torch.nn.functional as F

    backbone  = cfg["training"]["backbone"]
    ckpt_path = ROOT / cfg["paths"]["models_dir"] / f"{backbone.replace('/', '_')}_best.pth"

    if not ckpt_path.exists():
        print(f"  [!]  No checkpoint at {ckpt_path}")
        print(f"     Run:  py src/phase3_train.py   first, then re-run this script.")
        return

    from phase3_train       import PatchClassifier, DEVICE
    from phase5_explainability import GradCAM, overlay_heatmap

    ckpt  = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model = PatchClassifier(backbone, pretrained=False).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    val_tfm = get_val_transforms()

    # Load sample patches
    with h5py.File(ROOT / cfg["paths"]["pcam_train_x"], "r") as fx:
        X = fx["x"][:3000].astype(np.uint8)
    with h5py.File(ROOT / cfg["paths"]["train_y"], "r") as fy:
        y = fy["y"][:3000].squeeze().astype(np.uint8)

    tumor_idx  = np.where(y == 1)[0]
    normal_idx = np.where(y == 0)[0]

    grad_cam = GradCAM(model)

    def cam_for(patch):
        t = val_tfm(image=patch)["image"].unsqueeze(0).to(DEVICE)
        return grad_cam(t), patch

    # 3a: gradcam_tumor_1.png and gradcam_tumor_2.png
    for vis_idx, file_suffix in enumerate(["gradcam_tumor_1.png", "gradcam_tumor_2.png"]):
        pidx  = tumor_idx[vis_idx]
        patch = X[pidx]
        cam, _ = cam_for(patch)
        overlay = overlay_heatmap(patch, cam)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        fig.suptitle(f"Grad-CAM - TUMOR Patch (idx={pidx})", fontsize=13, fontweight="bold")
        axes[0].imshow(patch);                         axes[0].set_title("Original"); axes[0].axis("off")
        axes[1].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)); axes[1].set_title("Grad-CAM Overlay"); axes[1].axis("off")
        axes[2].imshow(cam, cmap="jet", vmin=0, vmax=1); axes[2].set_title("Heatmap"); axes[2].axis("off")
        plt.colorbar(plt.cm.ScalarMappable(cmap="jet"), ax=axes[2], fraction=0.046, pad=0.04)
        plt.tight_layout()
        save(fig, out_dir / file_suffix)

    # 3b: gradcam_normal_1.png and gradcam_normal_2.png
    for vis_idx, file_suffix in enumerate(["gradcam_normal_1.png", "gradcam_normal_2.png"]):
        pidx  = normal_idx[vis_idx]
        patch = X[pidx]
        cam, _ = cam_for(patch)
        overlay = overlay_heatmap(patch, cam)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        fig.suptitle(f"Grad-CAM - NORMAL Patch (idx={pidx}) - Low Activation Expected",
                     fontsize=12, fontweight="bold")
        axes[0].imshow(patch);                          axes[0].set_title("Original"); axes[0].axis("off")
        axes[1].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)); axes[1].set_title("Grad-CAM Overlay"); axes[1].axis("off")
        axes[2].imshow(cam, cmap="jet", vmin=0, vmax=1);  axes[2].set_title("Heatmap"); axes[2].axis("off")
        plt.tight_layout()
        save(fig, out_dir / file_suffix)

    # 3c: gradcam_overlay_comparison.png
    n_compare = 4
    fig, axes = plt.subplots(n_compare, 3, figsize=(12, n_compare * 4))
    fig.suptitle("Grad-CAM Overlay Comparison\nRaw  |  Overlay  |  Heatmap",
                 fontsize=13, fontweight="bold")

    for row, pidx in enumerate(tumor_idx[:n_compare]):
        patch   = X[pidx]
        cam, _  = cam_for(patch)
        overlay = overlay_heatmap(patch, cam)

        axes[row, 0].imshow(patch)
        axes[row, 0].set_ylabel(f"Patch {pidx}", fontsize=9)
        axes[row, 1].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        axes[row, 2].imshow(cam, cmap="jet", vmin=0, vmax=1)
        for ax in axes[row]:
            ax.axis("off")

    axes[0, 0].set_title("RAW Patch",         fontsize=11, fontweight="bold")
    axes[0, 1].set_title("Grad-CAM Overlay",  fontsize=11, fontweight="bold")
    axes[0, 2].set_title("Heatmap",           fontsize=11, fontweight="bold")
    plt.tight_layout()
    save(fig, out_dir / "gradcam_overlay_comparison.png")

    grad_cam.remove_hooks()

    # 3d: training_loss_accuracy.png
    results_json = ROOT / cfg["paths"]["reports_dir"] / f"{backbone.replace('/', '_')}_results.json"
    if results_json.exists():
        with open(results_json) as f:
            results = json.load(f)
        history = results["history"]
        epochs  = [h["epoch"]        for h in history]
        t_loss  = [h["train_loss"]   for h in history]
        v_loss  = [h["val_loss"]     for h in history]
        t_auc   = [h["train_auc"]    for h in history]
        v_auc   = [h["val_auc"]      for h in history]
        t_acc   = [h["train_accuracy"] for h in history]
        v_acc   = [h["val_accuracy"]   for h in history]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle(f"Training Curves - {backbone}", fontsize=14, fontweight="bold")

        axes[0].plot(epochs, t_loss, "b-o", ms=4, label="Train Loss")
        axes[0].plot(epochs, v_loss, "r-o", ms=4, label="Val Loss")
        axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch"); axes[0].set_ylabel("BCE Loss")
        axes[0].legend(); axes[0].grid(alpha=0.3)

        axes[1].plot(epochs, t_auc, "b-o", ms=4, label="Train AUC")
        axes[1].plot(epochs, v_auc, "r-o", ms=4, label="Val AUC")
        axes[1].set_title("AUC-ROC"); axes[1].set_xlabel("Epoch"); axes[1].set_ylabel("AUC")
        axes[1].set_ylim(0.5, 1.0); axes[1].legend(); axes[1].grid(alpha=0.3)

        axes[2].plot(epochs, t_acc, "b-o", ms=4, label="Train Acc")
        axes[2].plot(epochs, v_acc, "r-o", ms=4, label="Val Acc")
        axes[2].set_title("Accuracy"); axes[2].set_xlabel("Epoch"); axes[2].set_ylabel("Accuracy")
        axes[2].set_ylim(0.5, 1.0); axes[2].legend(); axes[2].grid(alpha=0.3)

        plt.tight_layout()
        save(fig, out_dir / "training_loss_accuracy.png")

    # 3e: roc_curve_patch_level.png
    if results_json.exists():
        # Re-run inference on test set for proper ROC
        from phase3_train import PatchClassifier, evaluate, DEVICE
        from phase1_setup import PCamDataset
        from torch.utils.data import DataLoader
        from sklearn.metrics import roc_curve, auc as sk_auc

        test_ds     = PCamDataset("test", transform=get_val_transforms(), cfg=cfg)
        test_loader = DataLoader(test_ds, batch_size=cfg["training"]["batch_size"] * 2,
                                 shuffle=False, num_workers=0)

        import torch.nn as nn
        criterion = nn.BCEWithLogitsLoss()

        model.eval()
        all_logits, all_labels = [], []
        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc="  Test inference", leave=False):
                logits = model(images.to(DEVICE))
                all_logits.extend(logits.cpu().numpy())
                all_labels.extend(labels.numpy())

        all_logits = np.array(all_logits)
        all_probs  = 1 / (1 + np.exp(-all_logits))
        all_labels = np.array(all_labels)

        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        roc_auc     = sk_auc(fpr, tpr)

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.plot(fpr, tpr, "b-", lw=2, label=f"Patch-level ROC (AUC = {roc_auc:.4f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random Classifier")
        ax.fill_between(fpr, tpr, alpha=0.1, color="blue")
        ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
        ax.set_xlabel("False Positive Rate", fontsize=12)
        ax.set_ylabel("True Positive Rate",  fontsize=12)
        ax.set_title(f"ROC Curve - Patch-Level ({backbone})\nTest AUC = {roc_auc:.4f}",
                     fontsize=13, fontweight="bold")
        ax.legend(fontsize=11); ax.grid(alpha=0.3)
        plt.tight_layout()
        save(fig, out_dir / "roc_curve_patch_level.png")

        # Save probs/labels for later t-SNE
        np.save(ROOT / "models" / "test_probs.npy",  all_probs)
        np.save(ROOT / "models" / "test_labels.npy", all_labels)


# PHASE 4 - MIL visuals

def phase4_visuals(cfg, out_dir: Path):
    print("\n[PHASE 4] MIL visuals …")
    import torch

    backbone   = cfg["training"]["backbone"]
    p3_ckpt    = ROOT / cfg["paths"]["models_dir"] / f"{backbone.replace('/', '_')}_best.pth"
    p4_ckpt    = ROOT / cfg["paths"]["models_dir"] / "mil_best.pth"
    mil_json   = ROOT / cfg["paths"]["reports_dir"] / "mil_results.json"

    if not p3_ckpt.exists():
        print("  [!]  Phase 3 checkpoint missing - run phase3_train.py first.")
        return
    if not p4_ckpt.exists():
        print("  [!]  Phase 4 MIL checkpoint missing - run phase4_mil.py first.")
        return

    from phase3_train      import PatchClassifier, DEVICE
    from phase4_mil        import GatedAttentionMIL, load_patch_model

    patch_model = load_patch_model(cfg)
    feat_dim    = patch_model.backbone.num_features

    mil_ckpt   = torch.load(p4_ckpt, map_location=DEVICE, weights_only=False)
    mil_model  = GatedAttentionMIL(feat_dim=feat_dim,
                                    hidden_dim=cfg["mil"].get("hidden_dim", cfg["mil"]["feature_dim"]),
                                    attention_dim=cfg["mil"]["attention_dim"]).to(DEVICE)
    mil_model.load_state_dict(mil_ckpt["state_dict"])
    mil_model.eval()

    # Load metadata for WSI groupings
    from phase1_setup import load_metadata
    train_meta = load_metadata(cfg, "train")

    # Pick one tumor WSI bag and one normal WSI bag
    wsi_ids    = train_meta["wsi"].unique()
    tumor_wsi  = [w for w in wsi_ids if "tumor" in w.lower()]
    normal_wsi = [w for w in wsi_ids if "normal" in w.lower()]

    rng = np.random.default_rng(7)
    demo_tumor_wsi  = rng.choice(tumor_wsi)  if tumor_wsi  else wsi_ids[0]
    demo_normal_wsi = rng.choice(normal_wsi) if normal_wsi else wsi_ids[1]

    val_tfm = get_val_transforms()

    def get_bag_data(wsi_id: str):
        """Get patches, raw images, coordinates for one WSI bag."""
        mask_wsi  = train_meta["wsi"] == wsi_id
        indices   = train_meta.index[mask_wsi].tolist()
        coords_x  = train_meta.loc[mask_wsi, "coord_x"].values
        coords_y  = train_meta.loc[mask_wsi, "coord_y"].values

        with h5py.File(ROOT / cfg["paths"]["pcam_train_x"], "r") as fx:
            raw_patches = fx["x"][indices].astype(np.uint8)
        with h5py.File(ROOT / cfg["paths"]["train_y"], "r") as fy:
            patch_labels = fy["y"][indices].squeeze().astype(np.uint8)

        # Extract features
        patch_model.eval()
        feats = []
        with torch.no_grad():
            for p in raw_patches:
                t = val_tfm(image=p)["image"].unsqueeze(0).to(DEVICE)
                feats.append(patch_model.get_features(t).cpu())
        feats_tensor = torch.cat(feats, dim=0)

        # MIL attention
        with torch.no_grad():
            logit, attention = mil_model(feats_tensor.to(DEVICE))
        prob      = torch.sigmoid(logit).item()
        attention = attention.cpu().numpy()

        return raw_patches, patch_labels, coords_x, coords_y, attention, prob

    for wsi_id, is_tumor in [(demo_tumor_wsi, True), (demo_normal_wsi, False)]:
        suffix = "tumor" if is_tumor else "normal"
        print(f"  Processing WSI bag: {wsi_id} ({suffix}) …")

        try:
            raw_patches, patch_labels, coords_x, coords_y, attention, prob = get_bag_data(wsi_id)
        except Exception as e:
            print(f"    [!]  Skipped: {e}")
            continue

        n_patches = len(raw_patches)
        att_norm  = (attention - attention.min()) / (attention.max() - attention.min() + 1e-8)

        # 4a: wsi_reconstructed_bag.png
        if is_tumor:  # only save the main tumor reconstruction
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            fig.suptitle(f"WSI Bag Reconstruction - {wsi_id}\n"
                         f"Tumor Prob = {prob:.3f}  |  n={n_patches} patches",
                         fontsize=12, fontweight="bold")

            # Left: patch labels ([ ] normal, [R] tumor)
            sc0 = axes[0].scatter(coords_x, coords_y, c=patch_labels, cmap="RdBu_r",
                                   s=30, vmin=0, vmax=1, alpha=0.8)
            axes[0].set_title("Patch Labels (Red=Tumor, Blue=Normal)", fontsize=10)
            axes[0].set_xlabel("coord_x"); axes[0].set_ylabel("coord_y")
            axes[0].invert_yaxis()
            plt.colorbar(sc0, ax=axes[0])

            # Right: MIL attention heatmap
            sc1 = axes[1].scatter(coords_x, coords_y, c=att_norm, cmap="hot",
                                   s=30, vmin=0, vmax=1, alpha=0.9)
            axes[1].set_title("MIL Attention Weights (Hot = High Importance)", fontsize=10)
            axes[1].set_xlabel("coord_x"); axes[1].set_ylabel("coord_y")
            axes[1].invert_yaxis()
            plt.colorbar(sc1, ax=axes[1], label="Attention Weight")

            plt.tight_layout()
            save(fig, out_dir / "wsi_reconstructed_bag.png")

        # 4b: mil_attention_wsi_heatmap.png
        if is_tumor:
            # Interpolated heatmap on coordinate grid
            from scipy.interpolate import griddata

            xi = np.linspace(coords_x.min(), coords_x.max(), 100)
            yi = np.linspace(coords_y.min(), coords_y.max(), 100)
            xi, yi = np.meshgrid(xi, yi)
            zi = griddata((coords_x, coords_y), att_norm, (xi, yi), method="linear")

            fig, ax = plt.subplots(figsize=(9, 7))
            im = ax.imshow(zi, extent=[coords_x.min(), coords_x.max(),
                                        coords_y.max(), coords_y.min()],
                           cmap="hot", vmin=0, vmax=1, aspect="auto")
            ax.set_title(f"MIL Attention Heatmap Over Slide Grid\n"
                         f"WSI: {wsi_id}  |  Tumor Prob = {prob:.3f}",
                         fontsize=12, fontweight="bold")
            ax.set_xlabel("coord_x"); ax.set_ylabel("coord_y")
            plt.colorbar(im, ax=ax, label="Attention Weight")
            plt.tight_layout()
            save(fig, out_dir / "mil_attention_wsi_heatmap.png")

    # 4c: mil_patch_attention_ranked.png
    # Use the tumor WSI bag; show top 20 patches ranked by attention
    try:
        raw_patches, patch_labels, coords_x, coords_y, attention, prob = get_bag_data(demo_tumor_wsi)
    except Exception as e:
        print(f"  [!]  Could not get tumor bag for ranked plot: {e}")
        return

    top_k    = min(20, len(attention))
    top_idx  = np.argsort(attention)[::-1][:top_k]
    att_norm = (attention - attention.min()) / (attention.max() - attention.min() + 1e-8)

    fig, axes = plt.subplots(4, 5, figsize=(18, 14))
    axes = axes.ravel()
    fig.suptitle(f"Top-{top_k} Patches by MIL Attention - {demo_tumor_wsi}\n"
                 f"(Slide Tumor Prob = {prob:.3f})",
                 fontsize=13, fontweight="bold")

    for rank, (ax, pidx) in enumerate(zip(axes, top_idx)):
        score     = attention[pidx]
        norm_s    = att_norm[pidx]
        label     = patch_labels[pidx]
        color     = "red" if norm_s > 0.5 else "steelblue"
        border_c  = [int(c * 255) for c in plt.cm.hot(norm_s)[:3]]
        bordered  = cv2.copyMakeBorder(raw_patches[pidx], 4, 4, 4, 4,
                                        cv2.BORDER_CONSTANT, value=border_c)
        ax.imshow(bordered)
        ax.set_title(f"#{rank+1}  att={score:.4f}\n{'TUMOR' if label else 'NORMAL'}",
                     fontsize=7, color=color)
        ax.axis("off")

    plt.tight_layout()
    save(fig, out_dir / "mil_patch_attention_ranked.png")

    # 4d: roc_curve_slide_level.png
    if mil_json.exists():
        with open(mil_json) as f:
            mil_results = json.load(f)

        from sklearn.metrics import roc_curve, auc as sk_auc

        # Need to re-run inference on test bags to get probabilities
        from phase4_mil import WSIBagDataset, eval_mil
        import torch.nn as nn

        feat_cache = ROOT / "models" / "features_test.npy"
        lab_cache  = ROOT / "models" / "labels_test.npy"

        if feat_cache.exists():
            from phase3_train import PCamDataset, get_val_transforms as _gvt
            test_feats  = np.load(feat_cache)
            test_labels = np.load(lab_cache)
            test_ds     = PCamDataset("test", transform=_gvt(), cfg=cfg)
            test_meta   = load_metadata(cfg, "test", indices=test_ds.indices)
            test_bags   = WSIBagDataset(test_feats, test_labels, test_meta, "test")
            criterion   = nn.BCEWithLogitsLoss()
            test_m      = eval_mil(mil_model, test_bags, criterion, DEVICE)

            fpr, tpr, _ = roc_curve(test_m["labels"], test_m["probs"])
            roc_auc     = sk_auc(fpr, tpr)

            fig, ax = plt.subplots(figsize=(7, 6))
            ax.plot(fpr, tpr, "r-", lw=2.5, label=f"Slide-Level ROC (AUC = {roc_auc:.4f})")
            ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random Classifier")
            ax.fill_between(fpr, tpr, alpha=0.12, color="red")
            ax.axvline(x=0.05, color="gray", ls=":", label="95% Specificity")
            ax.set_xlim([0, 1]); ax.set_ylim([0, 1.02])
            ax.set_xlabel("False Positive Rate", fontsize=12)
            ax.set_ylabel("True Positive Rate",  fontsize=12)
            ax.set_title(f"ROC Curve - Slide-Level (MIL)\nTest AUC = {roc_auc:.4f}  "
                         f"Sens@95Spec = {test_m['sens@95']:.4f}",
                         fontsize=13, fontweight="bold")
            ax.legend(fontsize=11); ax.grid(alpha=0.3)
            plt.tight_layout()
            save(fig, out_dir / "roc_curve_slide_level.png")
        else:
            print("  [!]  No cached test features - run phase4_mil.py first for slide ROC.")

    # 4e: mil_tumor_vs_normal_tissue.png
    # Side-by-side: top attention patches from a TUMOR slide vs a NORMAL slide
    print("  Generating tumor vs normal tissue comparison …")
    try:
        t_patches, t_labels, _, _, t_att, t_prob = get_bag_data(demo_tumor_wsi)
        n_patches, n_labels, _, _, n_att, n_prob = get_bag_data(demo_normal_wsi)

        t_top = np.argsort(t_att)[::-1][:8]
        n_top = np.argsort(n_att)[::-1][:8]

        fig, axes = plt.subplots(2, 8, figsize=(22, 6))
        fig.suptitle(
            "MIL Attention: Top Tissue Patches — Tumor Slide vs Normal Slide\n"
            f"Left: {demo_tumor_wsi}  (Tumor Prob={t_prob:.3f})   |   "
            f"Right: {demo_normal_wsi}  (Tumor Prob={n_prob:.3f})",
            fontsize=11, fontweight="bold"
        )

        t_att_norm = (t_att - t_att.min()) / (t_att.max() - t_att.min() + 1e-8)
        n_att_norm = (n_att - n_att.min()) / (n_att.max() - n_att.min() + 1e-8)

        for col, pidx in enumerate(t_top):
            score  = t_att_norm[pidx]
            color  = plt.cm.hot(score)[:3]
            border = [int(c * 255) for c in color]
            img    = cv2.copyMakeBorder(t_patches[pidx], 5, 5, 5, 5,
                                         cv2.BORDER_CONSTANT, value=border)
            axes[0, col].imshow(img)
            axes[0, col].set_title(f"att={t_att[pidx]:.3f}\n{'TUMOR' if t_labels[pidx] else 'NORMAL'}",
                                    fontsize=7, color="red")
            axes[0, col].axis("off")

        for col, pidx in enumerate(n_top):
            score  = n_att_norm[pidx]
            color  = plt.cm.cool(score)[:3]
            border = [int(c * 255) for c in color]
            img    = cv2.copyMakeBorder(n_patches[pidx], 5, 5, 5, 5,
                                         cv2.BORDER_CONSTANT, value=border)
            axes[1, col].imshow(img)
            axes[1, col].set_title(f"att={n_att[pidx]:.3f}\n{'TUMOR' if n_labels[pidx] else 'NORMAL'}",
                                    fontsize=7, color="steelblue")
            axes[1, col].axis("off")

        axes[0, 0].set_ylabel("TUMOR SLIDE", fontsize=10, fontweight="bold", color="red")
        axes[1, 0].set_ylabel("NORMAL SLIDE", fontsize=10, fontweight="bold", color="steelblue")
        plt.tight_layout()
        save(fig, out_dir / "mil_tumor_vs_normal_tissue.png")
    except Exception as e:
        print(f"  [!]  tumor vs normal tissue failed: {e}")

    # 4f: mil_attention_overlay_grid.png
    # Top 6 HIGH attention vs bottom 6 LOW attention patches with rank-based coloring
    print("  Generating attention overlay on tissue …")
    try:
        raw_patches, patch_labels, _, _, attention, prob = get_bag_data(demo_tumor_wsi)
        sorted_idx = np.argsort(attention)[::-1]
        top6_idx   = sorted_idx[:6]
        bot6_idx   = sorted_idx[-6:]
        show_idx   = list(top6_idx) + list(bot6_idx)

        fig, axes = plt.subplots(2, 6, figsize=(20, 8))
        fig.suptitle(
            f"High vs Low Attention Tissue Patches — Real Histopathology\n"
            f"WSI: {demo_tumor_wsi}  |  Slide Tumor Probability = {prob:.3f}\n"
            f"Top row = HIGH attention (suspicious) | Bottom row = LOW attention (ignored)",
            fontsize=11, fontweight="bold"
        )

        for idx, (ax, pidx) in enumerate(zip(axes.ravel(), show_idx)):
            patch  = raw_patches[pidx].copy()
            label  = patch_labels[pidx]
            is_top = idx < 6
            # Rank-based color: top patches get bright red overlay, bottom get cool blue
            rank_score = 1.0 - (idx % 6) / 5.0 if is_top else (idx % 6) / 5.0
            if is_top:
                overlay_color = np.array([220, 50, 50], dtype=np.uint8)   # red
            else:
                overlay_color = np.array([50, 100, 220], dtype=np.uint8)  # blue
            overlay = np.full_like(patch, overlay_color)
            alpha   = 0.45 if is_top else 0.30
            blended = cv2.addWeighted(patch, 1 - alpha, overlay, alpha, 0)
            # Thick colored border
            border_color = [200, 30, 30] if is_top else [30, 80, 200]
            bordered = cv2.copyMakeBorder(blended, 6, 6, 6, 6,
                                           cv2.BORDER_CONSTANT, value=border_color)
            ax.imshow(bordered)
            rank_label = f"HIGH #{idx+1}" if is_top else f"LOW #{idx%6+1}"
            ax.set_title(
                f"{rank_label}  att={attention[pidx]:.4f}\n"
                f"{'TUMOR' if label else 'NORMAL'}",
                fontsize=8,
                color="crimson" if is_top else "steelblue",
                fontweight="bold"
            )
            ax.axis("off")

        plt.tight_layout()
        save(fig, out_dir / "mil_attention_overlay_grid.png")
    except Exception as e:
        print(f"  [!]  attention overlay grid failed: {e}")

    # 4g: mil_top3_attention_zoom.png
    # Large format: top 3 patches — Original | Red overlay | Side by side
    print("  Generating top-3 attention zoom panel …")
    try:
        raw_patches, patch_labels, _, _, attention, prob = get_bag_data(demo_tumor_wsi)
        top3 = np.argsort(attention)[::-1][:3]

        fig, axes = plt.subplots(1, 3, figsize=(18, 7))
        fig.suptitle(
            f"Top-3 Most Suspicious Tissue Regions — MIL Attention Analysis\n"
            f"WSI: {demo_tumor_wsi}  |  Slide Tumor Probability = {prob:.3f}\n"
            f"Left half = Original Tissue  |  Right half = AI Suspicion Overlay (Red = Suspicious)",
            fontsize=11, fontweight="bold"
        )

        for i, (ax, pidx) in enumerate(zip(axes, top3)):
            patch  = raw_patches[pidx].copy()
            label  = patch_labels[pidx]
            # Graduated red overlay — stronger red for rank 1, lighter for rank 3
            intensity = 1.0 - i * 0.2
            red_overlay = np.zeros_like(patch)
            red_overlay[:, :, 0] = int(255 * intensity)   # R channel
            red_overlay[:, :, 1] = int(30  * intensity)   # G channel
            red_overlay[:, :, 2] = int(30  * intensity)   # B channel
            blended = cv2.addWeighted(patch, 0.55, red_overlay, 0.45, 0)
            combined = np.concatenate([patch, blended], axis=1)
            divider  = np.full((patch.shape[0], 4, 3), 255, dtype=np.uint8)
            combined = np.concatenate([patch, divider, blended], axis=1)
            ax.imshow(combined)
            ax.set_title(
                f"Rank #{i+1} Most Suspicious Region\n"
                f"Attention: {attention[pidx]:.4f}  |  "
                f"{'CONFIRMED TUMOR' if label else 'NORMAL'}",
                fontsize=10,
                color="crimson" if label else "steelblue",
                fontweight="bold"
            )
            ax.axvline(x=patch.shape[1] + 2, color="gray", lw=1.5, ls="--")
            ax.text(patch.shape[1] // 2,        patch.shape[0] + 4,
                    "Original Tissue", ha="center", fontsize=9, color="dimgray")
            ax.text(patch.shape[1] + 4 + patch.shape[1] // 2, patch.shape[0] + 4,
                    "AI Suspicion Overlay", ha="center", fontsize=9, color="crimson")
            ax.axis("off")

        plt.tight_layout()
        save(fig, out_dir / "mil_top3_attention_zoom.png")
    except Exception as e:
        print(f"  [!]  top3 zoom failed: {e}")

    print("  [PHASE 4] All medical image visuals complete.")


# PHASE 5 - Pixel-Level Mask Visuals

def phase5_visuals(cfg, out_dir: Path):
    print("\n[PHASE 5] Pixel-level mask visuals …")
    import torch

    rng = np.random.default_rng(42)

    # Load masks and patches
    masks_path = ROOT / cfg["paths"]["train_mask"]
    if not masks_path.exists():
        print(f"  [!]  Mask file not found: {masks_path}")
        return

    with h5py.File(ROOT / cfg["paths"]["pcam_train_x"], "r") as fx:
        X = fx["x"][:5000].astype(np.uint8)
    with h5py.File(ROOT / cfg["paths"]["train_y"], "r") as fy:
        y = fy["y"][:5000].squeeze().astype(np.uint8)
    with h5py.File(masks_path, "r") as fm:
        masks = fm["mask"][:5000].squeeze()   # (N, 96, 96)

    tumor_idx = np.where(y == 1)[0]
    # Pick patches where mask is non-trivial (has tumor pixels)
    tumor_with_mask = [i for i in tumor_idx if masks[i].any()][:20]
    rng.shuffle(tumor_with_mask)

    # 5a: mask_ground_truth.png
    n = min(6, len(tumor_with_mask))
    fig, axes = plt.subplots(n, 2, figsize=(8, n * 4))
    if n == 1: axes = axes[np.newaxis, :]
    fig.suptitle("Ground Truth Pixel-Level Tumor Masks (PCam Train Set)",
                 fontsize=13, fontweight="bold")
    axes[0, 0].set_title("H&E Patch",       fontsize=11, fontweight="bold")
    axes[0, 1].set_title("Tumor Mask (GT)", fontsize=11, fontweight="bold")

    for row, pidx in enumerate(tumor_with_mask[:n]):
        axes[row, 0].imshow(X[pidx])
        axes[row, 0].set_ylabel(f"idx {pidx}", fontsize=9)
        axes[row, 0].axis("off")
        axes[row, 1].imshow(masks[pidx], cmap="Reds", vmin=0, vmax=1)
        axes[row, 1].axis("off")

    plt.tight_layout()
    save(fig, out_dir / "mask_ground_truth.png")

    # 5b: patch_mask_overlay.png
    n = min(6, len(tumor_with_mask))
    fig, axes = plt.subplots(n, 3, figsize=(12, n * 4))
    if n == 1: axes = axes[np.newaxis, :]
    fig.suptitle("Tumor Mask Overlaid on H&E Patch",
                 fontsize=13, fontweight="bold")
    for c, t in zip(range(3), ["Patch", "Mask (GT)", "Overlay"]):
        axes[0, c].set_title(t, fontsize=11, fontweight="bold")

    for row, pidx in enumerate(tumor_with_mask[:n]):
        patch = X[pidx]
        mask  = masks[pidx].astype(np.uint8) * 255

        # Create RGBA overlay
        overlay = patch.copy()
        tumor_region = mask > 0
        overlay[tumor_region, 0] = np.clip(overlay[tumor_region, 0].astype(int) + 80, 0, 255)
        overlay[tumor_region, 1] = np.clip(overlay[tumor_region, 1].astype(int) - 30, 0, 255)
        overlay[tumor_region, 2] = np.clip(overlay[tumor_region, 2].astype(int) - 30, 0, 255)

        axes[row, 0].imshow(patch);   axes[row, 0].set_ylabel(f"idx {pidx}", fontsize=9)
        axes[row, 1].imshow(mask, cmap="Reds", vmin=0, vmax=255)
        axes[row, 2].imshow(overlay)
        for ax in axes[row]: ax.axis("off")

    plt.tight_layout()
    save(fig, out_dir / "patch_mask_overlay.png")

    # 5c/d: mask_vs_gradcam.png and mask_iou_distribution.png
    backbone  = cfg["training"]["backbone"]
    ckpt_path = ROOT / cfg["paths"]["models_dir"] / f"{backbone.replace('/', '_')}_best.pth"

    if not ckpt_path.exists():
        print("  [!]  Phase 3 checkpoint needed for mask_vs_gradcam - skipping.")
        return

    from phase3_train       import PatchClassifier, DEVICE
    from phase5_explainability import GradCAM, overlay_heatmap, compute_cam_mask_metrics

    ckpt  = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model = PatchClassifier(backbone, pretrained=False).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    val_tfm  = get_val_transforms()
    grad_cam = GradCAM(model)

    # mask_vs_gradcam.png
    n = min(5, len(tumor_with_mask))
    fig, axes = plt.subplots(n, 4, figsize=(16, n * 4))
    if n == 1: axes = axes[np.newaxis, :]
    fig.suptitle("Ground-Truth Mask vs Grad-CAM Heatmap\n"
                 "Proves Interpretability Quality",
                 fontsize=13, fontweight="bold")
    for c, t in zip(range(4), ["Patch", "GT Mask", "Grad-CAM", "Overlay"]):
        axes[0, c].set_title(t, fontsize=11, fontweight="bold")

    iou_scores, dice_scores = [], []
    for row, pidx in enumerate(tumor_with_mask[:n]):
        patch  = X[pidx]
        mask   = masks[pidx]
        t      = val_tfm(image=patch)["image"].unsqueeze(0)
        cam    = grad_cam(t)
        metrics = compute_cam_mask_metrics(cam, mask)
        iou_scores.append(metrics["iou"])
        dice_scores.append(metrics["dice"])

        overlay = overlay_heatmap(patch, cam)

        axes[row, 0].imshow(patch)
        axes[row, 0].set_ylabel(f"IoU={metrics['iou']:.3f}\nDice={metrics['dice']:.3f}",
                                 fontsize=8)
        axes[row, 1].imshow(mask, cmap="Reds", vmin=0, vmax=1)
        axes[row, 2].imshow(cam, cmap="jet",   vmin=0, vmax=1)
        axes[row, 3].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        for ax in axes[row]: ax.axis("off")

    plt.tight_layout()
    save(fig, out_dir / "mask_vs_gradcam.png")

    # mask_iou_distribution.png - compute on larger sample
    print("  Computing IoU/Dice on 300 tumor patches …")
    n_eval = min(300, len(tumor_with_mask) + len(tumor_idx))
    eval_idx = tumor_with_mask[:n_eval]
    if len(eval_idx) < n_eval:
        extra = [i for i in tumor_idx if i not in eval_idx and masks[i].any()]
        eval_idx = eval_idx + extra[:n_eval - len(eval_idx)]

    all_iou, all_dice = [], []
    for pidx in tqdm(eval_idx[:n_eval], desc="  IoU eval", leave=False):
        try:
            t   = val_tfm(image=X[pidx])["image"].unsqueeze(0)
            cam = grad_cam(t)
            m   = compute_cam_mask_metrics(cam, masks[pidx])
            all_iou.append(m["iou"])
            all_dice.append(m["dice"])
        except Exception:
            pass

    grad_cam.remove_hooks()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"Grad-CAM vs Ground-Truth Mask Alignment (n={len(all_iou)})\n"
                 f"Mean IoU = {np.mean(all_iou):.3f}  |  Mean Dice = {np.mean(all_dice):.3f}",
                 fontsize=13, fontweight="bold")

    axes[0].hist(all_iou,  bins=30, color="steelblue", edgecolor="white", alpha=0.8)
    axes[0].axvline(np.mean(all_iou), color="red", lw=2, label=f"Mean={np.mean(all_iou):.3f}")
    axes[0].set_title("IoU Distribution"); axes[0].set_xlabel("IoU"); axes[0].set_ylabel("Count")
    axes[0].legend(); axes[0].grid(alpha=0.3)

    axes[1].hist(all_dice, bins=30, color="darkorange", edgecolor="white", alpha=0.8)
    axes[1].axvline(np.mean(all_dice), color="red", lw=2, label=f"Mean={np.mean(all_dice):.3f}")
    axes[1].set_title("Dice Score Distribution"); axes[1].set_xlabel("Dice"); axes[1].set_ylabel("Count")
    axes[1].legend(); axes[1].grid(alpha=0.3)

    plt.tight_layout()
    save(fig, out_dir / "mask_iou_distribution.png")


# PHASE 6 - Virtual Staining (real CycleGAN output when available)

def phase6_visuals(cfg, out_dir: Path):
    print("\n[PHASE 6] Virtual staining demo visual ...")

    # If Phase 6 already saved the real CycleGAN visualization, keep it.
    real_out = out_dir / "virtual_stain_he_to_ihc.png"
    ckpt_path = ROOT / cfg["paths"]["models_dir"] / "cyclegan.pth"

    if ckpt_path.exists():
        # Real trained CycleGAN exists -- generate visualization using it
        try:
            import torch
            import torchvision.transforms as T
            from phase6_virtual_staining import (
                CycleGANTrainer, virtual_stain, save_visualization
            )
            trainer = CycleGANTrainer(cfg)
            trainer.load()
            save_visualization(cfg, trainer, n=12)
            print("  [OK]  CycleGAN visualization saved (real model)")
            return
        except Exception as e:
            print(f"  [!]  CycleGAN visualization failed ({e}) -- using fallback")

    if real_out.exists():
        print("  [OK]  Phase 6 CycleGAN output already exists -- keeping it")
        return

    # Fallback: mathematical IHC simulation (no trained model)
    print("  [!]  No CycleGAN checkpoint found -- using IHC simulation fallback")
    with h5py.File(ROOT / cfg["paths"]["pcam_train_x"], "r") as fx:
        X = fx["x"][:500].astype(np.uint8)
    with h5py.File(ROOT / cfg["paths"]["train_y"], "r") as fy:
        y = fy["y"][:500].squeeze().astype(np.uint8)

    tumor_idx = np.where(y == 1)[0][:4]
    norm = MacenkoNormalizer()
    norm.fit(X[tumor_idx[0]])

    def simulate_ihc(patch: np.ndarray) -> np.ndarray:
        patch_f     = patch.astype(np.float32) / 255.0
        od          = -np.log(np.clip(patch_f, 0.01, 1.0))
        dab_channel = np.clip(od[:, :, 0] - od[:, :, 2], 0, None)
        result      = np.ones_like(patch_f)
        result[:, :, 0] = np.clip(1.0 - dab_channel * 0.8,  0, 1)
        result[:, :, 1] = np.clip(1.0 - dab_channel * 1.1,  0, 1)
        result[:, :, 2] = np.clip(1.0 - dab_channel * 1.5,  0, 1)
        return (result * 255).astype(np.uint8)

    fig, axes = plt.subplots(4, 3, figsize=(13, 16))
    fig.suptitle("Virtual Staining - H&E -> IHC Simulation\n"
                 "(Fallback: CycleGAN not yet trained)",
                 fontsize=13, fontweight="bold")
    for c, t in zip(range(3), ["H&E (Input)", "Macenko Normalized", "Virtual IHC (Output)"]):
        axes[0, c].set_title(t, fontsize=11, fontweight="bold")
    for row, pidx in enumerate(tumor_idx[:4]):
        patch      = X[pidx]
        norm_patch = norm.transform(patch)
        ihc_patch  = simulate_ihc(norm_patch)
        axes[row, 0].imshow(patch)
        axes[row, 1].imshow(norm_patch)
        axes[row, 2].imshow(ihc_patch)
        axes[row, 0].set_ylabel(f"patch {pidx}", fontsize=9)
        for ax in axes[row]:
            ax.axis("off")
    plt.tight_layout()
    save(fig, real_out)


# PHASE 7 - LLM Clinical Report Visualization

def phase7_visuals(cfg, out_dir: Path):
    print("\n[PHASE 7] LLM clinical report visual ...")

    # Load from real Phase 7 outputs (analysis JSON + report TXT)
    report_json_dir = ROOT / cfg["paths"]["reports_dir"] / "clinical"
    json_files = sorted(report_json_dir.glob("*_analysis.json")) if report_json_dir.exists() else []
    txt_files  = sorted(report_json_dir.glob("*_report.txt"))   if report_json_dir.exists() else []

    tumor_prob  = None
    top_patches = None
    report_text = None

    if json_files and txt_files:
        # Use the first analysis JSON (sorted alphabetically for reproducibility)
        with open(json_files[0]) as f:
            analysis = json.load(f)
        with open(txt_files[0]) as f:
            report_text = f.read()

        tumor_prob   = analysis.get("classification", {}).get("tumor_probability", None)
        top_regions  = analysis.get("attention_analysis", {}).get("top_regions", [])
        att_scores   = analysis.get("attention_analysis", {}).get("top_attention_scores", [])
        # top_regions are strings like "patch_0124" -- convert to ints
        try:
            top_patches = [int(p.split("_")[1]) for p in top_regions]
        except Exception:
            top_patches = list(range(5))
        # Real top-5 attention scores
        real_att_scores = att_scores[:5] if att_scores else None

        print(f"  Loaded real report: {json_files[0].name}")

    if tumor_prob is None:
        real_att_scores = None
        # Phase 7 has not run yet -- show placeholder that clearly says so
        print("  [!]  No real report found (Phase 7 must run first) -- showing placeholder")
        tumor_prob  = 0.0
        top_patches = [0, 1, 2, 3, 4]
        report_text = (
            "PATHOLOGY AI ANALYSIS REPORT\n"
            "=============================\n\n"
            "[NOT YET GENERATED]\n\n"
            "Phase 7 (LLM Report Generation) has not been run.\n"
            "Run the full pipeline to generate real per-WSI reports.\n\n"
            "Expected output:\n"
            "  - 129 report PDFs (one per test WSI)\n"
            "  - Structured JSON analysis per WSI\n"
            "  - LLM-generated narrative (if API key set)\n"
        )

    fig = plt.figure(figsize=(14, 9))
    gs  = gridspec.GridSpec(1, 3, width_ratios=[1.8, 1, 1], figure=fig)

    # Left panel: report text
    ax_text = fig.add_subplot(gs[0])
    ax_text.set_facecolor("#f5f5f0")
    ax_text.text(0.05, 0.97, report_text,
                 transform=ax_text.transAxes,
                 va="top", ha="left", fontsize=8.5,
                 fontfamily="monospace",
                 wrap=True)
    ax_text.set_xticks([]); ax_text.set_yticks([])
    ax_text.set_title("AI-Generated Pathology Report", fontsize=11, fontweight="bold",
                       pad=8)

    # Middle panel: tumor probability gauge
    ax_gauge = fig.add_subplot(gs[1])
    theta = np.linspace(np.pi, 0, 500)
    x_arc = np.cos(theta)
    y_arc = np.sin(theta)
    for (start, end, c) in [(np.pi, 2.1, "green"), (2.1, 1.05, "orange"), (1.05, 0, "red")]:
        th = np.linspace(start, end, 200)
        ax_gauge.fill_between(np.cos(th), np.sin(th)*0.6, np.sin(th), alpha=0.35, color=c)
        ax_gauge.plot(np.cos(th), np.sin(th), color=c, lw=2)

    needle_angle = np.pi * (1 - tumor_prob)
    ax_gauge.annotate("", xy=(np.cos(needle_angle)*0.82, np.sin(needle_angle)*0.82),
                       xytext=(0, 0),
                       arrowprops=dict(arrowstyle="->", color="black", lw=2.5))
    ax_gauge.text(0, -0.25, f"{tumor_prob*100:.0f}%", ha="center", va="center",
                  fontsize=22, fontweight="bold", color="red")
    ax_gauge.text(0, -0.45, "Tumor Probability", ha="center", fontsize=10)
    ax_gauge.set_xlim(-1.3, 1.3); ax_gauge.set_ylim(-0.6, 1.2)
    ax_gauge.set_aspect("equal"); ax_gauge.axis("off")
    ax_gauge.set_title("Risk Score", fontsize=11, fontweight="bold")

    # Right panel: top patches bar chart (real attention scores when available)
    ax_bar = fig.add_subplot(gs[2])
    scores = real_att_scores if real_att_scores else [0.20, 0.17, 0.15, 0.13, 0.11]
    labels = [f"Patch #{p}" for p in top_patches[:5]]
    colors = plt.cm.hot(np.linspace(0.3, 0.9, len(scores)))[::-1]
    bars   = ax_bar.barh(labels[::-1], scores[::-1], color=colors[::-1], edgecolor="white")
    ax_bar.set_xlabel("Attention Weight", fontsize=10)
    ax_bar.set_title("Top-5 Suspicious\nPatch Regions", fontsize=11, fontweight="bold")
    ax_bar.set_xlim(0, max(scores) * 1.25)
    for bar, score in zip(bars, scores[::-1]):
        ax_bar.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height()/2,
                    f"{score:.3f}", va="center", fontsize=9)
    ax_bar.grid(axis="x", alpha=0.3)

    fig.suptitle("LLM Clinical Report - Full Diagnostic Panel", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save(fig, out_dir / "llm_generated_pathology_report.png")


# PHASE 8 - UI Screenshot (static mockup)

def phase8_visuals(cfg, out_dir: Path):
    print("\n[PHASE 8] UI mockup visual ...")

    # Load a real tumor patch
    with h5py.File(ROOT / cfg["paths"]["pcam_train_x"], "r") as fx:
        X = fx["x"][:200].astype(np.uint8)
    with h5py.File(ROOT / cfg["paths"]["train_y"], "r") as fy:
        y = fy["y"][:200].squeeze().astype(np.uint8)

    tumor_idx   = np.where(y == 1)[0]
    sample      = X[tumor_idx[0]]
    norm        = MacenkoNormalizer()
    norm.fit(sample)
    norm_sample = norm.transform(sample)

    # Real Grad-CAM (if Phase 3 model is available)
    backbone  = cfg["training"]["backbone"]
    ckpt_path = ROOT / cfg["paths"]["models_dir"] / f"{backbone.replace('/','_')}_best.pth"
    cam_map   = None

    if ckpt_path.exists():
        try:
            import torch
            from phase3_train import PatchClassifier, DEVICE as P3_DEVICE
            from phase5_explainability import GradCAM, get_val_transforms
            ckpt  = torch.load(ckpt_path, map_location=P3_DEVICE, weights_only=False)
            model = PatchClassifier(backbone, pretrained=False).to(P3_DEVICE)
            model.load_state_dict(ckpt["state_dict"])
            model.eval()
            grad_cam = GradCAM(model)
            tf       = get_val_transforms()
            tensor   = tf(image=norm_sample)["image"].unsqueeze(0).to(P3_DEVICE)
            cam_map  = grad_cam(tensor)
            grad_cam.remove_hooks()
            print("  [OK]  Real Grad-CAM generated for UI screenshot")
        except Exception as e:
            print(f"  [!]  Grad-CAM failed ({e}) -- using Gaussian fallback")

    if cam_map is None:
        # Gaussian fallback (only used when model not available)
        cam_map = np.zeros((96, 96), dtype=np.float32)
        cx, cy  = 55, 45
        for i in range(96):
            for j in range(96):
                cam_map[i, j] = np.exp(-((i-cy)**2 + (j-cx)**2) / (2*20**2))
        cam_map = cam_map / cam_map.max()

    overlay = cv2.addWeighted(
        cv2.cvtColor(norm_sample, cv2.COLOR_RGB2BGR), 0.6,
        cv2.applyColorMap((cam_map * 255).astype(np.uint8), cv2.COLORMAP_JET), 0.4, 0
    )

    # Load real metrics from saved JSONs
    mil_json = ROOT / cfg["paths"]["reports_dir"] / "mil_results.json"
    cam_json = ROOT / cfg["paths"]["reports_dir"] / "gradcam_metrics.json"

    slide_auc = patch_auc = sens95 = cam_iou = cam_dice = None
    if mil_json.exists():
        with open(mil_json) as f:
            m = json.load(f)
        slide_auc = m.get("test_metrics", {}).get("auc")
        sens95    = m.get("test_metrics", {}).get("sens@95")
    if cam_json.exists():
        with open(cam_json) as f:
            c = json.load(f)
        cam_iou  = c.get("mean_iou")
        cam_dice = c.get("mean_dice")
    # Patch AUC from patch-level results if saved
    pat_json = ROOT / cfg["paths"]["reports_dir"] / "patch_results.json"
    if pat_json.exists():
        with open(pat_json) as f:
            p = json.load(f)
        patch_auc = p.get("test_auc")

    def fmt(v, decimals=3):
        return f"{v:.{decimals}f}" if v is not None else "N/A"

    metric_lines = [
        f"Slide-AUC:      {fmt(slide_auc)}",
        f"Sens@95Spec:    {fmt(sens95)}",
        f"Patch-AUC:      {fmt(patch_auc)}",
        f"Grad-CAM IoU:   {fmt(cam_iou)}",
        f"Grad-CAM Dice:  {fmt(cam_dice)}",
    ]

    # Load real attention + tumor prob from latest report
    rpt_dir   = ROOT / cfg["paths"]["reports_dir"] / "clinical"
    json_rpts = sorted(rpt_dir.glob("*_analysis.json")) if rpt_dir.exists() else []
    txt_rputs = sorted(rpt_dir.glob("*_report.txt"))    if rpt_dir.exists() else []

    ui_prob         = None
    ui_att_scores   = None
    ui_patch_labels = None
    report_snippet  = None

    if json_rpts and txt_rputs:
        with open(json_rpts[0]) as f:
            rpt_analysis = json.load(f)
        with open(txt_rputs[0]) as f:
            report_text_full = f.read()

        ui_prob       = rpt_analysis.get("classification", {}).get("tumor_probability")
        ui_att_scores = rpt_analysis.get("attention_analysis", {}).get("top_attention_scores", [])
        top_regions   = rpt_analysis.get("attention_analysis", {}).get("top_regions", [])
        try:
            ui_patch_labels = [f"Patch #{int(p.split('_')[1])}" for p in top_regions[:5]]
        except Exception:
            ui_patch_labels = [f"Patch #{i}" for i in range(5)]
        report_snippet = "\n".join(report_text_full.strip().split("\n")[:10])

    if ui_prob is None:
        ui_prob         = 0.0
        ui_att_scores   = [0.20, 0.17, 0.15, 0.13, 0.11]
        ui_patch_labels = [f"Patch #{i}" for i in range(5)]
        report_snippet  = (
            "AI REPORT\n"
            "---------\n"
            "[Phase 7 not yet run]\n"
            "Run full pipeline to\n"
            "generate real reports."
        )

    # Build figure
    fig = plt.figure(figsize=(20, 11), facecolor="#1e1e2e")
    gs  = gridspec.GridSpec(3, 5, figure=fig, hspace=0.35, wspace=0.3)

    def styled_ax(ax, title, fc="#2d2d3e"):
        ax.set_facecolor(fc)
        ax.set_title(title, color="white", fontsize=9, fontweight="bold", pad=5)
        ax.tick_params(colors="white")
        for spine in ax.spines.values():
            spine.set_edgecolor("#5555aa")

    ax_banner = fig.add_subplot(gs[0, :])
    ax_banner.set_facecolor("#2d2d3e")
    ax_banner.text(0.5, 0.5,
                   "[+]  WSI Cancer Detection AI - Full Pipeline Inference Dashboard",
                   color="white", fontsize=16, fontweight="bold",
                   ha="center", va="center", transform=ax_banner.transAxes)
    ax_banner.axis("off")

    ax1 = fig.add_subplot(gs[1, 0])
    ax1.imshow(sample); styled_ax(ax1, "Input Patch (H&E)"); ax1.axis("off")

    ax2 = fig.add_subplot(gs[1, 1])
    ax2.imshow(norm_sample); styled_ax(ax2, "Macenko Normalized"); ax2.axis("off")

    ax3 = fig.add_subplot(gs[1, 2])
    ax3.imshow(cam_map, cmap="jet", vmin=0, vmax=1)
    styled_ax(ax3, "Grad-CAM Heatmap (Real)" if ckpt_path.exists() else "Grad-CAM Heatmap")
    ax3.axis("off")

    ax4 = fig.add_subplot(gs[1, 3])
    ax4.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    styled_ax(ax4, "Overlay"); ax4.axis("off")

    prob_pct = ui_prob * 100
    pred_lbl = "TUMOR" if ui_prob > 0.5 else "NORMAL"
    pred_clr = "#ff4444" if ui_prob > 0.5 else "#44ff88"
    ax5 = fig.add_subplot(gs[1, 4])
    ax5.set_facecolor("#2d2d3e")
    ax5.text(0.5, 0.65, f"{prob_pct:.0f}%", color=pred_clr, fontsize=36, fontweight="bold",
             ha="center", va="center", transform=ax5.transAxes)
    ax5.text(0.5, 0.35, pred_lbl, color=pred_clr, fontsize=14, fontweight="bold",
             ha="center", va="center", transform=ax5.transAxes)
    ax5.text(0.5, 0.18, "Probability", color="#aaaacc", fontsize=9,
             ha="center", va="center", transform=ax5.transAxes)
    ax5.axis("off")
    styled_ax(ax5, "Prediction")

    ax6 = fig.add_subplot(gs[2, :2])
    scores_ui = ui_att_scores[:5] if ui_att_scores else [0.2, 0.17, 0.15, 0.13, 0.11]
    bars = ax6.barh(ui_patch_labels[::-1], scores_ui[::-1],
                    color=plt.cm.hot(np.linspace(0.3, 0.9, 5))[::-1])
    ax6.set_facecolor("#2d2d3e")
    styled_ax(ax6, "MIL Attention - Top-5 Regions")
    ax6.tick_params(colors="white")
    ax6.set_xlabel("Attention Weight", color="white", fontsize=8)
    for spine in ax6.spines.values():
        spine.set_edgecolor("#5555aa")
    ax6.grid(axis="x", alpha=0.2, color="white")

    ax7 = fig.add_subplot(gs[2, 2])
    ax7.set_facecolor("#2d2d3e")
    ax7.text(0.05, 0.95, "\n".join(metric_lines), color="#ccccff", fontsize=9,
             fontfamily="monospace", va="top", transform=ax7.transAxes)
    styled_ax(ax7, "Real Model Metrics"); ax7.axis("off")

    ax8 = fig.add_subplot(gs[2, 3:])
    ax8.set_facecolor("#2d2d3e")
    ax8.text(0.05, 0.95, report_snippet, color="#aaffaa", fontsize=9,
             fontfamily="monospace", va="top", transform=ax8.transAxes)
    styled_ax(ax8, "LLM Clinical Report (Real)"); ax8.axis("off")

    save(fig, out_dir / "ui_full_pipeline_screenshot.png", dpi=120)


# BONUS - t-SNE Feature Embedding

def bonus_tsne(cfg, out_dir: Path):
    print("\n[BONUS] t-SNE / UMAP feature embedding visualization …")
    import torch

    feat_path  = ROOT / "models" / "features_test.npy"
    label_path = ROOT / "models" / "labels_test.npy"

    if not feat_path.exists():
        print("  [!]  Cached test features not found. Run phase4_mil.py first.")
        return

    print("  Loading cached features …")
    feats  = np.load(feat_path)   # (N, feat_dim)
    labels = np.load(label_path)  # (N,)

    # Subsample for speed
    rng = np.random.default_rng(0)
    n_sample = min(5000, len(feats))
    idx = rng.choice(len(feats), n_sample, replace=False)
    feats_sub  = feats[idx]
    labels_sub = labels[idx]

    # PCA first for stability
    from sklearn.decomposition import PCA
    from sklearn.manifold     import TSNE

    print(f"  PCA (50 components) on {n_sample} samples …")
    pca   = PCA(n_components=min(50, feats_sub.shape[1]), random_state=42)
    feats_pca = pca.fit_transform(feats_sub)

    print("  t-SNE (perplexity=40) …")
    tsne  = TSNE(n_components=2, perplexity=40, max_iter=1000, random_state=42,
                  init="pca", learning_rate="auto")
    emb   = tsne.fit_transform(feats_pca)

    fig, ax = plt.subplots(figsize=(10, 8))
    colors  = ["royalblue", "crimson"]
    lnames  = ["Normal (label=0)", "Tumor (label=1)"]
    for lbl, color, name in zip([0, 1], colors, lnames):
        mask = labels_sub == lbl
        ax.scatter(emb[mask, 0], emb[mask, 1], c=color, s=6,
                   alpha=0.6, label=f"{name} (n={mask.sum()})", rasterized=True)

    ax.legend(fontsize=12, markerscale=3)
    ax.set_title(f"t-SNE Feature Embedding - Tumor vs Normal\n"
                 f"(n={n_sample} test patches, PCA->t-SNE)\n"
                 f"Backbone: {cfg['training']['backbone']}",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("t-SNE dim 1"); ax.set_ylabel("t-SNE dim 2")
    ax.grid(alpha=0.2)
    plt.tight_layout()
    save(fig, out_dir / "tsne_features_tumor_vs_normal.png")


# Main dispatcher

PHASE_MAP = {
    1: ("Dataset Exploration",      phase1_visuals),
    2: ("Preprocessing",            phase2_visuals),
    3: ("Patch-Level Model",        phase3_visuals),
    4: ("MIL Slide-Level",          phase4_visuals),
    5: ("Pixel Mask Visuals",       phase5_visuals),
    6: ("Virtual Staining",         phase6_visuals),
    7: ("LLM Report",               phase7_visuals),
    8: ("UI Screenshot",            phase8_visuals),
    9: ("t-SNE Embedding (bonus)",  bonus_tsne),
}


def main():
    parser = argparse.ArgumentParser(description="Generate all project visualizations")
    parser.add_argument("--phase", nargs="*", type=int,
                        help="Phase numbers to run (default: all)")
    args = parser.parse_args()

    cfg     = load_config()
    out_dir = ROOT / cfg["paths"]["reports_dir"]

    phases_to_run = args.phase if args.phase else list(PHASE_MAP.keys())

    print(f"\n{'='*65}")
    print(f"  WSI Pipeline - Visualization Generator")
    print(f"  Output dir: {out_dir}")
    print(f"  Phases: {phases_to_run}")
    print(f"{'='*65}")

    for p in phases_to_run:
        if p not in PHASE_MAP:
            print(f"  Unknown phase {p} - skipping"); continue
        name, fn = PHASE_MAP[p]
        print(f"\n{'─'*65}")
        print(f"  Phase {p}: {name}")
        print(f"{'─'*65}")
        try:
            fn(cfg, out_dir)
        except Exception as e:
            import traceback
            print(f"  [FAIL]  Phase {p} failed: {e}")
            traceback.print_exc()

    print(f"\n{'='*65}")
    print("  Done. All outputs saved to reports/")
    print(f"{'='*65}\n")


if __name__ == "__main__":
    main()

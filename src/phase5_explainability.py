# Phase 5 - Grad-CAM, ViT attention rollout, and MIL heatmap visualization

import sys
import json
from pathlib import Path
from typing import Optional, List, Tuple

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.cm as cm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase1_setup      import load_config, PCamDataset, load_masks
from phase2_preprocessing import get_val_transforms, MacenkoNormalizer
from phase3_train       import PatchClassifier, DEVICE, load_config
from phase4_mil         import GatedAttentionMIL, load_patch_model

ROOT = Path(__file__).resolve().parent.parent


# 1. Grad-CAM for CNN (EfficientNet / ResNet)

class GradCAM:
    """
    Gradient-weighted Class Activation Mapping.

    Works with any CNN that has a final convolutional layer.
    Auto-detects target layer for EfficientNet, ResNet, and ViT.
    """

    def __init__(self, model: PatchClassifier, target_layer: Optional[nn.Module] = None):
        self.model = model
        self.model.eval()
        self.gradients  : Optional[torch.Tensor] = None
        self.activations: Optional[torch.Tensor] = None

        # Auto-detect target layer
        if target_layer is None:
            target_layer = self._auto_target_layer()
        self.target_layer = target_layer

        # Register hooks
        self._fwd_hook = target_layer.register_forward_hook(self._save_activation)
        self._bwd_hook = target_layer.register_full_backward_hook(self._save_gradient)

    def _auto_target_layer(self) -> nn.Module:
        name = self.model.backbone_name.lower()
        backbone = self.model.backbone

        if "efficientnet" in name:
            # timm EfficientNet: last conv block
            return backbone.conv_head

        elif "resnet" in name:
            return backbone.layer4[-1].conv2

        elif "vit" in name:
            # For ViT, we use the last attention block
            return backbone.blocks[-1].norm1

        else:
            # Fallback: find last Conv2d
            last_conv = None
            for module in backbone.modules():
                if isinstance(module, nn.Conv2d):
                    last_conv = module
            if last_conv is None:
                raise ValueError("Could not auto-detect target layer.")
            return last_conv

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, image_tensor: torch.Tensor) -> np.ndarray:
        """
        Generate Grad-CAM heatmap for a single image.

        Args:
            image_tensor : (1, 3, H, W) normalized tensor on DEVICE

        Returns:
            heatmap : (H, W)  float32  values in [0, 1]
        """
        self.model.zero_grad()
        image_tensor = image_tensor.to(DEVICE).requires_grad_(True)

        logit = self.model(image_tensor)
        logit.backward()

        if self.gradients is None or self.activations is None:
            raise RuntimeError("Hooks did not fire — check target layer.")

        # Global average pooling of gradients → weights
        weights = self.gradients.mean(dim=[2, 3], keepdim=True)   # (1, C, 1, 1)
        cam     = (weights * self.activations).sum(dim=1).squeeze(0)  # (h, w)
        cam     = F.relu(cam)

        # Resize to input resolution
        h, w = image_tensor.shape[2:]
        cam   = cam.unsqueeze(0).unsqueeze(0)
        cam   = F.interpolate(cam, size=(h, w), mode="bilinear", align_corners=False)
        cam   = cam.squeeze().cpu().numpy()

        # Normalize to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max > cam_min:
            cam = (cam - cam_min) / (cam_max - cam_min)
        return cam.astype(np.float32)

    def remove_hooks(self):
        self._fwd_hook.remove()
        self._bwd_hook.remove()


def overlay_heatmap(
    image    : np.ndarray,
    heatmap  : np.ndarray,
    alpha    : float = 0.4,
    colormap : int   = cv2.COLORMAP_JET,
) -> np.ndarray:
    """
    Overlay a heatmap on an RGB image.

    Args:
        image   : (H, W, 3) uint8 RGB
        heatmap : (H, W)    float32 in [0, 1]
        alpha   : blending weight for heatmap

    Returns:
        overlay : (H, W, 3) uint8 BGR
    """
    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, colormap)   # BGR
    image_bgr     = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    overlay       = cv2.addWeighted(image_bgr, 1 - alpha, heatmap_color, alpha, 0)
    return overlay


# 2. ViT Attention Rollout

class ViTAttentionRollout:
    """
    Attention Rollout for Vision Transformers.

    Reference:
        Abnar & Zuidema, "Quantifying Attention Flow in Transformers", ACL 2020.

    Recursively multiplies attention matrices from all layers,
    producing a single spatial attention map.
    """

    def __init__(self, model: PatchClassifier, discard_ratio: float = 0.9):
        assert "vit" in model.backbone_name.lower(), "ViT model required"
        self.model        = model
        self.discard_ratio = discard_ratio
        self.attentions   : List[torch.Tensor] = []
        self._hooks       : list = []
        self._register_hooks()

    def _register_hooks(self):
        for block in self.model.backbone.blocks:
            hook = block.attn.register_forward_hook(self._save_attention)
            self._hooks.append(hook)

    def _save_attention(self, module, input, output):
        # timm ViT returns (x, attn) when attn_output=True — else just x
        # We need to access the raw attention matrix
        # Workaround: re-run attention computation
        # For standard timm ViT, use attn.attn_drop output
        pass    # handled in get_map via forward with output_attentions

    def get_map(self, image_tensor: torch.Tensor) -> np.ndarray:
        """
        Compute attention rollout map.

        Args:
            image_tensor : (1, 3, 224, 224)  — ViT expects 224×224

        Returns:
            rollout_map : (H_patches, W_patches)  float32 in [0, 1]
        """
        self.model.eval()
        B = image_tensor.shape[0]

        # timm ViT supports output_attentions via forward_features
        # We patch forward to capture attention matrices
        attn_maps = []

        def hook_fn(module, input, output):
            # The attention weight is computed inside timm's Attention module
            # We re-compute it from q, k without dropout
            with torch.no_grad():
                qkv = module.qkv(input[0])
                B_, N, C3 = qkv.shape
                qkv  = qkv.reshape(B_, N, 3, module.num_heads, C3 // 3 // module.num_heads)
                qkv  = qkv.permute(2, 0, 3, 1, 4)
                q, k, v = qkv.unbind(0)
                scale    = (C3 // 3 // module.num_heads) ** -0.5
                attn     = (q @ k.transpose(-2, -1)) * scale
                attn     = attn.softmax(dim=-1)           # (B, heads, N, N)
                attn_maps.append(attn.mean(dim=1).cpu())  # avg over heads → (B, N, N)

        hooks = []
        for block in self.model.backbone.blocks:
            hooks.append(block.attn.register_forward_hook(hook_fn))

        with torch.no_grad():
            _ = self.model.backbone(image_tensor.to(DEVICE))

        for h in hooks:
            h.remove()

        if not attn_maps:
            raise RuntimeError("Attention hooks did not fire.")

        # Rollout
        result = torch.eye(attn_maps[0].shape[-1])
        for attn in attn_maps:
            attn  = attn.squeeze(0)        # (N, N)
            # Add residual connection
            attn  = attn + torch.eye(attn.shape[-1])
            attn  = attn / attn.sum(dim=-1, keepdim=True)
            result = attn @ result

        # CLS token → spatial patches
        # result[0, 1:] = attention from CLS to all patch tokens
        n_patches = result.shape[-1] - 1
        side      = int(n_patches ** 0.5)
        mask = result[0, 1:].reshape(side, side).numpy()

        # Discard low-attention regions
        flat = mask.flatten()
        threshold = np.percentile(flat, self.discard_ratio * 100)
        mask = np.where(mask > threshold, mask, 0)

        # Normalize
        if mask.max() > 0:
            mask = mask / mask.max()
        return mask.astype(np.float32)

    def cleanup(self):
        for h in self._hooks:
            h.remove()


# 3. MIL Attention Visualization

def visualize_mil_attention(
    patch_images    : np.ndarray,
    attention_scores: np.ndarray,
    top_k           : int = 8,
    save_path       : Optional[str] = None,
):
    """
    Display patches ranked by MIL attention score.
    Red border = high attention (most suspicious).
    Blue border = low attention (benign).

    Args:
        patch_images    : (N, 96, 96, 3) uint8
        attention_scores: (N,) float — MIL attention weights
        top_k           : number of patches to show
    """
    sorted_idx = np.argsort(attention_scores)[::-1]

    n_show = min(top_k, len(sorted_idx))
    fig, axes = plt.subplots(2, n_show // 2, figsize=(n_show * 2, 5))
    axes = axes.ravel()
    fig.suptitle("MIL Attention — Patch Importance\n"
                 "(Red = high tumor probability, Blue = low)",
                 fontsize=13, fontweight="bold")

    for rank, (ax, pidx) in enumerate(zip(axes, sorted_idx[:n_show])):
        patch = patch_images[pidx]
        score = attention_scores[pidx]

        # Border color: interpolate blue→red by score
        norm_score = (score - attention_scores.min()) / (
            attention_scores.max() - attention_scores.min() + 1e-8)
        border_color = plt.cm.RdBu_r(norm_score)

        # Add colored border
        bordered = cv2.copyMakeBorder(patch, 5, 5, 5, 5, cv2.BORDER_CONSTANT,
                                       value=[int(c*255) for c in border_color[:3]])
        ax.imshow(bordered)
        ax.set_title(f"#{rank+1}  att={score:.4f}", fontsize=8,
                     color=("red" if norm_score > 0.5 else "blue"))
        ax.axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved MIL attention viz → {save_path}")
    plt.close()


# 4. Quantitative Grad-CAM evaluation vs pixel masks

def compute_cam_mask_metrics(
    cam      : np.ndarray,
    gt_mask  : np.ndarray,
    threshold: float = 0.5,
) -> dict:
    """
    Compare binarized Grad-CAM heatmap to ground-truth pixel mask.

    Args:
        cam     : (H, W) float32 in [0, 1]
        gt_mask : (H, W) bool / float
        threshold: binarization threshold for CAM

    Returns:
        dict: iou, dice, precision, recall
    """
    pred = (cam >= threshold).astype(np.float32)
    gt   = (gt_mask > 0).astype(np.float32)

    intersection = (pred * gt).sum()
    union        = (pred + gt).clip(0, 1).sum()
    iou          = float(intersection / (union + 1e-8))

    dice = float(2 * intersection / (pred.sum() + gt.sum() + 1e-8))

    tp = intersection
    fp = pred.sum() - tp
    fn = gt.sum() - tp
    precision = float(tp / (tp + fp + 1e-8))
    recall    = float(tp / (tp + fn + 1e-8))

    return {"iou": iou, "dice": dice, "precision": precision, "recall": recall}


def batch_cam_evaluation(
    model       : PatchClassifier,
    cfg         : dict,
    n_samples   : int = None,
    threshold   : float = 0.5,
) -> dict:
    """
    Compute mean IoU / Dice for Grad-CAM vs ground-truth masks.
    n_samples=None evaluates ALL available tumor patches in the training set.
    """
    grad_cam    = GradCAM(model)
    transform   = get_val_transforms()
    masks_all   = load_masks(cfg)   # (262144, 96, 96, 1)

    # Sample tumor patches (masks only available for train)
    import h5py
    with h5py.File(ROOT / cfg["paths"]["train_y"], "r") as f:
        y_all = f["y"][:].squeeze()
    tumor_idx = np.where(y_all == 1)[0]

    if n_samples is None or n_samples >= len(tumor_idx):
        sample_idx = tumor_idx          # ALL tumor patches
        n_samples  = len(tumor_idx)
    else:
        rng        = np.random.default_rng(42)   # fixed seed for reproducibility
        sample_idx = rng.choice(tumor_idx, size=n_samples, replace=False)
        sample_idx = np.sort(sample_idx)         # sorted for sequential h5py access (faster)

    with h5py.File(ROOT / cfg["paths"]["pcam_train_x"], "r") as fx:
        images = fx["x"][sample_idx]    # (n_samples, 96, 96, 3)

    all_iou, all_dice = [], []
    for i, (img, midx) in enumerate(zip(images, sample_idx)):
        tensor = transform(image=img)["image"].unsqueeze(0).to(DEVICE)
        cam    = grad_cam(tensor)
        mask   = masks_all[midx].squeeze()   # (96, 96)
        m      = compute_cam_mask_metrics(cam, mask, threshold)
        all_iou.append(m["iou"])
        all_dice.append(m["dice"])

    grad_cam.remove_hooks()
    n_evaluated = len(all_iou)
    results = {
        "mean_iou" : float(np.mean(all_iou)),
        "mean_dice": float(np.mean(all_dice)),
        "std_iou"  : float(np.std(all_iou)),
        "std_dice" : float(np.std(all_dice)),
        "n_samples": n_evaluated,
    }
    print(f"\n  Grad-CAM vs GT Mask  (n={n_evaluated:,})")
    print(f"    Mean IoU  : {results['mean_iou']:.4f} +/- {results['std_iou']:.4f}")
    print(f"    Mean Dice : {results['mean_dice']:.4f} +/- {results['std_dice']:.4f}")
    return results


# 5. Full visualization pipeline

def generate_explainability_grid(
    model       : PatchClassifier,
    images      : np.ndarray,
    labels      : np.ndarray,
    n           : int = 4,
    save_dir    : Optional[str] = None,
):
    """
    For each selected patch, show:
    col 1: Original patch
    col 2: Grad-CAM overlay
    col 3: Grad-CAM heatmap only
    """
    transform = get_val_transforms()
    grad_cam  = GradCAM(model)

    is_vit   = "vit" in model.backbone_name.lower()
    rollout  = ViTAttentionRollout(model) if is_vit else None

    # Select n tumor patches
    tumor_idx = np.where(labels == 1)[0][:n]
    fig, axes = plt.subplots(n, 3 if not is_vit else 4,
                              figsize=(15 if not is_vit else 20, n * 4))
    if n == 1:
        axes = axes[np.newaxis, :]
    col_titles = ["Original", "Grad-CAM Overlay", "Heatmap"]
    if is_vit:
        col_titles.append("ViT Attention Rollout")

    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=12, fontweight="bold")

    for row, pidx in enumerate(tumor_idx):
        img    = images[pidx]
        tensor = transform(image=img)["image"].unsqueeze(0)

        # Grad-CAM
        cam     = grad_cam(tensor.clone())
        overlay = overlay_heatmap(img, cam)

        axes[row, 0].imshow(img)
        axes[row, 0].set_ylabel(f"Patch {pidx}", fontsize=9)
        axes[row, 0].axis("off")

        axes[row, 1].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        axes[row, 1].axis("off")

        axes[row, 2].imshow(cam, cmap="jet", vmin=0, vmax=1)
        axes[row, 2].axis("off")

        # ViT rollout
        if is_vit:
            inp_224 = F.interpolate(tensor.to(DEVICE), 224, mode="bilinear",
                                     align_corners=False)
            rollout_map = rollout.get_map(inp_224)
            rollout_up  = cv2.resize(rollout_map, (96, 96))
            axes[row, 3].imshow(rollout_up, cmap="hot", vmin=0, vmax=1)
            axes[row, 3].axis("off")

    plt.tight_layout()
    if save_dir:
        Path(save_dir).mkdir(parents=True, exist_ok=True)
        save_path = Path(save_dir) / "explainability_grid.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Saved grid → {save_path}")
    plt.close()

    grad_cam.remove_hooks()
    if rollout:
        rollout.cleanup()


# Entry point

if __name__ == "__main__":
    import h5py

    cfg   = load_config()
    model = load_patch_model(cfg)
    model.eval()

    reports_dir = ROOT / cfg["paths"]["reports_dir"] / "heatmaps"
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Load a sample of training patches
    print("Loading sample patches …")
    with h5py.File(ROOT / cfg["paths"]["pcam_train_x"], "r") as f:
        images = f["x"][:500]
    import h5py as h5
    with h5.File(ROOT / cfg["paths"]["train_y"], "r") as f:
        labels = f["y"][:500].squeeze()

    # 1. Generate explainability grid
    print("\n[1] Generating explainability visualization grid …")
    generate_explainability_grid(model, images, labels, n=4,
                                  save_dir=str(reports_dir))

    # 2. Quantitative Grad-CAM evaluation
    print("\n[2] Quantitative Grad-CAM vs GT mask evaluation …")
    metrics = batch_cam_evaluation(model, cfg, n_samples=100)
    metrics_path = ROOT / cfg["paths"]["reports_dir"] / "gradcam_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Metrics saved → {metrics_path}")

    print("\n✓ Phase 5 explainability complete.")

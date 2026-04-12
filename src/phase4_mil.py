# Phase 4 - gated attention MIL for slide-level cancer prediction (Ilse et al., 2018)

import sys
import json
import time
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    roc_curve, confusion_matrix, cohen_kappa_score, matthews_corrcoef,
)
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase1_setup      import load_config, PCamDataset
from phase2_preprocessing import get_val_transforms
from phase3_train       import PatchClassifier, extract_features, DEVICE, set_seed, stable_sigmoid

ROOT = Path(__file__).resolve().parent.parent


# Bag (WSI) Dataset

class WSIBagDataset(Dataset):
    """
    Each item = one WSI bag (variable-length list of patch feature vectors).

    Args:
        features  : np.ndarray  (N_patches, feat_dim)  — pre-extracted
        labels    : np.ndarray  (N_patches,)           — patch labels
        metadata  : pd.DataFrame with 'wsi' column     — WSI provenance
        split     : used for display only
    """

    def __init__(
        self,
        features : np.ndarray,
        labels   : np.ndarray,
        metadata : pd.DataFrame,
        split    : str = "train",
    ):
        self.split = split
        bags       = {}   # wsi_id → {"features": [...], "label": int}

        for i, row in metadata.iterrows():
            wsi_id = row["wsi"]
            if wsi_id not in bags:
                bags[wsi_id] = {"features": [], "labels": []}
            bags[wsi_id]["features"].append(features[i])
            bags[wsi_id]["labels"].append(int(labels[i]))

        self.wsi_ids    = list(bags.keys())
        self.bag_feats  = [np.stack(bags[w]["features"]) for w in self.wsi_ids]
        # Bag label = 1 if ANY patch in bag is tumor
        self.bag_labels = [int(any(bags[w]["labels"])) for w in self.wsi_ids]

        n_pos = sum(self.bag_labels)
        n_neg = len(self.bag_labels) - n_pos
        print(f"  [{split.upper()} Bags]  total={len(self.wsi_ids)}  "
              f"tumor={n_pos}  normal={n_neg}")

    def __len__(self) -> int:
        return len(self.wsi_ids)

    def __getitem__(self, idx: int):
        feats = torch.tensor(self.bag_feats[idx], dtype=torch.float32)
        label = torch.tensor(self.bag_labels[idx], dtype=torch.float32)
        return feats, label   # feats: (n_patches_in_bag, feat_dim)


def collate_bags(batch):
    """Custom collate: bags have variable patch counts — return lists."""
    feats, labels = zip(*batch)
    return list(feats), torch.stack(labels)


# Gated Attention MIL Model

class GatedAttentionMIL(nn.Module):
    """
    Attention-based MIL (Ilse et al., "Attention-based Deep MIL", ICML 2018).

    Gated attention allows the model to selectively focus on important patches.

    Architecture:
        patch_feats  →  projection  →  (U, V branches)  →  softmax attention
        →  weighted aggregation  →  classifier
    """

    def __init__(
        self,
        feat_dim    : int = 512,
        hidden_dim  : int = 256,
        attention_dim: int = 128,
        dropout     : float = 0.25,
    ):
        super().__init__()

        # Feature projection
        self.projection = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Gated attention branches
        self.attention_V = nn.Sequential(
            nn.Linear(hidden_dim, attention_dim),
            nn.Tanh(),
        )
        self.attention_U = nn.Sequential(
            nn.Linear(hidden_dim, attention_dim),
            nn.Sigmoid(),
        )
        self.attention_w = nn.Linear(attention_dim, 1, bias=False)

        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
        )

    def forward(
        self, bag: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            bag : (n_patches, feat_dim)

        Returns:
            logit       : scalar  (before sigmoid)
            attention   : (n_patches,)  normalized attention weights
        """
        h = self.projection(bag)           # (n, hidden)

        # Gated attention
        A_V = self.attention_V(h)          # (n, att_dim)
        A_U = self.attention_U(h)          # (n, att_dim)
        A   = self.attention_w(A_V * A_U)  # (n, 1)
        A   = F.softmax(A, dim=0)          # (n, 1)  — normalized

        # Weighted bag representation
        M   = (A * h).sum(dim=0)           # (hidden,)

        logit = self.classifier(M).squeeze(-1)  # scalar
        return logit, A.squeeze(1)              # (scalar, n_patches)

    def predict_bag(
        self, bag: torch.Tensor, device: torch.device
    ) -> Tuple[float, np.ndarray]:
        """Convenience: single-bag inference → probability + attention array."""
        self.eval()
        with torch.no_grad():
            bag = bag.to(device)
            logit, att = self(bag)
            prob = torch.sigmoid(logit).item()
            att  = att.cpu().numpy()
        return prob, att


# Sensitivity @ specificity helper

def sensitivity_at_specificity(
    y_true: np.ndarray,
    y_score: np.ndarray,
    target_specificity: float = 0.95,
) -> float:
    fpr, tpr, thresholds = roc_curve(y_true, y_score)
    specificity = 1 - fpr
    # Find operating point closest to target specificity
    idx = np.argmin(np.abs(specificity - target_specificity))
    return float(tpr[idx])


# Train / eval loops

def _mil_metrics(logits_arr: np.ndarray, labels_arr: np.ndarray,
                  threshold: float = 0.5) -> dict:
    """Full publication-quality metrics for slide-level MIL evaluation."""
    probs = stable_sigmoid(logits_arr)
    preds = (probs >= threshold).astype(int)

    cm = confusion_matrix(labels_arr, preds)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, int(labels_arr.sum()))
    sens = tp / (tp + fn + 1e-8)
    spec = tn / (tn + fp + 1e-8)

    fpr, tpr, _ = roc_curve(labels_arr, probs)
    spec_arr    = 1 - fpr
    sens_at_95  = float(tpr[np.argmin(np.abs(spec_arr - 0.95))])
    sens_at_99  = float(tpr[np.argmin(np.abs(spec_arr - 0.99))])

    return {
        "auc"          : float(roc_auc_score(labels_arr, probs)),
        "auc_pr"       : float(average_precision_score(labels_arr, probs)),
        "f1"           : float(f1_score(labels_arr, preds, zero_division=0)),
        "sensitivity"  : float(sens),
        "specificity"  : float(spec),
        "kappa"        : float(cohen_kappa_score(labels_arr, preds)),
        "mcc"          : float(matthews_corrcoef(labels_arr, preds)),
        "sens@95"      : sens_at_95,
        "sens@99"      : sens_at_99,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def _mil_bootstrap_ci(logits_arr: np.ndarray, labels_arr: np.ndarray,
                       n_iter: int = 1000, ci: float = 0.95,
                       threshold: float = 0.5, seed: int = 42) -> dict:
    """Bootstrap confidence intervals for slide-level metrics."""
    rng     = np.random.default_rng(seed)
    n       = len(labels_arr)
    records = []
    for _ in range(n_iter):
        idx = rng.integers(0, n, size=n)
        try:
            m = _mil_metrics(logits_arr[idx], labels_arr[idx], threshold)
            records.append(m)
        except Exception:
            continue
    alpha = 1 - ci
    ci_out = {}
    for key in ["auc", "auc_pr", "f1", "sensitivity", "specificity",
                 "kappa", "mcc", "sens@95", "sens@99"]:
        vals = sorted(r[key] for r in records)
        ci_out[key] = {
            "mean": float(np.mean(vals)),
            "lo"  : vals[int(alpha / 2 * len(vals))],
            "hi"  : vals[int((1 - alpha / 2) * len(vals))],
            "ci"  : ci,
        }
    return ci_out


def train_mil_epoch(
    model        : GatedAttentionMIL,
    dataset      : WSIBagDataset,
    optimizer    : optim.Optimizer,
    criterion    : nn.Module,
    device       : torch.device,
    grad_clip    : float = 1.0,
    threshold    : float = 0.5,
) -> dict:
    """MIL training -- one bag at a time (variable bag size)."""
    model.train()
    indices    = torch.randperm(len(dataset)).tolist()
    total_loss = 0.0
    all_logits, all_labels = [], []

    for idx in indices:
        bag, label = dataset[idx]
        bag   = bag.to(device)
        label = label.to(device).unsqueeze(0)

        optimizer.zero_grad()
        logit, _ = model(bag)
        loss = criterion(logit.unsqueeze(0), label)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()

        total_loss += loss.item()
        all_logits.append(logit.item())
        all_labels.append(label.item())

    logits_arr = np.array(all_logits)
    labels_arr = np.array(all_labels)
    m          = _mil_metrics(logits_arr, labels_arr, threshold)
    m["loss"]  = total_loss / len(dataset)
    return m


@torch.no_grad()
def eval_mil(
    model     : GatedAttentionMIL,
    dataset   : WSIBagDataset,
    criterion : nn.Module,
    device    : torch.device,
    threshold : float = 0.5,
) -> dict:
    model.eval()
    total_loss = 0.0
    all_logits, all_labels, all_wsi_ids = [], [], []

    for idx in range(len(dataset)):
        bag, label = dataset[idx]
        bag_d      = bag.to(device)
        label_t    = label.to(device).unsqueeze(0)

        logit, _ = model(bag_d)
        loss = criterion(logit.unsqueeze(0), label_t)
        total_loss += loss.item()
        all_logits.append(logit.item())
        all_labels.append(label.item())
        all_wsi_ids.append(dataset.wsi_ids[idx])

    logits_arr = np.array(all_logits)
    labels_arr = np.array(all_labels)
    m          = _mil_metrics(logits_arr, labels_arr, threshold)
    m["loss"]  = total_loss / len(dataset)
    m["probs"] = stable_sigmoid(logits_arr).tolist()
    m["labels"]  = labels_arr.tolist()
    m["wsi_ids"] = all_wsi_ids
    return m


# Feature extraction from trained Phase 3 model

def load_patch_model(cfg: dict, backbone: str = None) -> PatchClassifier:
    backbone   = backbone or cfg["training"]["backbone"]
    ckpt_path  = ROOT / cfg["paths"]["models_dir"] / f"{backbone.replace('/', '_')}_best.pth"
    assert ckpt_path.exists(), (
        f"No checkpoint found at {ckpt_path}. Run phase3_train.py first.")

    ckpt  = torch.load(ckpt_path, map_location=DEVICE)
    model = PatchClassifier(backbone, pretrained=False).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print(f"  Loaded patch model: {backbone}  (val_auc={ckpt['best_val_auc']:.4f})")
    return model


def extract_all_features(
    cfg: dict,
    patch_model: PatchClassifier,
    split: str = "train",
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract features for every patch in a split."""
    t_cfg   = cfg["training"]
    dataset = PCamDataset(split, transform=get_val_transforms(), cfg=cfg)
    loader  = DataLoader(
        dataset,
        batch_size=t_cfg["batch_size"] * 2,
        shuffle=False,
        num_workers=t_cfg["num_workers"],
        pin_memory=True,
    )
    print(f"  Extracting features for {split} ({len(dataset):,} patches) …")
    feats, labels = extract_features(patch_model, loader, DEVICE)
    print(f"  Features shape: {feats.shape}")

    # Cache to disk
    cache_dir = ROOT / "models"
    np.save(cache_dir / f"features_{split}.npy", feats)
    np.save(cache_dir / f"labels_{split}.npy",   labels)
    return feats, labels


def _model_hash(patch_model: PatchClassifier) -> str:
    """Short hash of model weights -- used to invalidate stale feature caches."""
    import hashlib
    buf = b""
    for p in patch_model.parameters():
        buf += p.data.cpu().numpy().tobytes()[:256]   # sample first 256 bytes
    return hashlib.md5(buf).hexdigest()[:8]


def load_or_extract_features(
    cfg: dict,
    patch_model: PatchClassifier,
    split: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load cached features if available AND from same model checkpoint.
    Re-extracts if model has changed (hash mismatch) to prevent stale features.
    """
    cache_dir  = ROOT / "models"
    feat_path  = cache_dir / f"features_{split}.npy"
    label_path = cache_dir / f"labels_{split}.npy"
    hash_path  = cache_dir / f"features_{split}_model_hash.txt"

    current_hash = _model_hash(patch_model)

    if feat_path.exists() and label_path.exists() and hash_path.exists():
        saved_hash = hash_path.read_text().strip()
        if saved_hash == current_hash:
            print(f"  Loading cached features for {split} (model hash OK: {current_hash})")
            return np.load(feat_path), np.load(label_path)
        else:
            print(f"  [!] Feature cache for {split} is STALE "
                  f"(saved={saved_hash}, current={current_hash}) -- re-extracting")

    feats, labels = extract_all_features(cfg, patch_model, split)
    hash_path.write_text(current_hash)
    return feats, labels


# Main MIL training orchestrator

def train_mil(cfg: dict):
    m_cfg     = cfg["mil"]
    t_cfg     = cfg["training"]
    threshold = m_cfg.get("threshold", 0.5)
    grad_clip = t_cfg.get("grad_clip_norm", 1.0)
    set_seed(t_cfg["seed"])

    print(f"\n{'='*60}")
    print(f"  Phase 4 - Attention-Based MIL")
    print(f"  Device    : {DEVICE}")
    print(f"  Epochs    : {m_cfg['epochs']}  (patience={m_cfg.get('early_stopping_patience',15)})")
    print(f"  LR        : {m_cfg['lr']}  WD: {m_cfg.get('weight_decay',1e-5)}")
    print(f"{'='*60}")

    from phase1_setup import load_metadata

    # 1. Phase 3 model
    patch_model = load_patch_model(cfg)
    feat_dim    = patch_model.backbone.num_features

    # 2. Features (with cache versioning)
    train_feats, train_labels = load_or_extract_features(cfg, patch_model, "train")
    valid_feats, valid_labels = load_or_extract_features(cfg, patch_model, "valid")
    test_feats,  test_labels  = load_or_extract_features(cfg, patch_model, "test")

    # 3. Metadata & bag datasets
    train_ds = PCamDataset("train", transform=get_val_transforms(), cfg=cfg)
    valid_ds = PCamDataset("valid", transform=get_val_transforms(), cfg=cfg)
    test_ds  = PCamDataset("test",  transform=get_val_transforms(), cfg=cfg)

    train_meta = load_metadata(cfg, "train", indices=train_ds.indices)
    valid_meta = load_metadata(cfg, "valid", indices=valid_ds.indices)
    test_meta  = load_metadata(cfg, "test",  indices=test_ds.indices)

    train_bags = WSIBagDataset(train_feats, train_labels, train_meta, "train")
    valid_bags = WSIBagDataset(valid_feats, valid_labels, valid_meta, "valid")
    test_bags  = WSIBagDataset(test_feats,  test_labels,  test_meta,  "test")

    # Log bag size statistics
    bag_sizes = [len(b) for b in train_bags.bag_feats]
    print(f"\n  Bag sizes - min={min(bag_sizes)}  max={max(bag_sizes)}  "
          f"mean={np.mean(bag_sizes):.1f}  median={np.median(bag_sizes):.1f}")

    # 4. MIL model
    mil_model = GatedAttentionMIL(
        feat_dim      = feat_dim,
        hidden_dim    = m_cfg.get("hidden_dim", m_cfg["feature_dim"]),
        attention_dim = m_cfg["attention_dim"],
        dropout       = m_cfg.get("dropout", 0.25),
    ).to(DEVICE)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(mil_model.parameters(),
                           lr=m_cfg["lr"],
                           weight_decay=m_cfg.get("weight_decay", 1e-5))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=m_cfg["epochs"]
    )

    # 5. Training loop
    best_auc       = 0.0
    patience_count = 0
    patience_max   = m_cfg.get("early_stopping_patience", 15)
    history        = []
    model_dir      = ROOT / cfg["paths"]["models_dir"]
    model_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path      = model_dir / "mil_best.pth"

    print(f"\n  MIL Training: {len(train_bags)} train WSIs | "
          f"{len(valid_bags)} val | {len(test_bags)} test")

    for epoch in range(1, m_cfg["epochs"] + 1):
        t0 = time.time()

        train_m = train_mil_epoch(mil_model, train_bags, optimizer, criterion,
                                  DEVICE, grad_clip, threshold)
        valid_m = eval_mil(mil_model, valid_bags, criterion, DEVICE, threshold)
        scheduler.step()

        elapsed = time.time() - t0
        print(
            f"  Ep {epoch:03d}/{m_cfg['epochs']}  "
            f"| Tr loss={train_m['loss']:.4f} auc={train_m['auc']:.4f} "
            f"auc_pr={train_m['auc_pr']:.4f} sens={train_m['sensitivity']:.3f} "
            f"| Val loss={valid_m['loss']:.4f} auc={valid_m['auc']:.4f} "
            f"auc_pr={valid_m['auc_pr']:.4f} sens@95={valid_m['sens@95']:.3f} "
            f"| t={elapsed:.0f}s",
            flush=True,
        )

        skip_keys = ("probs", "labels", "wsi_ids", "confusion_matrix")
        history.append({
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_m.items() if k not in skip_keys},
            **{f"val_{k}": v for k, v in valid_m.items() if k not in skip_keys},
        })

        if valid_m["auc"] > best_auc:
            best_auc       = valid_m["auc"]
            patience_count = 0
            torch.save({
                "epoch"      : epoch,
                "state_dict" : mil_model.state_dict(),
                "feat_dim"   : feat_dim,
                "val_auc"    : best_auc,
                "val_auc_pr" : valid_m["auc_pr"],
                "val_sens95" : valid_m["sens@95"],
                "cfg"        : m_cfg,
            }, ckpt_path)
            print(f"    [BEST] val_auc={best_auc:.4f}  "
                  f"val_auc_pr={valid_m['auc_pr']:.4f}  "
                  f"val_sens@95={valid_m['sens@95']:.4f}", flush=True)
        else:
            patience_count += 1
            if patience_count >= patience_max:
                print(f"\n  Early stopping at epoch {epoch} "
                      f"(no AUC gain for {patience_count} epochs)", flush=True)
                break

    # Save history CSV
    import csv
    reports_dir = ROOT / cfg["paths"]["reports_dir"]
    reports_dir.mkdir(parents=True, exist_ok=True)
    csv_path = reports_dir / "mil_training_history.csv"
    if history:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=history[0].keys())
            writer.writeheader()
            writer.writerows(history)

    # 6. Final test evaluation
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    mil_model.load_state_dict(ckpt["state_dict"])
    test_m = eval_mil(mil_model, test_bags, criterion, DEVICE, threshold)

    # Bootstrap CI
    test_logits_arr = np.log(
        np.clip(np.array(test_m["probs"]), 1e-7, 1 - 1e-7) /
        np.clip(1 - np.array(test_m["probs"]), 1e-7, 1)
    )   # convert probs back to logits for bootstrap
    test_labels_arr = np.array(test_m["labels"])
    print(f"\n  Computing bootstrap 95% CI ({m_cfg.get('bootstrap_n_iter',1000)} iters) ...",
          flush=True)
    ci = _mil_bootstrap_ci(
        test_logits_arr, test_labels_arr,
        n_iter    = m_cfg.get("bootstrap_n_iter", 1000),
        ci        = m_cfg.get("bootstrap_ci", 0.95),
        threshold = threshold,
        seed      = t_cfg["seed"],
    )

    print(f"\n{'='*60}")
    print(f"  FINAL MIL TEST RESULTS")
    print(f"  Slide-AUC       : {test_m['auc']:.4f}  "
          f"[{ci['auc']['lo']:.4f}, {ci['auc']['hi']:.4f}] 95% CI")
    print(f"  Slide-AUC-PR    : {test_m['auc_pr']:.4f}  "
          f"[{ci['auc_pr']['lo']:.4f}, {ci['auc_pr']['hi']:.4f}] 95% CI")
    print(f"  Slide-F1        : {test_m['f1']:.4f}  "
          f"[{ci['f1']['lo']:.4f}, {ci['f1']['hi']:.4f}] 95% CI")
    print(f"  Sensitivity     : {test_m['sensitivity']:.4f}  "
          f"[{ci['sensitivity']['lo']:.4f}, {ci['sensitivity']['hi']:.4f}] 95% CI")
    print(f"  Specificity     : {test_m['specificity']:.4f}  "
          f"[{ci['specificity']['lo']:.4f}, {ci['specificity']['hi']:.4f}] 95% CI")
    print(f"  Sens@95Spec     : {test_m['sens@95']:.4f}  "
          f"[{ci['sens@95']['lo']:.4f}, {ci['sens@95']['hi']:.4f}] 95% CI")
    print(f"  Sens@99Spec     : {test_m['sens@99']:.4f}")
    print(f"  Kappa           : {test_m['kappa']:.4f}")
    print(f"  MCC             : {test_m['mcc']:.4f}")
    cm = test_m["confusion_matrix"]
    print(f"  Confusion Matrix:")
    print(f"    TN={cm['tn']}  FP={cm['fp']}")
    print(f"    FN={cm['fn']}  TP={cm['tp']}")

    # Per-WSI predictions (FP/FN analysis)
    probs_arr    = np.array(test_m["probs"])
    wsi_ids_list = test_m["wsi_ids"]
    per_wsi_diag = []
    for wsi_id, prob, true_lbl in zip(wsi_ids_list, probs_arr, test_labels_arr):
        pred = int(prob >= threshold)
        outcome = ("TP" if pred == 1 and true_lbl == 1 else
                   "TN" if pred == 0 and true_lbl == 0 else
                   "FP" if pred == 1 and true_lbl == 0 else "FN")
        per_wsi_diag.append({"wsi_id": str(wsi_id), "prob": round(float(prob), 4),
                              "true": int(true_lbl), "pred": pred, "outcome": outcome})
    # Save per-WSI diagnostics
    diag_path = reports_dir / "mil_per_wsi_diagnostics.json"
    with open(diag_path, "w") as f:
        json.dump(per_wsi_diag, f, indent=2)
    fp_wsis = [d["wsi_id"] for d in per_wsi_diag if d["outcome"] == "FP"]
    fn_wsis = [d["wsi_id"] for d in per_wsi_diag if d["outcome"] == "FN"]
    print(f"\n  FP WSIs ({len(fp_wsis)}): {fp_wsis[:5]}{'...' if len(fp_wsis)>5 else ''}")
    print(f"  FN WSIs ({len(fn_wsis)}): {fn_wsis[:5]}{'...' if len(fn_wsis)>5 else ''}")
    print(f"  Per-WSI diagnostics -> {diag_path}")
    print(f"{'='*60}")

    skip_keys = ("probs", "labels", "wsi_ids")
    results = {
        "best_val_auc" : best_auc,
        "test_metrics" : {k: v for k, v in test_m.items() if k not in skip_keys},
        "bootstrap_ci" : ci,
        "history"      : history,
        "threshold"    : threshold,
        "n_test_wsis"  : len(test_bags),
    }
    results_path = reports_dir / "mil_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results -> {results_path}")

    return mil_model, results


# Inference on a single bag (used by Phase 8 UI)

def predict_wsi_bag(
    patch_images : np.ndarray,
    patch_model  : PatchClassifier,
    mil_model    : GatedAttentionMIL,
    transform    = None,
    device       : torch.device = DEVICE,
) -> Dict:
    """
    End-to-end WSI prediction from raw patch images.

    Args:
        patch_images : (N, 96, 96, 3) uint8
        patch_model  : trained CNN/ViT from Phase 3
        mil_model    : trained Attention-MIL
        transform    : val transforms

    Returns:
        dict with tumor_probability, attention_scores, top_patches
    """
    if transform is None:
        transform = get_val_transforms()

    # Extract features
    patch_model.eval()
    feats = []
    for patch in patch_images:
        t = transform(image=patch)["image"].unsqueeze(0).to(device)
        with torch.no_grad():
            feat = patch_model.get_features(t)
        feats.append(feat.cpu())
    feats_tensor = torch.cat(feats, dim=0)    # (N, feat_dim)

    # MIL prediction
    prob, attention = mil_model.predict_bag(feats_tensor, device)

    # Top-5 most attended patches
    top_idx = np.argsort(attention)[::-1][:5].tolist()

    return {
        "tumor_probability" : round(prob, 4),
        "attention_scores"  : attention.tolist(),
        "top_patch_indices" : top_idx,
        "n_patches"         : len(patch_images),
    }


# Per-WSI predictions for ALL test slides (used by Phase 7)

def predict_all_test_wsis(cfg: dict) -> list:
    """
    Load trained Phase 3 CNN + Phase 4 MIL model and run per-WSI predictions
    on every test WSI.  Requires Phase 3 and Phase 4 checkpoints.

    Returns:
        List of dicts (one per test WSI):
            wsi_id, tumor_probability, attention_scores,
            top_patch_indices, n_patches, true_label
    """
    from phase1_setup import load_metadata

    # Phase 3 model
    patch_model = load_patch_model(cfg)
    feat_dim    = patch_model.backbone.num_features

    # Phase 4 MIL model
    ckpt_path = ROOT / cfg["paths"]["models_dir"] / "mil_best.pth"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"MIL checkpoint not found: {ckpt_path}\n"
            "Phase 4 must complete successfully before Phase 7."
        )
    m_cfg = cfg["mil"]
    mil_model = GatedAttentionMIL(
        feat_dim      = feat_dim,
        hidden_dim    = m_cfg.get("hidden_dim", m_cfg["feature_dim"]),
        attention_dim = m_cfg["attention_dim"],
    ).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    mil_model.load_state_dict(ckpt["state_dict"])
    mil_model.eval()
    print(f"  Loaded MIL model  (val_auc={ckpt.get('val_auc', 0):.4f})", flush=True)

    # Features (use Phase 4 cache if available)
    test_feats, test_labels = load_or_extract_features(cfg, patch_model, "test")
    test_ds                 = PCamDataset("test", transform=get_val_transforms(), cfg=cfg)
    test_meta               = load_metadata(cfg, "test", indices=test_ds.indices)

    # Build WSI bags
    test_bags = WSIBagDataset(test_feats, test_labels, test_meta, "test")

    print(f"  Running per-WSI MIL inference on {len(test_bags)} test WSIs ...",
          flush=True)

    predictions = []
    for idx in range(len(test_bags)):
        wsi_id     = test_bags.wsi_ids[idx]
        bag_feats  = torch.tensor(test_bags.bag_feats[idx], dtype=torch.float32)
        true_label = test_bags.bag_labels[idx]

        prob, attention = mil_model.predict_bag(bag_feats, DEVICE)
        top_idx = np.argsort(attention)[::-1][:5].tolist()

        predictions.append({
            "wsi_id"           : str(wsi_id),
            "tumor_probability": round(prob, 4),
            "attention_scores" : attention.tolist(),
            "top_patch_indices": top_idx,
            "n_patches"        : len(test_bags.bag_feats[idx]),
            "true_label"       : true_label,
        })

        if (idx + 1) % 25 == 0 or idx == 0:
            lbl = "TUMOR" if prob > 0.5 else "NORMAL"
            print(f"  [{idx+1:3d}/{len(test_bags)}] {wsi_id}: prob={prob:.4f} ({lbl})",
                  flush=True)

    # Save all predictions to disk for downstream use
    out_path = ROOT / cfg["paths"]["reports_dir"] / "wsi_predictions.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(predictions, f, indent=2)
    print(f"  Per-WSI predictions saved --> {out_path}", flush=True)

    return predictions


# Entry point

if __name__ == "__main__":
    cfg = load_config()
    train_mil(cfg)

# Phase 3 - patch-level EfficientNet-B3 training with AMP, cosine LR, and early stopping

import argparse
import os
import sys
import time
import json
import random
import platform
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast
import timm

from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    accuracy_score, confusion_matrix, cohen_kappa_score,
    matthews_corrcoef, roc_curve, precision_recall_curve,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase1_setup        import load_config, PCamDataset
from phase2_preprocessing import get_train_transforms, get_val_transforms

ROOT   = Path(__file__).resolve().parent.parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def stable_sigmoid(logits: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid for metric computation."""
    logits = np.asarray(logits, dtype=np.float64)
    logits = np.nan_to_num(logits, nan=0.0, posinf=60.0, neginf=-60.0)
    logits = np.clip(logits, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-logits))


# Reproducibility

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# Environment metadata (for reproducibility logging)

def get_env_metadata() -> dict:
    import subprocess
    gpu_name = "N/A"
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(ROOT), stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        git_hash = "N/A"
    return {
        "python_version": platform.python_version(),
        "torch_version":  torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version":   torch.version.cuda or "N/A",
        "gpu":            gpu_name,
        "platform":       platform.platform(),
        "git_hash":       git_hash,
        "device":         str(DEVICE),
    }


# Model factory

def build_model(backbone: str, pretrained: bool = True, num_classes: int = 1) -> nn.Module:
    print(f"  Building model: {backbone}  (pretrained={pretrained})")
    return timm.create_model(backbone, pretrained=pretrained, num_classes=num_classes)


class PatchClassifier(nn.Module):
    """Thin wrapper around a timm backbone; handles ViT input resize."""

    def __init__(self, backbone: str, pretrained: bool = True):
        super().__init__()
        self.is_vit        = "vit" in backbone.lower()
        self.backbone      = build_model(backbone, pretrained, num_classes=0)
        self.classifier    = nn.Linear(self.backbone.num_features, 1)
        self.backbone_name = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_vit and x.shape[-1] != 224:
            x = nn.functional.interpolate(x, size=224, mode="bilinear", align_corners=False)
        return self.classifier(self.backbone(x)).squeeze(1)

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        if self.is_vit and x.shape[-1] != 224:
            x = nn.functional.interpolate(x, size=224, mode="bilinear", align_corners=False)
        return self.backbone(x)


# Metrics (publication-quality)

def expected_calibration_error(probs: np.ndarray, labels: np.ndarray,
                                n_bins: int = 15) -> float:
    """Expected Calibration Error (ECE) - measures probability calibration."""
    bins     = np.linspace(0, 1, n_bins + 1)
    ece      = 0.0
    n        = len(probs)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() == 0:
            continue
        acc  = labels[mask].mean()
        conf = probs[mask].mean()
        ece += mask.sum() / n * abs(acc - conf)
    return float(ece)


def compute_metrics(logits: np.ndarray, labels: np.ndarray,
                    threshold: float = 0.5) -> dict:
    probs = stable_sigmoid(logits)
    preds = (probs >= threshold).astype(int)

    cm  = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, int(labels.sum()))
    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    ppv         = tp / (tp + fp + 1e-8)
    npv         = tn / (tn + fn + 1e-8)

    # ROC curve for Sensitivity@Specificity targets
    fpr, tpr, _ = roc_curve(labels, probs)
    spec_arr    = 1 - fpr
    sens_at_95  = float(tpr[np.argmin(np.abs(spec_arr - 0.95))])
    sens_at_99  = float(tpr[np.argmin(np.abs(spec_arr - 0.99))])

    return {
        "auc"         : float(roc_auc_score(labels, probs)),
        "auc_pr"      : float(average_precision_score(labels, probs)),
        "f1"          : float(f1_score(labels, preds, zero_division=0)),
        "accuracy"    : float(accuracy_score(labels, preds)),
        "sensitivity" : float(sensitivity),
        "specificity" : float(specificity),
        "ppv"         : float(ppv),
        "npv"         : float(npv),
        "kappa"       : float(cohen_kappa_score(labels, preds)),
        "mcc"         : float(matthews_corrcoef(labels, preds)),
        "ece"         : float(expected_calibration_error(probs, labels)),
        "sens_at_95spec": sens_at_95,
        "sens_at_99spec": sens_at_99,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def bootstrap_ci(logits: np.ndarray, labels: np.ndarray,
                 n_iter: int = 1000, ci: float = 0.95,
                 threshold: float = 0.5, seed: int = 42) -> dict:
    """Bootstrap confidence intervals for all metrics."""
    rng     = np.random.default_rng(seed)
    n       = len(labels)
    records = []
    for _ in range(n_iter):
        idx = rng.integers(0, n, size=n)
        try:
            m = compute_metrics(logits[idx], labels[idx], threshold)
            records.append(m)
        except Exception:
            continue

    alpha = 1 - ci
    ci_dict = {}
    scalar_keys = [k for k in records[0] if k != "confusion_matrix"]
    for key in scalar_keys:
        vals = sorted(r[key] for r in records)
        lo   = vals[int(alpha / 2 * len(vals))]
        hi   = vals[int((1 - alpha / 2) * len(vals))]
        ci_dict[key] = {"mean": float(np.mean(vals)), "lo": lo, "hi": hi, "ci": ci}
    return ci_dict


# Training loop

def train_one_epoch(model, loader, optimizer, criterion, scaler,
                    grad_clip_norm: float = 1.0,
                    use_amp: bool = False) -> dict:
    model.train()
    total_loss = 0.0
    all_logits, all_labels = [], []
    grad_norms = []

    for images, labels in loader:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)

        optimizer.zero_grad()
        with autocast(device_type='cuda', enabled=use_amp):
            logits = model(images)
            loss   = criterion(logits, labels)

        if not torch.isfinite(loss):
            print("  [WARN] Non-finite training loss detected; skipping batch.", flush=True)
            optimizer.zero_grad(set_to_none=True)
            continue

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        if not torch.isfinite(grad_norm):
            print("  [WARN] Non-finite gradient norm; skipping optimizer step.", flush=True)
            optimizer.zero_grad(set_to_none=True)
            grad_norms.append(float("nan"))
            scaler.update()
            continue
        grad_norms.append(float(grad_norm))
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item() * len(labels)
        all_logits.append(logits.detach().cpu().float().numpy())
        all_labels.append(labels.cpu().numpy())

    if not all_logits:
        raise RuntimeError("All training batches were skipped due to non-finite loss.")

    all_logits = np.concatenate(all_logits)
    all_labels = np.concatenate(all_labels)
    m          = compute_metrics(all_logits, all_labels)
    m["loss"]       = total_loss / len(all_labels)
    m["grad_norm"]  = float(np.nanmean(grad_norms)) if grad_norms else float("nan")
    return m


@torch.no_grad()
def evaluate(model, loader, criterion, threshold: float = 0.5,
             use_amp: bool = False) -> dict:
    model.eval()
    total_loss = 0.0
    all_logits, all_labels = [], []

    for images, labels in loader:
        images = images.to(DEVICE, non_blocking=True)
        labels = labels.to(DEVICE, non_blocking=True)
        with autocast(device_type='cuda', enabled=use_amp):
            logits = model(images)
            loss   = criterion(logits, labels)
        if not torch.isfinite(loss):
            print("  [WARN] Non-finite validation/test loss detected; skipping batch.", flush=True)
            continue
        total_loss += loss.item() * len(labels)
        all_logits.append(logits.cpu().float().numpy())
        all_labels.append(labels.cpu().numpy())

    if not all_logits:
        raise RuntimeError("All evaluation batches were skipped due to non-finite loss.")

    all_logits = np.concatenate(all_logits)
    all_labels = np.concatenate(all_labels)
    m          = compute_metrics(all_logits, all_labels, threshold)
    m["loss"]  = total_loss / len(all_labels)
    return m, all_logits, all_labels


# Scheduler factory

def build_scheduler(optimizer, t_cfg: dict):
    sched_name    = t_cfg.get("scheduler", "cosine")
    epochs        = t_cfg["epochs"]
    warmup_epochs = t_cfg.get("warmup_epochs", 5)

    if sched_name == "cosine":
        def lr_lambda(epoch):
            if epoch < warmup_epochs:
                return (epoch + 1) / max(1, warmup_epochs)
            progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
            return 0.5 * (1.0 + np.cos(np.pi * progress))
        return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    elif sched_name == "step":
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size = t_cfg.get("step_size", 20),
            gamma     = t_cfg.get("step_gamma", 0.5),
        )
    elif sched_name == "plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max",
            patience = t_cfg.get("plateau_patience", 8),
            factor   = t_cfg.get("plateau_factor", 0.5),
        )
    else:
        raise ValueError(f"Unknown scheduler: {sched_name}")


# Main training orchestrator

def train(cfg: dict, backbone: str = None):
    t_cfg     = cfg["training"]
    threshold = t_cfg.get("threshold", 0.5)
    set_seed(t_cfg["seed"])

    backbone = backbone or t_cfg["backbone"]

    print(f"\n{'='*60}")
    print(f"  Phase 3 - Patch Classifier Training")
    print(f"  Backbone  : {backbone}")
    print(f"  Device    : {DEVICE}")
    print(f"  Epochs    : {t_cfg['epochs']}  (early_stop patience={t_cfg['early_stopping_patience']})")
    print(f"  Scheduler : {t_cfg.get('scheduler','cosine')}  warmup={t_cfg.get('warmup_epochs',5)}")
    print(f"{'='*60}")

    # Save environment metadata
    env_meta = get_env_metadata()
    env_meta.update({"backbone": backbone, "config": t_cfg, "seed": t_cfg["seed"]})
    reports_dir = ROOT / cfg["paths"]["reports_dir"]
    reports_dir.mkdir(parents=True, exist_ok=True)
    model_dir   = ROOT / cfg["paths"]["models_dir"]
    model_dir.mkdir(parents=True, exist_ok=True)
    env_path    = reports_dir / f"{backbone.replace('/','_')}_environment.json"
    with open(env_path, "w") as f:
        json.dump(env_meta, f, indent=2)
    print(f"  Environment metadata -> {env_path}")

    # Datasets
    train_ds = PCamDataset("train", transform=get_train_transforms(), cfg=cfg)
    valid_ds = PCamDataset("valid", transform=get_val_transforms(),   cfg=cfg)
    test_ds  = PCamDataset("test",  transform=get_val_transforms(),   cfg=cfg)

    train_loader = DataLoader(
        train_ds, batch_size=t_cfg["batch_size"], shuffle=True,
        num_workers=t_cfg["num_workers"], pin_memory=True,
        persistent_workers=t_cfg["num_workers"] > 0,
    )
    val_loader = DataLoader(
        valid_ds, batch_size=t_cfg["batch_size"] * 2, shuffle=False,
        num_workers=t_cfg["num_workers"], pin_memory=True,
        persistent_workers=t_cfg["num_workers"] > 0,
    )
    test_loader = DataLoader(
        test_ds, batch_size=t_cfg["batch_size"] * 2, shuffle=False,
        num_workers=t_cfg["num_workers"],
    )

    print(f"\n  Train: {len(train_ds):,}  |  Val: {len(valid_ds):,}  |  Test: {len(test_ds):,}")
    print(f"  Batch: {t_cfg['batch_size']}  |  LR: {t_cfg['lr']}  |  WD: {t_cfg['weight_decay']}\n")

    # Model + optimizer
    model     = PatchClassifier(backbone, pretrained=t_cfg["pretrained"]).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(),
                            lr=t_cfg["lr"], weight_decay=t_cfg["weight_decay"])
    scheduler   = build_scheduler(optimizer, t_cfg)
    use_amp     = bool(t_cfg.get("use_amp", False)) and DEVICE.type == "cuda"
    scaler      = GradScaler('cuda', enabled=use_amp)
    grad_clip   = t_cfg.get("grad_clip_norm", 1.0)
    is_plateau  = t_cfg.get("scheduler", "cosine") == "plateau"

    # Training loop
    best_auc       = 0.0
    patience_count = 0
    history        = []
    ckpt_path      = model_dir / f"{backbone.replace('/','_')}_best.pth"

    for epoch in range(1, t_cfg["epochs"] + 1):
        t0      = time.time()
        train_m = train_one_epoch(model, train_loader, optimizer, criterion,
                                  scaler, grad_clip, use_amp=use_amp)
        val_m, _, _ = evaluate(model, val_loader, criterion, threshold, use_amp=use_amp)

        if is_plateau:
            scheduler.step(val_m["auc"])
        else:
            scheduler.step()

        lr_now  = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - t0

        print(
            f"  Ep {epoch:03d}/{t_cfg['epochs']}  "
            f"| Tr  loss={train_m['loss']:.4f} auc={train_m['auc']:.4f} "
            f"f1={train_m['f1']:.4f} sens={train_m['sensitivity']:.4f} "
            f"gn={train_m['grad_norm']:.3f}"
            f"| Val loss={val_m['loss']:.4f} auc={val_m['auc']:.4f} "
            f"f1={val_m['f1']:.4f} sens={val_m['sensitivity']:.4f} "
            f"| LR={lr_now:.2e} t={elapsed:.0f}s",
            flush=True,
        )

        row = {"epoch": epoch, "lr": lr_now,
               **{f"train_{k}": v for k, v in train_m.items()
                  if k != "confusion_matrix"},
               **{f"val_{k}": v for k, v in val_m.items()
                  if k != "confusion_matrix"}}
        history.append(row)

        if val_m["auc"] > best_auc:
            best_auc       = val_m["auc"]
            patience_count = 0
            torch.save({
                "epoch"          : epoch,
                "backbone"       : backbone,
                "state_dict"     : model.state_dict(),
                "optimizer"      : optimizer.state_dict(),
                "best_val_auc"   : best_auc,
                "val_metrics"    : {k: v for k, v in val_m.items()
                                    if k != "confusion_matrix"},
                "env"            : env_meta,
                "cfg"            : t_cfg,
            }, ckpt_path)
            print(f"    [BEST] val_auc={best_auc:.4f}  val_auc_pr={val_m['auc_pr']:.4f}  "
                  f"sens@95={val_m['sens_at_95spec']:.4f}", flush=True)
        else:
            patience_count += 1
            if patience_count >= t_cfg["early_stopping_patience"]:
                print(f"\n  Early stopping at epoch {epoch} "
                      f"(no AUC gain for {patience_count} epochs)", flush=True)
                break

    # Save training history as CSV
    import csv
    csv_path = reports_dir / f"{backbone.replace('/','_')}_training_history.csv"
    if history:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=history[0].keys())
            writer.writeheader()
            writer.writerows(history)
        print(f"  Training history CSV -> {csv_path}")

    # Final test evaluation
    print(f"\n  Loading best checkpoint from {ckpt_path} ...")
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])

    test_m, test_logits, test_labels = evaluate(
        model, test_loader, criterion, threshold, use_amp=use_amp
    )

    print(f"\n{'='*60}")
    print(f"  FINAL TEST RESULTS - {backbone}")
    print(f"  AUC-ROC   : {test_m['auc']:.4f}")
    print(f"  AUC-PR    : {test_m['auc_pr']:.4f}")
    print(f"  F1        : {test_m['f1']:.4f}")
    print(f"  Accuracy  : {test_m['accuracy']:.4f}")
    print(f"  Sensitivity : {test_m['sensitivity']:.4f}")
    print(f"  Specificity : {test_m['specificity']:.4f}")
    print(f"  Sens@95Spec : {test_m['sens_at_95spec']:.4f}")
    print(f"  Sens@99Spec : {test_m['sens_at_99spec']:.4f}")
    print(f"  Kappa     : {test_m['kappa']:.4f}")
    print(f"  MCC       : {test_m['mcc']:.4f}")
    print(f"  ECE       : {test_m['ece']:.4f}")
    cm = test_m["confusion_matrix"]
    print(f"  Confusion Matrix:")
    print(f"    TN={cm['tn']}  FP={cm['fp']}")
    print(f"    FN={cm['fn']}  TP={cm['tp']}")
    print(f"{'='*60}")

    # Bootstrap CI
    print(f"\n  Computing bootstrap 95% CI ({t_cfg.get('bootstrap_n_iter',1000)} iterations) ...",
          flush=True)
    ci = bootstrap_ci(
        test_logits, test_labels,
        n_iter    = t_cfg.get("bootstrap_n_iter", 1000),
        ci        = t_cfg.get("bootstrap_ci", 0.95),
        threshold = threshold,
        seed      = t_cfg["seed"],
    )
    print(f"  AUC-ROC  : {ci['auc']['mean']:.4f} "
          f"[{ci['auc']['lo']:.4f}, {ci['auc']['hi']:.4f}] 95% CI")
    print(f"  AUC-PR   : {ci['auc_pr']['mean']:.4f} "
          f"[{ci['auc_pr']['lo']:.4f}, {ci['auc_pr']['hi']:.4f}] 95% CI")
    print(f"  F1       : {ci['f1']['mean']:.4f} "
          f"[{ci['f1']['lo']:.4f}, {ci['f1']['hi']:.4f}] 95% CI")
    print(f"  Sens@95  : {ci['sens_at_95spec']['mean']:.4f} "
          f"[{ci['sens_at_95spec']['lo']:.4f}, {ci['sens_at_95spec']['hi']:.4f}] 95% CI")

    # Save test predictions for downstream analysis
    preds_path = reports_dir / f"{backbone.replace('/','_')}_test_predictions.npz"
    np.savez(str(preds_path),
             logits=test_logits, probs=stable_sigmoid(test_logits), labels=test_labels)
    print(f"  Test predictions -> {preds_path}")

    # Save full results JSON
    results = {
        "backbone"      : backbone,
        "best_val_auc"  : best_auc,
        "test_metrics"  : {k: v for k, v in test_m.items() if k != "confusion_matrix"},
        "confusion_matrix": test_m["confusion_matrix"],
        "bootstrap_ci"  : ci,
        "history"       : history,
        "environment"   : env_meta,
        "threshold"     : threshold,
    }
    results_path = reports_dir / f"{backbone.replace('/','_')}_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    # Also save to canonical patch_results.json for downstream phases
    patch_results = {
        "backbone" : backbone,
        "test_auc" : test_m["auc"],
        "test_auc_pr": test_m["auc_pr"],
        "test_f1"  : test_m["f1"],
        "test_accuracy": test_m["accuracy"],
        "test_sensitivity": test_m["sensitivity"],
        "test_specificity": test_m["specificity"],
        "test_sens_at_95spec": test_m["sens_at_95spec"],
        "test_kappa": test_m["kappa"],
        "test_mcc"  : test_m["mcc"],
        "test_ece"  : test_m["ece"],
        "confusion_matrix": test_m["confusion_matrix"],
        "bootstrap_ci": ci,
    }
    with open(reports_dir / "patch_results.json", "w") as f:
        json.dump(patch_results, f, indent=2)

    print(f"  Results -> {results_path}")
    return model, results


# Feature extraction (used by Phase 4 MIL)

from typing import Tuple

@torch.no_grad()
def extract_features(
    model : nn.Module,
    loader: DataLoader,
    device: torch.device = DEVICE,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract CNN/ViT features for all patches. Returns (N, feat_dim), (N,)."""
    model.eval()
    all_feats, all_labels = [], []
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        with autocast(device_type='cuda', enabled=False):
            feats = model.get_features(images)
        all_feats.append(feats.cpu().float().numpy())
        all_labels.append(labels.numpy())
    return np.concatenate(all_feats), np.concatenate(all_labels)


# Entry point

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 3 - Patch Classifier Training")
    parser.add_argument("--backbone", type=str, default=None,
                        help="timm model name (overrides config.yaml)")
    args = parser.parse_args()
    cfg = load_config()
    train(cfg, backbone=args.backbone)

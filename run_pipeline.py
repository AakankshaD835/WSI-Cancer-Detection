"""
Master pipeline runner — executes all phases in order.

Usage:
    python run_pipeline.py                 # full pipeline
    python run_pipeline.py --phase 3       # single phase
    python run_pipeline.py --phase 3 4 5   # multiple phases
    python run_pipeline.py --demo          # demo mode (no training)
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from phase1_setup       import load_config, verify_files, print_dataset_summary
from phase2_preprocessing import (MacenkoNormalizer, get_train_transforms,
                                   get_val_transforms, visualize_normalization)
from phase3_train       import train as train_patch_classifier
from phase4_mil         import train_mil
from phase5_explainability import batch_cam_evaluation
from phase6_virtual_staining import demo_virtual_staining
from phase7_llm_report  import generate_clinical_report
import visualize_all


def run_phase1(cfg):
    print("\n" + "="*60)
    print("  PHASE 1 — Dataset Verification & Setup")
    print("="*60)
    ok = verify_files(cfg)
    if not ok:
        raise SystemExit("Missing dataset files. Check config.yaml paths.")
    print_dataset_summary(cfg)


def run_phase2(cfg):
    print("\n" + "="*60)
    print("  PHASE 2 — Preprocessing Preview")
    print("="*60)
    import h5py, numpy as np
    with h5py.File(ROOT / cfg["paths"]["pcam_train_x"], "r") as f:
        samples = f["x"][:8]
    norm = MacenkoNormalizer()
    norm.fit(samples[0])
    visualize_normalization(samples, norm, n=4,
                             save_path=str(ROOT / "reports" / "macenko_preview.png"))
    print("  Transforms defined:")
    print("    Train:", get_train_transforms())
    print("    Val  :", get_val_transforms())


def run_phase3(cfg):
    print("\n" + "="*60)
    print("  PHASE 3 — Patch Classifier Training")
    print("="*60)
    train_patch_classifier(cfg)


def run_phase4(cfg):
    print("\n" + "="*60)
    print("  PHASE 4 — MIL Training")
    print("="*60)
    train_mil(cfg)


def run_phase5(cfg):
    print("\n" + "="*60)
    print("  PHASE 5 — Explainability")
    print("="*60)
    from phase3_train import PatchClassifier
    import torch
    backbone  = cfg["training"]["backbone"]
    ckpt_path = ROOT / cfg["paths"]["models_dir"] / f"{backbone.replace('/','_')}_best.pth"
    if not ckpt_path.exists():
        print("  Skipping — no trained model found. Run Phase 3 first.")
        return
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model = PatchClassifier(backbone, pretrained=False)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    metrics = batch_cam_evaluation(model, cfg, n_samples=100)
    print(f"  Grad-CAM IoU={metrics['mean_iou']:.4f}  Dice={metrics['mean_dice']:.4f}")


def run_phase6(cfg):
    print("\n" + "="*60)
    print("  PHASE 6 — Virtual Staining Demo")
    print("="*60)
    demo_virtual_staining(cfg)


def run_phase7(cfg):
    print("\n" + "="*60)
    print("  PHASE 7 — LLM Clinical Report (Offline Demo)")
    print("="*60)
    import numpy as np
    np.random.seed(0)
    n = 30
    att = np.random.dirichlet(np.ones(n) * 0.5).tolist()
    top = sorted(range(n), key=lambda i: att[i], reverse=True)[:5]
    generate_clinical_report(
        tumor_probability = 0.89,
        mil_attention     = att,
        top_patch_indices = top,
        n_patches         = n,
        cfg               = cfg,
        wsi_id            = "demo_case_001",
        use_llm           = False,
    )


def run_phase8():
    print("\n" + "="*60)
    print("  PHASE 8 — Launch Streamlit UI")
    print("="*60)
    import subprocess
    subprocess.run(["streamlit", "run", str(ROOT / "ui_app" / "app.py")])


def run_visualize(cfg, phases=None):
    """Generate all visualizations (or specific phases)."""
    print("\n" + "="*60)
    print("  VISUALIZE — Generating all project visuals")
    print("="*60)
    import sys
    orig_argv = sys.argv
    if phases:
        sys.argv = ["visualize_all.py", "--phase"] + [str(p) for p in phases]
    else:
        sys.argv = ["visualize_all.py"]
    visualize_all.main()
    sys.argv = orig_argv


PHASES = {
    1: run_phase1,
    2: run_phase2,
    3: run_phase3,
    4: run_phase4,
    5: run_phase5,
    6: run_phase6,
    7: run_phase7,
    8: run_phase8,
    9: lambda cfg: run_visualize(cfg),    # full visualization pass
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", nargs="*", type=int,
                        help="Phase numbers to run (default: all)")
    parser.add_argument("--demo", action="store_true",
                        help="Demo mode: phases 1,2,6,7 only (no training)")
    args = parser.parse_args()

    cfg = load_config()

    if args.demo:
        phases_to_run = [1, 2, 6, 7]
    elif args.phase:
        phases_to_run = sorted(args.phase)
    else:
        phases_to_run = [1, 2, 3, 4, 5, 6, 7]

    print(f"\nRunning phases: {phases_to_run}")
    for p in phases_to_run:
        if p in PHASES:
            PHASES[p](cfg) if p != 8 else PHASES[p]()
        else:
            print(f"  Unknown phase {p} — skipping")

    print("\n✓ Pipeline complete.")

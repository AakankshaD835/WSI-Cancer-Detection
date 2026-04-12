# -*- coding: utf-8 -*-
# end-to-end pipeline runner: phases 1-7 + all visualizations

import sys, os, argparse, time, json, traceback
from pathlib import Path

# Load .env file if present (API keys etc.)
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent / ".env")
except ImportError:
    pass

# Force UTF-8 output so em-dashes and other non-ASCII chars don't crash on
# Windows consoles that default to cp949 / cp1252.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

os.environ.setdefault("PYTHONUTF8", "1")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))


def banner(title):
    bar = "=" * 65
    print(bar, flush=True)
    print(f"  {title}", flush=True)
    print(bar, flush=True)


def divider(title):
    print(f"\n--- {title} ---", flush=True)


def log_run_metadata(cfg):
    """Log hardware, software, config snapshot at pipeline start."""
    import platform, datetime
    import torch
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only"
    try:
        import subprocess
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(ROOT), stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        git_hash = "N/A"

    meta = {
        "run_timestamp" : datetime.datetime.now().isoformat(),
        "python_version": platform.python_version(),
        "torch_version" : torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version"  : torch.version.cuda or "N/A",
        "gpu"           : gpu,
        "platform"      : platform.platform(),
        "git_hash"      : git_hash,
        "config"        : cfg,
    }
    out = ROOT / cfg["paths"]["reports_dir"] / "run_metadata.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"  Run metadata -> {out}")
    print(f"  Python  : {meta['python_version']}")
    print(f"  PyTorch : {meta['torch_version']}")
    print(f"  GPU     : {gpu}")
    print(f"  Git     : {git_hash}")
    print(flush=True)


# ============================================================
# PHASE 1
# ============================================================
def run_phase1(cfg):
    banner("PHASE 1 - Dataset Verification & Summary")
    import h5py, numpy as np, pandas as pd
    from phase1_setup import verify_files

    ok = verify_files(cfg)
    if not ok:
        raise SystemExit("ERROR: Missing dataset files. Check config.yaml.")

    splits = {
        "TRAIN": ("pcam_train_x", "train_y", "train_meta"),
        "VALID": ("pcam_valid_x", "valid_y", "valid_meta"),
        "TEST":  ("pcam_test_x",  "test_y",  "test_meta"),
    }
    print("", flush=True)
    for split, (xk, yk, mk) in splits.items():
        with h5py.File(ROOT / cfg["paths"][xk], "r") as f:
            shape = f["x"].shape
        with h5py.File(ROOT / cfg["paths"][yk], "r") as f:
            y = f["y"][:].squeeze()
        meta = pd.read_csv(ROOT / cfg["paths"][mk], index_col=0)
        n_tumor  = int(y.sum())
        n_normal = int((y == 0).sum())
        print(f"  [{split}]  patches={shape[0]:,}  img={shape[1]}x{shape[2]}x{shape[3]}"
              f"  tumor={n_tumor:,}  normal={n_normal:,}  WSIs={meta['wsi'].nunique()}",
              flush=True)

    with h5py.File(ROOT / cfg["paths"]["train_mask"], "r") as f:
        mshape = f["mask"].shape
    print(f"  [MASK]  shape={mshape}  -- pixel-level tumor annotations", flush=True)

    try:
        import visualize_all
        visualize_all.phase1_visuals(cfg, ROOT / cfg["paths"]["reports_dir"])
    except Exception as e:
        print(f"  [!] Phase 1 visualizations failed: {e}", flush=True)

    print("\nPhase 1 DONE.\n", flush=True)


# ============================================================
# PHASE 2
# ============================================================
def run_phase2(cfg):
    banner("PHASE 2 - Preprocessing Pipeline")
    import h5py
    from phase2_preprocessing import MacenkoNormalizer, get_train_transforms, get_val_transforms

    with h5py.File(ROOT / cfg["paths"]["pcam_train_x"], "r") as f:
        sample = f["x"][0].copy()

    norm = MacenkoNormalizer()
    norm.fit(sample)
    print("  Macenko normalizer: fitted on reference patch.", flush=True)
    print(f"  Train augmentation pipeline: {len(get_train_transforms())} transforms", flush=True)
    print(f"  Val/Test pipeline: normalize + to tensor only", flush=True)

    try:
        import visualize_all
        visualize_all.phase2_visuals(cfg, ROOT / cfg["paths"]["reports_dir"])
    except Exception as e:
        print(f"  [!] Phase 2 visualizations failed: {e}", flush=True)

    print("\nPhase 2 DONE.\n", flush=True)


# ============================================================
# PHASE 3 - FULL TRAINING
# ============================================================
def run_phase3(cfg):
    banner("PHASE 3 - Patch-Level Training  (FULL DATA, NO SHORTCUTS)")
    print(f"  Backbone  : {cfg['training']['backbone']}", flush=True)
    print(f"  Epochs    : {cfg['training']['epochs']}  (early stop patience={cfg['training']['early_stopping_patience']})", flush=True)
    print(f"  Batch     : {cfg['training']['batch_size']}", flush=True)
    print(f"  LR        : {cfg['training']['lr']}", flush=True)
    print(f"  Train     : {cfg['data']['max_train_samples']:,} patches", flush=True)
    print(f"  Valid     : {cfg['data']['max_valid_samples']:,} patches", flush=True)
    print(f"  Test      : {cfg['data']['max_test_samples']:,} patches", flush=True)
    print("", flush=True)

    from phase3_train import train
    model, results = train(cfg)

    print(f"\n  === PHASE 3 RESULTS ===", flush=True)
    print(f"  Best Val AUC  : {results['best_val_auc']:.4f}", flush=True)
    print(f"  Test AUC      : {results['test_metrics']['auc']:.4f}", flush=True)
    print(f"  Test F1       : {results['test_metrics']['f1']:.4f}", flush=True)
    print(f"  Test Accuracy : {results['test_metrics']['accuracy']:.4f}", flush=True)

    try:
        import visualize_all
        visualize_all.phase3_visuals(cfg, ROOT / cfg["paths"]["reports_dir"])
    except Exception as e:
        print(f"  [!] Phase 3 visualizations failed: {e}", flush=True)

    print("\nPhase 3 DONE.\n", flush=True)
    return model


# ============================================================
# PHASE 4 - MIL TRAINING
# ============================================================
def run_phase4(cfg):
    banner("PHASE 4 - MIL Slide-Level Training  (FULL DATA, NO SHORTCUTS)")
    print(f"  Train WSIs : 216  |  Valid WSIs : 54  |  Test WSIs : 129", flush=True)
    print(f"  MIL epochs : {cfg['mil']['epochs']}  (early stop patience={cfg['mil'].get('early_stopping_patience', 15)})", flush=True)
    print(f"  LR         : {cfg['mil']['lr']}", flush=True)
    print("", flush=True)

    from phase4_mil import train_mil
    mil_model, results = train_mil(cfg)
    test_m = results["test_metrics"]

    print(f"\n  === PHASE 4 RESULTS ===", flush=True)
    print(f"  Slide AUC         : {test_m['auc']:.4f}", flush=True)
    print(f"  Slide F1          : {test_m['f1']:.4f}", flush=True)
    print(f"  Sens @ 95% Spec   : {test_m['sens@95']:.4f}", flush=True)

    try:
        import visualize_all
        visualize_all.phase4_visuals(cfg, ROOT / cfg["paths"]["reports_dir"])
    except Exception as e:
        print(f"  [!] Phase 4 visualizations failed: {e}", flush=True)

    print("\nPhase 4 DONE.\n", flush=True)
    return mil_model


# ============================================================
# PHASE 5 - EXPLAINABILITY
# ============================================================
def run_phase5(cfg):
    banner("PHASE 5 - Explainability: Grad-CAM + GT Mask Evaluation")
    import torch, h5py, numpy as np
    from phase3_train import PatchClassifier, DEVICE
    from phase5_explainability import (batch_cam_evaluation,
                                        generate_explainability_grid)

    backbone  = cfg["training"]["backbone"]
    ckpt_path = ROOT / cfg["paths"]["models_dir"] / f"{backbone.replace('/','_')}_best.pth"
    if not ckpt_path.exists():
        print("  Skipping -- no trained model found (Phase 3 must complete first).", flush=True)
        return

    ckpt  = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    model = PatchClassifier(backbone, pretrained=False).to(DEVICE)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # Quantitative Grad-CAM vs pixel mask (all available tumor patches)
    n_cam = cfg.get("explainability", {}).get("gradcam_n_samples", None)
    label_str = f"{n_cam}" if n_cam else "ALL"
    print(f"  Grad-CAM vs GT mask evaluation on {label_str} tumor patches ...", flush=True)
    metrics = batch_cam_evaluation(model, cfg, n_samples=n_cam)

    out_path = ROOT / cfg["paths"]["reports_dir"] / "gradcam_metrics.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"  Mean IoU  : {metrics['mean_iou']:.4f} +/- {metrics['std_iou']:.4f}", flush=True)
    print(f"  Mean Dice : {metrics['mean_dice']:.4f} +/- {metrics['std_dice']:.4f}", flush=True)

    # Visual explainability grid
    heatmap_dir = ROOT / cfg["paths"]["reports_dir"] / "heatmaps"
    heatmap_dir.mkdir(exist_ok=True)
    print("  Generating explainability grid (4 patches) ...", flush=True)
    with h5py.File(ROOT / cfg["paths"]["pcam_train_x"], "r") as fx:
        images = fx["x"][:500]
    with h5py.File(ROOT / cfg["paths"]["train_y"], "r") as fy:
        labels = fy["y"][:500].squeeze().astype("uint8")
    generate_explainability_grid(model, images, labels, n=4,
                                  save_dir=str(heatmap_dir))

    try:
        import visualize_all
        visualize_all.phase5_visuals(cfg, ROOT / cfg["paths"]["reports_dir"])
    except Exception as e:
        print(f"  [!] Phase 5 visualizations failed: {e}", flush=True)

    print("\nPhase 5 DONE.\n", flush=True)


# ============================================================
# PHASE 6 - VIRTUAL STAINING
# ============================================================
def run_phase6(cfg):
    banner("PHASE 6 - Virtual Staining (CycleGAN Full Training + Inference)")
    from phase6_virtual_staining import train_and_infer
    train_and_infer(cfg)

    try:
        import visualize_all
        visualize_all.phase6_visuals(cfg, ROOT / cfg["paths"]["reports_dir"])
    except Exception as e:
        print(f"  [!] Phase 6 visualizations failed: {e}", flush=True)

    print("\nPhase 6 DONE.\n", flush=True)


# ============================================================
# PHASE 7 - LLM CLINICAL REPORT
# ============================================================
def run_phase7(cfg):
    banner("PHASE 7 - LLM Clinical Report Generation (All Test WSIs)")
    from phase4_mil import predict_all_test_wsis
    from phase7_llm_report import generate_clinical_report

    # Determine LLM availability
    has_anthropic   = bool(os.environ.get("ANTHROPIC_API_KEY"))
    has_openai      = bool(os.environ.get("OPENAI_API_KEY"))
    has_groq        = bool(os.environ.get("GROQ_API_KEY"))
    has_gemini      = bool(os.environ.get("GEMINI_API_KEY"))
    has_huggingface = bool(os.environ.get("HF_API_KEY"))
    use_llm         = any([has_anthropic, has_openai, has_groq, has_gemini, has_huggingface])

    if use_llm:
        # Auto-select provider based on available keys (free providers preferred)
        provider = cfg["llm"]["provider"]
        if provider == "anthropic" and not has_anthropic:
            if has_groq:        provider = "groq"
            elif has_gemini:    provider = "gemini"
            elif has_huggingface: provider = "huggingface"
            elif has_openai:    provider = "openai"
            cfg["llm"]["provider"] = provider
        print(f"  LLM API key found ({provider}) -- real LLM will be called per WSI.",
              flush=True)
    else:
        print("  No LLM API key found.", flush=True)
        print("  FREE options: Groq (GROQ_API_KEY) | Gemini (GEMINI_API_KEY) | HuggingFace (HF_API_KEY)", flush=True)
        print("  Generating structured template reports from real model outputs.", flush=True)

    # --- Real Grad-CAM IoU from Phase 5 --------------------------------
    cam_json    = ROOT / cfg["paths"]["reports_dir"] / "gradcam_metrics.json"
    gradcam_iou = None
    if cam_json.exists():
        with open(cam_json) as f:
            cam_data = json.load(f)
        gradcam_iou = cam_data.get("mean_iou")
        print(f"  Real Grad-CAM mean IoU: {gradcam_iou:.4f}", flush=True)
    else:
        print("  gradcam_metrics.json not found -- Phase 5 may not have run.", flush=True)

    # --- Real per-WSI predictions from trained Phase 3 + Phase 4 models
    # (will fail loudly if models are missing -- no fallback to fake values)
    wsi_predictions = predict_all_test_wsis(cfg)

    print(f"\n  Generating clinical reports for {len(wsi_predictions)} test WSIs ...",
          flush=True)

    heatmap_dir = ROOT / cfg["paths"]["reports_dir"] / "heatmaps"
    n_success   = 0
    n_tumor     = sum(1 for w in wsi_predictions if w["tumor_probability"] > 0.5)

    print(f"  Model summary: {n_tumor} TUMOR / {len(wsi_predictions)-n_tumor} NORMAL",
          flush=True)

    for i, wsi in enumerate(wsi_predictions):
        wsi_id  = wsi["wsi_id"]
        hm_path = str(heatmap_dir / f"{wsi_id}_heatmap.png")
        hm_path = hm_path if Path(hm_path).exists() else None

        try:
            generate_clinical_report(
                tumor_probability = wsi["tumor_probability"],
                mil_attention     = wsi["attention_scores"],
                top_patch_indices = wsi["top_patch_indices"],
                n_patches         = wsi["n_patches"],
                cfg               = cfg,
                wsi_id            = wsi_id,
                gradcam_iou       = gradcam_iou,
                heatmap_path      = hm_path,
                use_llm           = use_llm,
            )
            n_success += 1
            if (i + 1) % 10 == 0 or i == 0:
                print(f"  [{i+1:3d}/{len(wsi_predictions)}] {wsi_id}  "
                      f"prob={wsi['tumor_probability']:.4f}", flush=True)
        except Exception as e:
            print(f"  [!] Report failed for {wsi_id}: {e}", flush=True)

    print(f"\n  Reports generated: {n_success}/{len(wsi_predictions)}", flush=True)
    print(f"  Saved to: {ROOT / cfg['paths']['reports_dir'] / 'clinical'}", flush=True)

    try:
        import visualize_all
        visualize_all.phase7_visuals(cfg, ROOT / cfg["paths"]["reports_dir"])
    except Exception as e:
        print(f"  [!] Phase 7 visualizations failed: {e}", flush=True)

    print("\nPhase 7 DONE.\n", flush=True)


# ============================================================
# VISUALIZATIONS
# ============================================================
def run_visualize(cfg):
    banner("VISUALIZATIONS - Bonus / UI Plots (phases 1-7 already saved inline)")
    import visualize_all
    out_dir = ROOT / cfg["paths"]["reports_dir"]
    # Phases 1-7 visuals are already saved at the end of each phase.
    # Only run the remaining bonus entries here.
    BONUS_PHASES = {k: v for k, v in visualize_all.PHASE_MAP.items() if k >= 8}
    for p, (name, fn) in BONUS_PHASES.items():
        print(f"\n  [VIZ {p}] {name} ...", flush=True)
        try:
            fn(cfg, out_dir)
        except Exception as e:
            print(f"  [!] Viz phase {p} failed: {e}", flush=True)
            traceback.print_exc()
    print("\nBonus visualizations DONE.\n", flush=True)


# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--from", dest="from_phase", type=int, default=1,
                        help="Resume from phase number (default=1 = full run)")
    args = parser.parse_args()

    from phase1_setup import load_config
    cfg = load_config()

    t_start = time.time()

    print("", flush=True)
    print("=" * 65, flush=True)
    print("  WSI FULL PIPELINE - END TO END", flush=True)
    print(f"  Starting from Phase {args.from_phase}", flush=True)
    print(f"  Train: 50,000 | Valid: 8,000 | Test: 8,000 patches", flush=True)
    print(f"  Train WSIs: 216 | Valid: 54 | Test: 129", flush=True)
    print("=" * 65, flush=True)
    print("", flush=True)

    log_run_metadata(cfg)

    PHASES = [
        (1,  "Phase 1 - Verification",   run_phase1),
        (2,  "Phase 2 - Preprocessing",  run_phase2),
        (3,  "Phase 3 - CNN Training",   run_phase3),
        (4,  "Phase 4 - MIL Training",   run_phase4),
        (5,  "Phase 5 - Explainability", run_phase5),
        (6,  "Phase 6 - Virtual Stain",  run_phase6),
        (7,  "Phase 7 - LLM Report",     run_phase7),
        (99, "Visualizations",           run_visualize),
    ]

    for num, label, fn in PHASES:
        if num < args.from_phase and num != 99:
            print(f"  Skipping {label} (resuming from Phase {args.from_phase})", flush=True)
            continue
        t0 = time.time()
        try:
            fn(cfg)
            elapsed = (time.time() - t0) / 60
            print(f"  [{label}] finished in {elapsed:.1f} min", flush=True)
        except Exception as e:
            print(f"\n[ERROR] {label} failed: {e}", flush=True)
            traceback.print_exc()
            print("  Continuing to next phase ...\n", flush=True)

    total = (time.time() - t_start) / 3600
    print("", flush=True)
    print("=" * 65, flush=True)
    print(f"  PIPELINE COMPLETE  ({total:.2f} hours total)", flush=True)
    print(f"  Models  -> {ROOT / 'models'}", flush=True)
    print(f"  Reports -> {ROOT / 'reports'}", flush=True)
    print("=" * 65, flush=True)

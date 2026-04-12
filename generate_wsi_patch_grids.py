"""
Generate per-WSI patch grid images showing the actual top-attention patches
extracted from the real histopathology HDF5 file.

Saves: reports/clinical/wsi_patches/<wsi_id>_patches.png
       (used in PDF reports as per-sample medical images)
"""
import sys, json, h5py, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

CLINICAL_DIR = ROOT / "reports" / "clinical"
PATCH_DIR    = CLINICAL_DIR / "wsi_patches"
PATCH_DIR.mkdir(exist_ok=True)

H5_PATH   = ROOT / "WSI-Data" / "pcam" / "test_split.h5"
META_PATH = ROOT / "WSI-Data" / "Metadata" / "Metadata" / "test_metadata.csv"

print("Loading metadata and patch images ...")
meta = pd.read_csv(META_PATH)
meta_sub = meta.iloc[:8000].reset_index(drop=True)  # matches features_test.npy

# Load all 8000 patches at once (avoids repeated H5 seeks)
print(f"Loading 8000 patches from HDF5 ...")
with h5py.File(H5_PATH, "r") as f:
    key = list(f.keys())[0]
    images = f[key][:8000]          # (8000, 96, 96, 3) uint8
print(f"Loaded: {images.shape}  dtype={images.dtype}")

# Get per-WSI attention data from JSON files
analysis_files = sorted(CLINICAL_DIR.glob("*_analysis.json"))
print(f"Processing {len(analysis_files)} WSIs ...\n")

is_tumor_map = {}   # wsi_id -> bool
for jf in analysis_files:
    try:
        d = json.loads(jf.read_text(encoding="utf-8"))
        wsi_id = d["metadata"]["wsi_id"]
        is_tumor_map[wsi_id] = d["classification"]["prediction"] == "TUMOR"
    except Exception:
        pass

def make_patch_grid(wsi_id, patch_indices, images, top_regions, top_scores, is_tumor):
    """
    Create a grid figure showing actual tissue patches with attention overlay.
    patch_indices: list of indices into the 8000-image array (sorted by position).
    top_regions: list like ['patch_0030', 'patch_0001', ...] (local WSI indices).
    """
    accent  = "#C0392B" if is_tumor else "#27AE60"
    bg_col  = "#FFF5F5" if is_tumor else "#F5FFF8"
    label   = "TUMOR" if is_tumor else "NORMAL"

    # Parse top attention local indices
    top_local = []
    for r in top_regions:
        try:
            top_local.append(int(r.replace("patch_", "")))
        except Exception:
            pass

    # Map local WSI index → global H5 index
    n = len(patch_indices)
    # patch_0000 = patch_indices[0], patch_0001 = patch_indices[1], etc.
    def local_to_global(local_idx):
        if 0 <= local_idx < n:
            return patch_indices[local_idx]
        return None

    # Build display list: show top-5 attention patches + 3 context patches
    show_local = top_local[:5]
    # Add some non-top patches as context (first few in the bag)
    context = [i for i in range(min(n, 10)) if i not in show_local][:3]
    all_show = show_local + context

    n_show = len(all_show)
    if n_show == 0:
        # fallback: show first 8 patches
        all_show = list(range(min(8, n)))
        n_show   = len(all_show)

    ncols = min(n_show, 4)
    nrows = (n_show + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * 2.5, nrows * 2.8),
                             facecolor=bg_col)
    fig.patch.set_edgecolor(accent)
    fig.patch.set_linewidth(2)

    axes_flat = np.array(axes).flatten() if n_show > 1 else [axes]

    for ax_i, local_idx in enumerate(all_show):
        ax = axes_flat[ax_i]
        g  = local_to_global(local_idx)
        if g is not None and g < len(images):
            img = images[g]            # (96, 96, 3) uint8
            ax.imshow(img)
        else:
            ax.imshow(np.ones((96, 96, 3), dtype=np.uint8) * 220)

        # Determine if this is a top-attention patch
        is_top = local_idx in top_local
        rank   = top_local.index(local_idx) + 1 if is_top else None

        # Border color: red/green for top, gray for context
        border_col = accent if is_top else "#AAAAAA"
        border_w   = 4 if is_top else 1.5
        for spine in ax.spines.values():
            spine.set_edgecolor(border_col)
            spine.set_linewidth(border_w)

        # Attention score annotation
        if is_top and rank is not None and rank <= len(top_scores):
            score_txt = f"#{rank}  Attn: {top_scores[rank-1]:.4f}"
            ax.set_title(score_txt, fontsize=7.5, fontweight="bold",
                         color=accent, pad=3)
        else:
            ax.set_title(f"Context patch", fontsize=7, color="#888888", pad=3)

        ax.set_xticks([])
        ax.set_yticks([])
        ax.tick_params(left=False, bottom=False)

    # Hide unused axes
    for ax_i in range(n_show, len(axes_flat)):
        axes_flat[ax_i].set_visible(False)

    # Legend patches
    handles = [
        mpatches.Patch(color=accent, label=f"Top-attention (suspicious)" if is_tumor else "Top-attention"),
        mpatches.Patch(color="#AAAAAA", label="Context patches"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2,
               fontsize=8, framealpha=0.9, bbox_to_anchor=(0.5, -0.01))

    short_id = wsi_id.replace("camelyon16_test_", "Test Slide #")
    fig.suptitle(
        f"{short_id}  -  {label}  -  Top Attention Patches (Real H&E Tissue)",
        fontsize=9.5, fontweight="bold", color=accent, y=1.02
    )

    out_path = PATCH_DIR / f"{wsi_id}_patches.png"
    fig.savefig(str(out_path), dpi=130, bbox_inches="tight",
                facecolor=bg_col, edgecolor=accent)
    plt.close(fig)
    return str(out_path)


ok = 0
for jf in sorted(analysis_files):
    try:
        d = json.loads(jf.read_text(encoding="utf-8"))
        wsi_id     = d["metadata"]["wsi_id"]
        top_regions= d["attention_analysis"]["top_regions"]
        top_scores = d["attention_analysis"]["top_attention_scores"]
        is_tumor   = d["classification"]["prediction"] == "TUMOR"

        # Get sorted patch indices for this WSI
        wsi_meta = meta_sub[meta_sub["wsi"] == wsi_id]
        patch_indices = wsi_meta.index.tolist()   # position in the 8000-array

        if len(patch_indices) == 0:
            print(f"  SKIP {wsi_id} — no patches in subset")
            continue

        out = make_patch_grid(wsi_id, patch_indices, images,
                              top_regions, top_scores, is_tumor)
        ok += 1
        print(f"  [{ok:3d}/{len(analysis_files)}]  {wsi_id}  "
              f"({'TUMOR' if is_tumor else 'NORMAL'})  "
              f"{len(patch_indices)} patches  -> {Path(out).name}")
    except Exception as e:
        print(f"  ERROR {jf.stem}: {e}")

print(f"\nDone. {ok}/{len(analysis_files)} WSI patch grids saved to {PATCH_DIR}")

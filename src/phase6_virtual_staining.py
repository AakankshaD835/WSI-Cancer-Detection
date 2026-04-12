# Phase 6 - CycleGAN virtual staining (H&E domain transfer, LSGAN, cycle-consistency)

import argparse
import sys
import itertools
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import matplotlib
matplotlib.use("Agg")          # non-interactive backend -- safe for pipeline
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase1_setup import load_config

ROOT   = Path(__file__).resolve().parent.parent
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Building blocks

def conv_norm_relu(
    in_ch: int, out_ch: int,
    kernel: int = 3, stride: int = 1,
    padding: int = 1, norm: bool = True,
) -> nn.Sequential:
    layers = [nn.Conv2d(in_ch, out_ch, kernel, stride, padding, bias=not norm)]
    if norm:
        layers.append(nn.InstanceNorm2d(out_ch, affine=True))
    layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)


class ResidualBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(ch, ch, 3, padding=0, bias=False),
            nn.InstanceNorm2d(ch, affine=True),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(1),
            nn.Conv2d(ch, ch, 3, padding=0, bias=False),
            nn.InstanceNorm2d(ch, affine=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


# Generator (encoder -> residual bottleneck -> decoder)

class CycleGANGenerator(nn.Module):
    """CycleGAN Generator: 9-ResBlock version for 96x96 images."""

    def __init__(self, in_ch: int = 3, out_ch: int = 3,
                 base_ch: int = 64, n_resblocks: int = 9):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_ch, base_ch, 7, padding=0, bias=False),
            nn.InstanceNorm2d(base_ch, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch, base_ch * 2, 3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(base_ch * 2, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(base_ch * 2, base_ch * 4, 3, stride=2, padding=1, bias=False),
            nn.InstanceNorm2d(base_ch * 4, affine=True),
            nn.ReLU(inplace=True),
        )

        self.res_blocks = nn.Sequential(
            *[ResidualBlock(base_ch * 4) for _ in range(n_resblocks)]
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(base_ch * 4, base_ch * 2, 3, stride=2,
                               padding=1, output_padding=1, bias=False),
            nn.InstanceNorm2d(base_ch * 2, affine=True),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(base_ch * 2, base_ch, 3, stride=2,
                               padding=1, output_padding=1, bias=False),
            nn.InstanceNorm2d(base_ch, affine=True),
            nn.ReLU(inplace=True),
            nn.ReflectionPad2d(3),
            nn.Conv2d(base_ch, out_ch, 7, padding=0),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.encoder(x)
        x = self.res_blocks(x)
        return self.decoder(x)


# Discriminator (PatchGAN)

class PatchGANDiscriminator(nn.Module):
    """70x70 PatchGAN discriminator."""

    def __init__(self, in_ch: int = 3, base_ch: int = 64, n_layers: int = 3):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, base_ch, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        ch = base_ch
        for _ in range(1, n_layers):
            next_ch = min(ch * 2, 512)
            layers += [
                nn.Conv2d(ch, next_ch, 4, stride=2, padding=1, bias=False),
                nn.InstanceNorm2d(next_ch, affine=True),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            ch = next_ch

        layers += [
            nn.Conv2d(ch, ch * 2, 4, stride=1, padding=1, bias=False),
            nn.InstanceNorm2d(ch * 2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(ch * 2, 1, 4, stride=1, padding=1),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# Image buffer (stabilises discriminator training)

class ImageBuffer:
    """Rolling buffer of generated images; returns random mix of old/new."""

    def __init__(self, capacity: int = 50):
        self.capacity = capacity
        self.buffer: list = []

    def push_and_pop(self, images: torch.Tensor) -> torch.Tensor:
        to_return = []
        for img in images:
            img = img.unsqueeze(0)
            if len(self.buffer) < self.capacity:
                self.buffer.append(img)
                to_return.append(img)
            else:
                if np.random.rand() > 0.5:
                    idx = np.random.randint(0, self.capacity)
                    to_return.append(self.buffer[idx].clone())
                    self.buffer[idx] = img
                else:
                    to_return.append(img)
        return torch.cat(to_return, dim=0)


# Dataset: two unpaired domains from PCam (lazy h5py loading)

class UnpairedPatchDataset(Dataset):
    """
    Memory-efficient dataset using lazy h5py reads.

    Domain A = normal patches (y == 0)
    Domain B = tumor  patches (y == 1)

    Only patch indices are held in RAM; images are fetched on demand.
    Set num_workers=0 in DataLoader (h5py is not fork-safe).
    """

    def __init__(self, cfg: dict, split: str = "train"):
        import h5py
        paths  = cfg["paths"]
        vs_cfg = cfg.get("virtual_staining", {})
        max_per_domain: Optional[int] = vs_cfg.get("max_per_domain", None)

        x_key = "pcam_train_x" if split == "train" else f"pcam_{split}_x"
        y_key = {"train": "train_y", "valid": "valid_y", "test": "test_y"}[split]

        self.x_path = ROOT / paths[x_key]

        # Load only labels (small) to build domain index arrays
        with h5py.File(ROOT / paths[y_key], "r") as fy:
            y = fy["y"][:].squeeze()

        idx_A = np.where(y == 0)[0]   # normal
        idx_B = np.where(y == 1)[0]   # tumor

        if max_per_domain is not None:
            rng = np.random.default_rng(42)
            if len(idx_A) > max_per_domain:
                idx_A = rng.choice(idx_A, max_per_domain, replace=False)
                idx_A.sort()
            if len(idx_B) > max_per_domain:
                idx_B = rng.choice(idx_B, max_per_domain, replace=False)
                idx_B.sort()

        self.idx_A = idx_A
        self.idx_B = idx_B

        self.transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),   # -> [-1, 1]
        ])

        print(f"  Domain A (normal): {len(self.idx_A):,}")
        print(f"  Domain B (tumor) : {len(self.idx_B):,}")

    def __len__(self) -> int:
        return max(len(self.idx_A), len(self.idx_B))

    def __getitem__(self, idx: int):
        import h5py
        a_idx = int(self.idx_A[idx % len(self.idx_A)])
        b_idx = int(self.idx_B[idx % len(self.idx_B)])
        with h5py.File(self.x_path, "r") as f:
            img_a = f["x"][a_idx]
            img_b = f["x"][b_idx]
        return self.transform(img_a), self.transform(img_b)


# CycleGAN trainer

class CycleGANTrainer:
    def __init__(self, cfg: dict):
        vs_cfg = cfg["virtual_staining"]
        self.cfg    = cfg
        self.epochs = vs_cfg["epochs"]
        self.lr     = vs_cfg["lr"]
        self.l_cyc  = vs_cfg["lambda_cycle"]
        self.l_id   = vs_cfg["lambda_identity"]

        self.G_AB = CycleGANGenerator().to(DEVICE)
        self.G_BA = CycleGANGenerator().to(DEVICE)
        self.D_A  = PatchGANDiscriminator().to(DEVICE)
        self.D_B  = PatchGANDiscriminator().to(DEVICE)

        self.opt_G  = optim.Adam(
            itertools.chain(self.G_AB.parameters(), self.G_BA.parameters()),
            lr=self.lr, betas=(0.5, 0.999)
        )
        self.opt_DA = optim.Adam(self.D_A.parameters(), lr=self.lr, betas=(0.5, 0.999))
        self.opt_DB = optim.Adam(self.D_B.parameters(), lr=self.lr, betas=(0.5, 0.999))

        def decay_rule(epoch: int) -> float:
            decay_start = self.epochs // 2
            if epoch < decay_start:
                return 1.0
            return max(0.0, 1.0 - (epoch - decay_start) / max(1, decay_start))

        self.sched_G  = optim.lr_scheduler.LambdaLR(self.opt_G,  decay_rule)
        self.sched_DA = optim.lr_scheduler.LambdaLR(self.opt_DA, decay_rule)
        self.sched_DB = optim.lr_scheduler.LambdaLR(self.opt_DB, decay_rule)

        self.crit_gan   = nn.MSELoss()
        self.crit_cycle = nn.L1Loss()
        self.crit_id    = nn.L1Loss()

        self.buf_A = ImageBuffer()
        self.buf_B = ImageBuffer()

        self.models_dir = ROOT / cfg["paths"]["models_dir"]
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def _disc_loss(self, D, real, fake_buf):
        real_lbl = torch.ones_like(D(real))
        fake_lbl = torch.zeros_like(D(fake_buf))
        return 0.5 * (self.crit_gan(D(real), real_lbl) +
                      self.crit_gan(D(fake_buf.detach()), fake_lbl))

    def train(self):
        vs_cfg     = self.cfg["virtual_staining"]
        batch_size = vs_cfg.get("train_batch_size", 4)

        dataset = UnpairedPatchDataset(self.cfg, split="train")
        loader  = DataLoader(
            dataset, batch_size=batch_size, shuffle=True,
            num_workers=0,          # h5py is not fork-safe
            pin_memory=(DEVICE.type == "cuda"),
        )

        print(f"\n{'='*60}")
        print(f"  Phase 6 -- CycleGAN Virtual Staining")
        print(f"  Device     : {DEVICE}")
        print(f"  Epochs     : {self.epochs}")
        print(f"  Batch size : {batch_size}")
        print(f"  Batches/ep : {len(loader)}")
        print(f"{'='*60}")

        best_g_loss = float("inf")

        for epoch in range(1, self.epochs + 1):
            g_losses, da_losses, db_losses = [], [], []

            for real_A, real_B in loader:
                real_A = real_A.to(DEVICE)
                real_B = real_B.to(DEVICE)

                # -- Generator update --
                self.opt_G.zero_grad()

                fake_B = self.G_AB(real_A)
                fake_A = self.G_BA(real_B)
                rec_A  = self.G_BA(fake_B)
                rec_B  = self.G_AB(fake_A)
                id_A   = self.G_BA(real_A)
                id_B   = self.G_AB(real_B)

                loss_adv = (
                    self.crit_gan(self.D_B(fake_B), torch.ones_like(self.D_B(fake_B))) +
                    self.crit_gan(self.D_A(fake_A), torch.ones_like(self.D_A(fake_A)))
                )
                loss_cycle = (self.crit_cycle(rec_A, real_A) +
                              self.crit_cycle(rec_B, real_B)) * self.l_cyc
                loss_id    = (self.crit_id(id_A, real_A) +
                              self.crit_id(id_B, real_B)) * self.l_id

                loss_G = loss_adv + loss_cycle + loss_id
                loss_G.backward()
                self.opt_G.step()
                g_losses.append(loss_G.item())

                # -- Discriminator A --
                self.opt_DA.zero_grad()
                fake_A_buf = self.buf_A.push_and_pop(fake_A.detach())
                loss_DA = self._disc_loss(self.D_A, real_A, fake_A_buf)
                loss_DA.backward()
                self.opt_DA.step()
                da_losses.append(loss_DA.item())

                # -- Discriminator B --
                self.opt_DB.zero_grad()
                fake_B_buf = self.buf_B.push_and_pop(fake_B.detach())
                loss_DB = self._disc_loss(self.D_B, real_B, fake_B_buf)
                loss_DB.backward()
                self.opt_DB.step()
                db_losses.append(loss_DB.item())

            self.sched_G.step()
            self.sched_DA.step()
            self.sched_DB.step()

            mean_g = np.mean(g_losses)
            if epoch % 5 == 0 or epoch == 1:
                print(f"  Epoch {epoch:03d}/{self.epochs}  "
                      f"G={mean_g:.4f}  "
                      f"D_A={np.mean(da_losses):.4f}  "
                      f"D_B={np.mean(db_losses):.4f}",
                      flush=True)

            # Save best checkpoint
            if mean_g < best_g_loss:
                best_g_loss = mean_g
                self._save(suffix="best")

        # Always save final
        self._save(suffix="final")
        print(f"  Training complete. Best G loss: {best_g_loss:.4f}")

    def _save(self, suffix: str = "final"):
        path = self.models_dir / f"cyclegan_{suffix}.pth"
        torch.save({
            "G_AB": self.G_AB.state_dict(),
            "G_BA": self.G_BA.state_dict(),
        }, path)
        # Keep canonical name pointing to best
        canonical = self.models_dir / "cyclegan.pth"
        torch.save({
            "G_AB": self.G_AB.state_dict(),
            "G_BA": self.G_BA.state_dict(),
        }, canonical)

    def load(self):
        ckpt_path = self.models_dir / "cyclegan.pth"
        if not ckpt_path.exists():
            raise FileNotFoundError(f"CycleGAN checkpoint not found: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
        self.G_AB.load_state_dict(ckpt["G_AB"])
        self.G_BA.load_state_dict(ckpt["G_BA"])
        self.G_AB.eval()
        self.G_BA.eval()
        print("  Loaded CycleGAN weights.")


# Inference helpers

def _to_uint8(tensor_batch: torch.Tensor) -> np.ndarray:
    """Convert a (B,3,H,W) tensor in [-1,1] to (B,H,W,3) uint8."""
    out = (tensor_batch * 0.5 + 0.5).clamp(0, 1)
    out = out.permute(0, 2, 3, 1).cpu().numpy()
    return (out * 255).astype(np.uint8)


def virtual_stain(
    image: np.ndarray,
    G: CycleGANGenerator,
    direction: str = "A2B",
) -> np.ndarray:
    """Apply virtual staining to a single (96,96,3) uint8 patch."""
    G.eval()
    transform = T.Compose([
        T.ToTensor(),
        T.Normalize([0.5] * 3, [0.5] * 3),
    ])
    x = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = G(x)
    return _to_uint8(out)[0]


def run_full_inference(
    cfg: dict,
    trainer: CycleGANTrainer,
    split: str = "test",
    batch_size: int = 64,
) -> str:
    """
    Run G_AB on ALL patches of `split`.
    Saves virtual-stained patches to reports/virtual_stained_<split>.npy.
    Returns the path to the saved file.
    """
    import h5py

    paths = cfg["paths"]
    x_key = {"train": "pcam_train_x",
             "valid": "pcam_valid_x",
             "test":  "pcam_test_x"}[split]
    x_path   = ROOT / paths[x_key]
    save_dir = ROOT / paths["reports_dir"]
    save_dir.mkdir(parents=True, exist_ok=True)
    out_path = save_dir / f"virtual_stained_{split}.npy"

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize([0.5] * 3, [0.5] * 3),
    ])

    trainer.G_AB.eval()

    with h5py.File(x_path, "r") as f:
        n_patches = f["x"].shape[0]
        print(f"  Running G_AB on {n_patches:,} {split} patches "
              f"(batch={batch_size}) ...", flush=True)

        stained = np.empty((n_patches, 96, 96, 3), dtype=np.uint8)

        for start in range(0, n_patches, batch_size):
            end   = min(start + batch_size, n_patches)
            batch = f["x"][start:end]                         # (B,96,96,3)
            tensors = torch.stack([transform(img) for img in batch]).to(DEVICE)
            with torch.no_grad():
                out = trainer.G_AB(tensors)
            stained[start:end] = _to_uint8(out)

            if start % (batch_size * 100) == 0 and start > 0:
                print(f"    {end:>6}/{n_patches}  ({end/n_patches*100:.1f}%)",
                      flush=True)

    np.save(str(out_path), stained)
    gb = stained.nbytes / 1e9
    print(f"  Saved --> {out_path}  ({gb:.2f} GB)", flush=True)
    return str(out_path)


# Visualisation

def save_visualization(cfg: dict, trainer: CycleGANTrainer, n: int = 12):
    """
    Save a before / virtual-stained / cycle-restored grid.
    Pulls n patches from the test set (does NOT require the .npy file).
    """
    import h5py

    paths    = cfg["paths"]
    x_path   = ROOT / paths["pcam_test_x"]
    save_dir = ROOT / paths["reports_dir"]
    save_dir.mkdir(parents=True, exist_ok=True)

    trainer.G_AB.eval()
    trainer.G_BA.eval()

    with h5py.File(x_path, "r") as f:
        patches = f["x"][:n]

    fig, axes = plt.subplots(n, 3, figsize=(10, n * 3))
    fig.suptitle("CycleGAN Virtual Staining -- Full Model", fontsize=14, fontweight="bold")

    for i, patch in enumerate(patches):
        stained   = virtual_stain(patch, trainer.G_AB, "A2B")
        restored  = virtual_stain(stained, trainer.G_BA, "B2A")

        axes[i, 0].imshow(patch);   axes[i, 0].set_title("Original H&E",         fontsize=8)
        axes[i, 1].imshow(stained); axes[i, 1].set_title("Virtual Stain (G_AB)", fontsize=8)
        axes[i, 2].imshow(restored);axes[i, 2].set_title("Cycle Restored",        fontsize=8)
        for ax in axes[i]:
            ax.axis("off")

    plt.tight_layout()
    out_path = save_dir / "virtual_stain_he_to_ihc.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Visualization saved --> {out_path}", flush=True)


# Main pipeline entry point (called by run_full_pipeline.py)

def train_and_infer(cfg: dict):
    """
    Full Phase 6 pipeline:
      1. Train CycleGAN on ALL training data (or resume from checkpoint)
      2. Run G_AB inference on train / valid / test -> .npy per split
      3. Save visualization grid
    """
    vs_cfg      = cfg.get("virtual_staining", {})
    infer_batch = vs_cfg.get("infer_batch_size", 64)

    trainer   = CycleGANTrainer(cfg)
    ckpt_path = ROOT / cfg["paths"]["models_dir"] / "cyclegan.pth"

    if ckpt_path.exists():
        print("  Found existing CycleGAN checkpoint -- skipping training.", flush=True)
        trainer.load()
    else:
        trainer.train()

    # Run inference on test split only (train/valid would create 10+ GB files)
    reports_dir = ROOT / cfg["paths"]["reports_dir"]
    for split in ("test",):
        out_npy = reports_dir / f"virtual_stained_{split}.npy"
        if out_npy.exists():
            print(f"  [{split}] already exists -- skipping inference.", flush=True)
        else:
            run_full_inference(cfg, trainer, split=split, batch_size=infer_batch)

    # Visualisation
    save_visualization(cfg, trainer, n=12)


# Kept for backward compatibility with visualization scripts

def demo_virtual_staining(cfg: dict, n: int = 12):
    """
    Show before/after virtual staining.
    If a trained checkpoint exists, uses the real CycleGAN.
    Otherwise falls back to stain augmentation (no-model demo).
    """
    import h5py

    trainer   = CycleGANTrainer(cfg)
    ckpt_path = ROOT / cfg["paths"]["models_dir"] / "cyclegan.pth"

    if ckpt_path.exists():
        trainer.load()
        save_visualization(cfg, trainer, n=n)
        return

    # No model yet -- stain augmentation fallback
    print("  No trained CycleGAN found -- running stain augmentation fallback.")
    with h5py.File(ROOT / cfg["paths"]["pcam_test_x"], "r") as f:
        patches = f["x"][:n]

    tf = T.Compose([
        T.ToTensor(),
        T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        T.ToPILImage(),
    ])

    fig, axes = plt.subplots(n, 2, figsize=(6, n * 3))
    fig.suptitle("Virtual Staining -- Stain Augmentation Fallback", fontsize=13)
    for i, patch in enumerate(patches):
        axes[i, 0].imshow(patch);                axes[i, 0].set_title("Original H&E", fontsize=8)
        axes[i, 1].imshow(np.array(tf(patch)));  axes[i, 1].set_title("Augmented",    fontsize=8)
        for ax in axes[i]:
            ax.axis("off")

    plt.tight_layout()
    save_path = ROOT / cfg["paths"]["reports_dir"] / "virtual_stain_he_to_ihc.png"
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved --> {save_path}", flush=True)


# Entry point

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phase 6 -- CycleGAN Virtual Staining")
    parser.add_argument("--train",  action="store_true", help="Train CycleGAN")
    parser.add_argument("--infer",  action="store_true", help="Run inference on all splits")
    parser.add_argument("--demo",   action="store_true", help="Visualisation only")
    parser.add_argument("--epochs", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config()
    if args.epochs:
        cfg["virtual_staining"]["epochs"] = args.epochs

    if args.train or (not args.demo and not args.infer):
        train_and_infer(cfg)
    elif args.infer:
        t = CycleGANTrainer(cfg)
        t.load()
        for split in ("train", "valid", "test"):
            run_full_inference(cfg, t, split=split)
    elif args.demo:
        demo_virtual_staining(cfg)

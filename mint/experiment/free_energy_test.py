# tests/test_free_energy.py
from __future__ import annotations

import os
import math
import numpy as np
import torch

# Use non-interactive backend for CI / servers
import matplotlib
matplotlib.use("Agg")

from omegaconf import OmegaConf

from mint.experiment.free_energy import FreeEnergy


def compute_dihedrals(
    positions: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    """
    positions : (T, B, N, 3)
    indices   : (M, 4)
    returns   : (T, B, M)
    """
    vecs = positions[:, :, indices]  # (T, B, M, 4, 3)
    r1, r2, r3, r4 = vecs[..., 0, :], vecs[..., 1, :], vecs[..., 2, :], vecs[..., 3, :]

    b1 = r2 - r1
    b2 = r3 - r2
    b3 = r4 - r3

    n1 = torch.nn.functional.normalize(torch.cross(b1, b2, dim=-1), dim=-1)
    n2 = torch.nn.functional.normalize(torch.cross(b2, b3, dim=-1), dim=-1)
    b2n = torch.nn.functional.normalize(b2, dim=-1)

    cos_phi = (n1 * n2).sum(dim=-1).clamp(-1.0, 1.0)
    sin_phi = (torch.cross(n1, n2, dim=-1) * b2n).sum(dim=-1)

    return torch.atan2(sin_phi, cos_phi)


def assert_density_normalized(edges, density, atol=1e-3):
    """Check sum(density * bin_volume) ~= 1."""
    widths = [np.diff(e) for e in edges]
    D = len(widths)
    vol = 1.0
    for d, w in enumerate(widths):
        shape = [1] * D
        shape[d] = w.shape[0]
        vol = vol * w.reshape(shape)
    total = float(np.sum(density * vol))
    assert abs(total - 1.0) < atol, f"density not normalized: sum(density*vol)={total}"


def test_tensor_input_D2(outdir="tmp_free_energy_test"):
    os.makedirs(outdir, exist_ok=True)

    # Fake trajectory: (T, N, 3)
    T, N = 2000, 22
    X = torch.randn(T, N, 3)

    # Observable: two "dihedral-like" angles (here we actually compute dihedrals from 4-atom tuples)
    # Need (T, B, N, 3)
    positions = X.unsqueeze(1)  # (T,1,N,3)
    indices = torch.tensor([[1, 2, 3, 4],
                            [5, 6, 7, 8]], dtype=torch.long)

    def f(traj_TN3: torch.Tensor):
        pos = traj_TN3.unsqueeze(1)  # (T,1,N,3)
        dih = compute_dihedrals(pos, indices)  # (T,1,2)
        # Flatten to (T,2)
        return dih[:, 0, :]

    cfg = OmegaConf.create({
        "bins": [128, 128],
        "min": [-math.pi, -math.pi],
        "max": [ math.pi,  math.pi],
        "epsilon": 1e-12,
        "smoothing": False,
        "kBT": 1.0,
        "labels": ["phi", "psi"],
        "cmap_prob": "viridis",
        "cmap_energy": "magma",
        "shift_energy": True,
        "colorbar": True,
    })

    exp = FreeEnergy(state=None, cfg=cfg, samples=X, function=f)
    grid, prob, F, fig_p, fig_F, stats = exp.run()

    # Basic sanity checks
    assert isinstance(grid, list) and len(grid) == 2
    assert prob.ndim == 2 and F.ndim == 2
    assert prob.shape == F.shape
    assert stats.mean.shape == (2,)
    assert stats.quantiles.shape[1] == 2

    # Normalization check
    assert_density_normalized(stats.edges, prob, atol=2e-3)

    # Save figures
    fig_p.savefig(os.path.join(outdir, "prob_D2.png"), dpi=150)
    fig_F.savefig(os.path.join(outdir, "free_energy_D2.png"), dpi=150)

    print("[OK] test_tensor_input_D2")
    print("  Saved:", os.path.join(outdir, "prob_D2.png"))
    print("  Saved:", os.path.join(outdir, "free_energy_D2.png"))


def test_generate_like_list_input_D1(outdir="tmp_free_energy_test"):
    os.makedirs(outdir, exist_ok=True)

    # Create "Generate-like" samples: list of dicts with 'x' and 'batch'
    # We'll create 10 batches; each batch contains B graphs, each graph has N nodes (same N for simplicity).
    num_steps = 10
    B, N = 8, 22
    samples = []

    for _ in range(num_steps):
        dense = torch.randn(B, N, 3)  # pretend this is after to_dense_batch
        flat = dense.reshape(B * N, 3)
        batch_index = torch.arange(B).repeat_interleave(N)
        samples.append({"x": flat, "batch": batch_index})

    # 1D observable: distance between atom 0 and atom 1 per frame (needs dense)
    def f(dense_or_traj):
        # After your FreeEnergy densify logic, input should be (steps, B, N, 3) OR concatenated.
        # In this synthetic test, FreeEnergy will collect and cat dict-extracted tensors along dim=0.
        # If torch_geometric is installed, each dict becomes (B,N,3) and cat -> (num_steps*B, N, 3).
        X = dense_or_traj
        if X.ndim == 3:  # (TB, N, 3)
            d = torch.norm(X[:, 0, :] - X[:, 1, :], dim=-1)  # (TB,)
            return d
        elif X.ndim == 4:  # (T, B, N, 3)
            d = torch.norm(X[:, :, 0, :] - X[:, :, 1, :], dim=-1)  # (T,B)
            return d.reshape(-1)
        else:
            raise ValueError(f"Unexpected X shape: {tuple(X.shape)}")

    cfg = OmegaConf.create({
        "bins": 100,
        "epsilon": 1e-12,
        "smoothing": False,
        "kBT": 1.0,
        "labels": ["d01"],
        "shift_energy": True,
    })

    exp = FreeEnergy(state=None, cfg=cfg, samples=samples, function=f)
    grid, prob, F, fig_p, fig_F, stats = exp.run()

    assert isinstance(grid, list) and len(grid) == 1
    assert prob.ndim == 1 and F.ndim == 1
    assert prob.shape == F.shape
    assert stats.mean.shape == (1,)

    # Normalization check (1D)
    assert_density_normalized(stats.edges, prob, atol=2e-3)

    fig_p.savefig(os.path.join(outdir, "prob_D1.png"), dpi=150)
    fig_F.savefig(os.path.join(outdir, "free_energy_D1.png"), dpi=150)

    print("[OK] test_generate_like_list_input_D1")
    print("  Saved:", os.path.join(outdir, "prob_D1.png"))
    print("  Saved:", os.path.join(outdir, "free_energy_D1.png"))


if __name__ == "__main__":
    # Run both tests
    test_tensor_input_D2()
    test_generate_like_list_input_D1()
    print("\nAll FreeEnergy tests passed.")
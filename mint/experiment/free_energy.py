from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

import torch
import matplotlib.pyplot as plt

from scipy.ndimage import gaussian_filter

from torch_geometric.utils import to_dense_batch  

from mint.experiment.abstract import Experiment
from mint.state import MINTState


@dataclass(frozen=True)
class FreeEnergyStats:
    """Summary statistics for the observable samples f(q)."""

    mean: np.ndarray  # (D,)
    std: np.ndarray  # (D,)
    quantiles: np.ndarray  # (Q, D)
    quantile_levels: Tuple[float, ...]
    num_samples: int

    # Helpful extras for downstream use / plotting
    edges: List[np.ndarray]
    centers: List[np.ndarray]


def _as_numpy(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _cfg_get(cfg: Any, key: str, default: Any) -> Any:
    """Config getter compatible with dict / OmegaConf-like / simple objects."""
    if cfg is None:
        return default
    if hasattr(cfg, "get"):
        try:
            return cfg.get(key, default)
        except TypeError:
            # Some configs use get(key) without default.
            try:
                v = cfg.get(key)
                return default if v is None else v
            except Exception:
                pass
    return getattr(cfg, key, default)


def _broadcast_bins(bins: Any, D: int) -> List[int]:
    # scalar -> repeat
    if np.isscalar(bins):
        return [int(bins)] * D

    # OmegaConf ListConfig / other sequences -> list(...) then validate
    if hasattr(bins, "__len__") and hasattr(bins, "__iter__") and not isinstance(bins, (str, bytes)):
        bins_list = list(bins)
        if len(bins_list) != D:
            raise ValueError(f"'bins' must have length D={D}, got {len(bins_list)}")
        return [int(b) for b in bins_list]

    raise TypeError(f"Unsupported type for 'bins': {type(bins)}")


def _broadcast_param(value: Any, D: int, name: str) -> List[float]:
    if value is None:
        raise ValueError(f"'{name}' must be provided (or inferrable)")

    # scalar -> repeat
    if np.isscalar(value):
        return [float(value)] * D

    # OmegaConf ListConfig / other sequences -> list(...) then validate
    if hasattr(value, "__len__") and hasattr(value, "__iter__") and not isinstance(value, (str, bytes)):
        v_list = list(value)
        if len(v_list) != D:
            raise ValueError(f"'{name}' must have length D={D}, got {len(v_list)}")
        return [float(v) for v in v_list]

    raise TypeError(f"Unsupported type for '{name}': {type(value)}")


def _centers_from_edges(edges: Sequence[np.ndarray]) -> List[np.ndarray]:
    return [0.5 * (e[1:] + e[:-1]) for e in edges]


def _widths_from_edges(edges: Sequence[np.ndarray]) -> List[np.ndarray]:
    return [np.diff(e) for e in edges]


def _volume_broadcast(widths: Sequence[np.ndarray]) -> np.ndarray:
    """Broadcast per-dimension bin widths into a full D-dim volume array."""
    D = len(widths)
    vol = 1.0
    for d, w in enumerate(widths):
        shape = [1] * D
        shape[d] = w.shape[0]
        vol = vol * w.reshape(shape)
    return np.asarray(vol, dtype=float)


def _marginalize_density(
    density: np.ndarray,
    keep_axes: Sequence[int],
    widths: Sequence[np.ndarray],
) -> np.ndarray:
    """
    Integrate a D-dim *density* down to 1D/2D by integrating out other axes.

    We assume `density` is a density w.r.t. bin volumes, i.e.:
        sum(density * vol) = 1

    Marginalization is numerical integration:
      - multiply by the dropped dimensions' bin widths
      - sum over dropped axes

    This matches the task requirement to define how higher-D histograms are
    marginalized in a corner.py-style visualization. :contentReference[oaicite:4]{index=4}
    """
    D = density.ndim
    keep_axes = list(keep_axes)
    drop_axes = [ax for ax in range(D) if ax not in keep_axes]

    integrand = density
    for ax in drop_axes:
        shape = [1] * D
        shape[ax] = widths[ax].shape[0]
        integrand = integrand * widths[ax].reshape(shape)

    for ax in sorted(drop_axes, reverse=True):
        integrand = integrand.sum(axis=ax)

    return integrand


def _extract_positions_from_batch(batch: Any) -> Any:
    """
    Best-effort extraction for common batch formats.

    Supports:
      - torch.Tensor / np.ndarray directly
      - dict batches:
          * uses common keys like 'x', 'pos', 'positions', ...
          * if both ('x' and 'batch') exist and torch_geometric is available,
            densifies graph nodes via to_dense_batch(x, batch)[0]
    """
    if isinstance(batch, torch.Tensor) or isinstance(batch, np.ndarray):
        return batch

    # torch_geometric Data object style
    if hasattr(batch, "pos"):
        return batch.pos

    if isinstance(batch, dict):
        # Prefer these keys
        for key in ("x", "pos", "positions", "X", "coords", "q", "x_target", "x_t", "x_base"):
            if key in batch:
                x = batch[key]
                # Handle graph-style: (total_nodes, 3) + batch vector
                if key == "x" and ("batch" in batch) and (to_dense_batch is not None):
                    try:
                        dense, mask = to_dense_batch(batch["x"], batch["batch"])
                        # If variable-sized graphs exist, users should handle masking in f(·).
                        # Here we return the padded dense tensor.
                        return dense
                    except Exception:
                        return x
                return x

        # Fallback: first tensor/ndarray value
        for v in batch.values():
            if isinstance(v, torch.Tensor) or isinstance(v, np.ndarray):
                return v

    return batch


def _infer_kBT(cfg: Any, *, kBT: Optional[float], kB: Optional[float], T: Optional[float]) -> float:
    """
    Thermodynamics parameter precedence:
      1) explicit kBT arg
      2) cfg.kBT
      3) explicit (kB, T) args -> kB*T
      4) cfg.(kB, T) -> kB*T
      5) default 1.0

    Task allows either (kB, T) or combined kBT. :contentReference[oaicite:5]{index=5}
    """
    if kBT is not None:
        return float(kBT)
    cfg_kbt = _cfg_get(cfg, "kBT", None)
    if cfg_kbt is not None:
        return float(cfg_kbt)

    if (kB is not None) and (T is not None):
        return float(kB) * float(T)

    cfg_kb = _cfg_get(cfg, "kB", None)
    cfg_T = _cfg_get(cfg, "T", None)
    if (cfg_kb is not None) and (cfg_T is not None):
        return float(cfg_kb) * float(cfg_T)

    return 1.0


class FreeEnergy(Experiment):
    """
    Free energy along a provided coordinate.

    Given samples q ~ ρ(q) and a user observable f(q) in R^D, we:
      1) compute f(q_i)
      2) build a D-dim histogram-based density estimate ρ_b on a grid
      3) compute free energy F = -kBT * ln ρ_b

    Zero-probability bins:
      We apply epsilon-regularization by adding a small constant epsilon to every bin
      before taking log, then renormalizing. This is consistent and documented as
      required by the task. :contentReference[oaicite:6]{index=6}

    Inputs:
      - state: MINTState (optional usage)
      - samples: tensor/array shaped (T, N, 3) OR an iterable/dataloader producing them
      - function: f(·) mapping samples -> R^D
      - cfg: histogram/grid config (bins, min/max, smoothing, etc.)
      - thermodynamics: (kB, T) or kBT :contentReference[oaicite:7]{index=7}

    Returns from run():
      (grid, probability, free_energy, fig_prob, fig_energy, stats) :contentReference[oaicite:8]{index=8}
    """

    def __init__(
        self,
        *,
        state: Optional[MINTState],
        cfg: Optional[Any],
        samples: Union[np.ndarray, torch.Tensor, Iterable[Any]],
        function: Callable[[Any], Any],
        kBT: Optional[float] = None,
        kB: Optional[float] = None,
        T: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.state = state
        self.cfg = cfg
        self.samples = samples
        self.function = function
        self.kBT = _infer_kBT(cfg, kBT=kBT, kB=kB, T=T)

    def __repr__(self) -> str:  # helpful "descriptive print" for experiments
        return f"FreeEnergy(kBT={self.kBT}, cfg={self.cfg})"

    def _collect_samples(self) -> Any:
        s = self.samples
        if isinstance(s, torch.Tensor) or isinstance(s, np.ndarray):
            return s

        if isinstance(s, Iterable):
            chunks: List[Any] = []
            for batch in s:
                x = _extract_positions_from_batch(batch)
                chunks.append(x)

            if len(chunks) == 0:
                raise ValueError("Empty samples iterable/dataloader")

            # If any are torch tensors -> torch.cat
            if any(isinstance(c, torch.Tensor) for c in chunks):
                chunks_t = [c if isinstance(c, torch.Tensor) else torch.as_tensor(c) for c in chunks]
                try:
                    return torch.cat(chunks_t, dim=0)
                except Exception as e:
                    raise ValueError(
                        "Failed to concatenate sample chunks with torch.cat(dim=0). "
                        "Make sure all batches have the same shape except for the leading dimension."
                    ) from e

            # Otherwise numpy concatenate
            try:
                return np.concatenate([np.asarray(c) for c in chunks], axis=0)
            except Exception as e:
                raise ValueError(
                    "Failed to concatenate sample chunks with np.concatenate(axis=0). "
                    "Make sure all batches have the same shape except for the leading dimension."
                ) from e

        return s

    def _compute_observable(self, X: Any) -> np.ndarray:
        f_val = self.function(X)
        f_np = _as_numpy(f_val)
        if f_np.ndim == 1:
            f_np = f_np[:, None]
        if f_np.ndim != 2:
            raise ValueError(f"function must return shape (M, D) or (M,), got {f_np.shape}")
        return np.asarray(f_np, dtype=float)

    def _labels(self, D: int) -> List[str]:
        labels = _cfg_get(self.cfg, "labels", None)
        if labels is None:
            return [f"f{d}" for d in range(D)]
        if len(labels) != D:
            raise ValueError(f"labels must have length D={D}")
        return list(labels)

    def _histogram(self, f_np: np.ndarray) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
        D = f_np.shape[1]
        bins_list = _broadcast_bins(_cfg_get(self.cfg, "bins", 100), D)

        cfg_min = _cfg_get(self.cfg, "min", None)
        cfg_max = _cfg_get(self.cfg, "max", None)
        if cfg_min is None:
            cfg_min = np.min(f_np, axis=0)
        if cfg_max is None:
            cfg_max = np.max(f_np, axis=0)

        mins = _broadcast_param(cfg_min, D, "min")
        maxs = _broadcast_param(cfg_max, D, "max")

        edges: List[np.ndarray] = []
        for d in range(D):
            if maxs[d] <= mins[d]:
                raise ValueError(f"Invalid bounds for dim {d}: min={mins[d]} max={maxs[d]}")
            edges.append(np.linspace(mins[d], maxs[d], bins_list[d] + 1, dtype=float))

        counts, edges_out = np.histogramdd(f_np, bins=edges, density=False)
        edges = [np.asarray(e, dtype=float) for e in edges_out]
        widths = _widths_from_edges(edges)
        vol = _volume_broadcast(widths)
        counts = counts.astype(float)

        # Base density estimate: density is w.r.t. volumes so sum(density*vol)=1
        density = counts / (np.sum(counts) * vol)

        # Optional smoothing as required by task config :contentReference[oaicite:9]{index=9}
        smoothing = bool(_cfg_get(self.cfg, "smoothing", False))
        if smoothing:
            method = str(_cfg_get(self.cfg, "smoothing_method", "gaussian")).lower()
            if method != "gaussian":
                raise ValueError("Only smoothing_method='gaussian' is supported")
            if gaussian_filter is None:
                raise ImportError("smoothing=True requires scipy (scipy.ndimage.gaussian_filter)")
            sigma = float(_cfg_get(self.cfg, "smoothing_sigma", 1.0))
            smoothing_on = str(_cfg_get(self.cfg, "smoothing_on", "counts")).lower()
            if smoothing_on not in {"counts", "probability"}:
                raise ValueError("smoothing_on must be 'counts' or 'probability'")

            if smoothing_on == "counts":
                smoothed_counts = gaussian_filter(counts, sigma=sigma, mode="nearest")
                density = smoothed_counts / (np.sum(smoothed_counts) * vol)
            else:
                density = gaussian_filter(density, sigma=sigma, mode="nearest")
                density = density / np.sum(density * vol)

        # Epsilon-regularization for zeros (documented behavior). :contentReference[oaicite:10]{index=10}
        epsilon = float(_cfg_get(self.cfg, "epsilon", 1e-12))
        density = density + epsilon
        density = density / np.sum(density * vol)

        free_energy = -self.kBT * np.log(density)

        if bool(_cfg_get(self.cfg, "shift_energy", True)):
            # Free energy is defined up to an additive constant; shift min to 0.
            finite = free_energy[np.isfinite(free_energy)]
            if finite.size:
                free_energy = free_energy - float(np.min(finite))

        return edges, density, free_energy

    def _plot_1d(
        self,
        x: np.ndarray,
        prob: np.ndarray,
        energy: np.ndarray,
        label: str,
        cmap_prob: str,
        cmap_energy: str,
    ):
        fig_p, ax_p = plt.subplots(figsize=(6, 4))
        ax_p.plot(x, prob, color=plt.get_cmap(cmap_prob)(0.7))
        ax_p.set_xlabel(label)
        ax_p.set_ylabel("ρ(f)")
        ax_p.set_title("Induced density")
        fig_p.tight_layout()

        fig_e, ax_e = plt.subplots(figsize=(6, 4))
        ax_e.plot(x, energy, color=plt.get_cmap(cmap_energy)(0.7))
        ax_e.set_xlabel(label)
        ax_e.set_ylabel("F(f)")
        ax_e.set_title("Free energy")
        fig_e.tight_layout()

        return fig_p, fig_e

    def _plot_corner(
        self,
        edges: List[np.ndarray],
        density: np.ndarray,
        *,
        cmap: str,
        title: str,
        as_energy: bool,
    ):
        """
        Corner.py-style layout:

          - diagonal: 1D marginals
          - off-diagonal (lower triangle): pairwise 2D marginals
          - upper triangle: blank

        This matches task rules for D>2 and also works for D=2. :contentReference[oaicite:11]{index=11}
        """
        D = density.ndim
        widths = _widths_from_edges(edges)
        centers = _centers_from_edges(edges)
        labels = self._labels(D)
        log_prob = bool(_cfg_get(self.cfg, "log_prob", False))

        # For energy plots, use a consistent color scale across panels.
        vmin = vmax = None
        if as_energy:
            eps = float(_cfg_get(self.cfg, "epsilon", 1e-12))
            energies: List[np.ndarray] = []

            for i in range(D):
                p_i = _marginalize_density(density, [i], widths)
                e_i = -self.kBT * np.log(p_i + eps)
                if bool(_cfg_get(self.cfg, "shift_energy", True)):
                    finite = e_i[np.isfinite(e_i)]
                    if finite.size:
                        e_i = e_i - float(np.min(finite))
                energies.append(e_i)

            for i in range(1, D):
                for j in range(i):
                    p_ji = _marginalize_density(density, [j, i], widths)
                    e_ji = -self.kBT * np.log(p_ji + eps)
                    if bool(_cfg_get(self.cfg, "shift_energy", True)):
                        finite = e_ji[np.isfinite(e_ji)]
                        if finite.size:
                            e_ji = e_ji - float(np.min(finite))
                    energies.append(e_ji)

            finite_all = np.concatenate([e[np.isfinite(e)].ravel() for e in energies if np.isfinite(e).any()])
            if finite_all.size:
                vmin, vmax = float(np.min(finite_all)), float(np.max(finite_all))

        fig, axes = plt.subplots(D, D, figsize=(2.4 * D, 2.4 * D), squeeze=False)

        # Optionally add one shared mappable for a single colorbar (energy only).
        last_mappable = None

        for i in range(D):
            for j in range(D):
                ax = axes[i, j]
                if i < j:
                    ax.axis("off")
                    continue

                if i == j:
                    p_i = _marginalize_density(density, [i], widths)
                    if as_energy:
                        eps = float(_cfg_get(self.cfg, "epsilon", 1e-12))
                        y = -self.kBT * np.log(p_i + eps)
                        if bool(_cfg_get(self.cfg, "shift_energy", True)):
                            finite = y[np.isfinite(y)]
                            if finite.size:
                                y = y - float(np.min(finite))
                        ax.plot(centers[i], y, color=plt.get_cmap(cmap)(0.7))
                        ax.set_ylabel("F")
                    else:
                        ax.plot(centers[i], p_i, color=plt.get_cmap(cmap)(0.7))
                        ax.set_ylabel("ρ")
                    ax.set_xlabel(labels[i])
                    continue

                # Off-diagonal: 2D marginal for (j, i)
                p_ji = _marginalize_density(density, [j, i], widths)
                if as_energy:
                    eps = float(_cfg_get(self.cfg, "epsilon", 1e-12))
                    z = -self.kBT * np.log(p_ji + eps)
                    if bool(_cfg_get(self.cfg, "shift_energy", True)):
                        finite = z[np.isfinite(z)]
                        if finite.size:
                            z = z - float(np.min(finite))
                    last_mappable = ax.pcolormesh(
                        edges[j], edges[i], z.T, shading="auto", cmap=cmap, vmin=vmin, vmax=vmax
                    )
                else:
                    plot_data = np.log10(p_ji + 1e-12) if log_prob else p_ji
                    last_mappable = ax.pcolormesh(edges[j], edges[i], plot_data.T, shading="auto", cmap=cmap)

                if i == D - 1:
                    ax.set_xlabel(labels[j])
                if j == 0:
                    ax.set_ylabel(labels[i])

        fig.suptitle(title)
        fig.tight_layout()

        # Single shared colorbar (useful especially for D=2)
        if last_mappable is not None and bool(_cfg_get(self.cfg, "colorbar", True)):
            fig.colorbar(last_mappable, ax=axes, fraction=0.02, pad=0.01)

        return fig

    def run(self):
        # 1) Collect q samples
        X = self._collect_samples()

        # 2) Compute observable samples f(q) in R^D
        f_np = self._compute_observable(X)
        D = f_np.shape[1]

        # 3) Stats required by task :contentReference[oaicite:12]{index=12}
        quantile_levels = tuple(float(q) for q in _cfg_get(self.cfg, "quantiles", [0.05, 0.5, 0.95]))
        quantiles = np.quantile(f_np, quantile_levels, axis=0)

        # 4) Histogram density + free energy
        edges, density, free_energy = self._histogram(f_np)
        centers = _centers_from_edges(edges)

        stats = FreeEnergyStats(
            mean=np.mean(f_np, axis=0),
            std=np.std(f_np, axis=0, ddof=0),
            quantiles=quantiles,
            quantile_levels=quantile_levels,
            num_samples=int(f_np.shape[0]),
            edges=edges,
            centers=centers,
        )

        # 5) Plotting rules by D :contentReference[oaicite:13]{index=13}
        cmap_prob = str(_cfg_get(self.cfg, "cmap_prob", "viridis"))
        cmap_energy = str(_cfg_get(self.cfg, "cmap_energy", "magma"))

        if D == 1:
            fig_prob, fig_energy = self._plot_1d(
                centers[0],
                density,
                free_energy,
                label=self._labels(1)[0],
                cmap_prob=cmap_prob,
                cmap_energy=cmap_energy,
            )
        else:
            fig_prob = self._plot_corner(edges, density, cmap=cmap_prob, title="Induced density", as_energy=False)
            fig_energy = self._plot_corner(edges, density, cmap=cmap_energy, title="Free energy", as_energy=True)

        # Return contract :contentReference[oaicite:14]{index=14}
        grid = centers
        return grid, density, free_energy, fig_prob, fig_energy, stats

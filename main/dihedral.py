import torch
import numpy as np
import math
from scipy.ndimage import gaussian_filter
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
from tqdm.auto import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Relevant functions

def read_xyz_file(filepath):
    """
    Read XYZ file and extract atomic coordinates.

    Returns:
        positions: numpy array of shape (n_frames, n_atoms, 3)
        elements: list of element symbols
    """
    frames = []
    elements = []

    with open(filepath, 'r') as f:
        lines = f.readlines()

    i = 0
    while i < len(lines):
        try:
            n_atoms = int(lines[i].strip())
        except ValueError:
            break

        i += 2  # Skip comment line

        coords = []
        frame_elements = []
        for j in range(n_atoms):
            parts = lines[i].split()
            frame_elements.append(parts[0])
            coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
            i += 1

        frames.append(np.array(coords, dtype=np.float64))
        if not elements:
            elements = frame_elements

    positions = np.array(frames, dtype=np.float64)  # (T, N, 3)
    return positions, elements

def compute_dihedrals(
    positions: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    """
    Compute signed dihedral angles from 4-atom indices.

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

# Load datasets

gen_positions_np, gen_elements = read_xyz_file('output.xyz')   # (T_gen, N, 3)

# Element-type IDs
positions = torch.tensor(gen_positions_np, dtype=torch.float32).unsqueeze(dim=1).to(device)  # (T_gen,1,N,3)

# Dihedral indices (defined in dataset.xyz indexing)
indices = torch.tensor([[6, 8, 14, 16],
                        [4, 6, 8, 14]], dtype=torch.long, device=device)

# Ramachandran + free energy
bins = 300
extent = [-math.pi, math.pi, -math.pi, math.pi]
range_ = [extent[:2], extent[2:]]

with torch.no_grad():
    dihedrals = compute_dihedrals(positions, indices)  # (T_gen, 1, 2)
data = dihedrals.reshape(-1, 2).cpu().numpy()

hist, _, _ = np.histogram2d(
    data[:, 0], data[:, 1],
    bins=bins, range=range_, density=True
)
hist_smooth = gaussian_filter(hist, sigma=5.0)

# Probability density
fig, ax = plt.subplots(figsize=(6, 5), dpi=120)
contours = ax.contour(
    hist_smooth, levels=50, linewidths=0.6, cmap='viridis',
    extent=extent, origin='lower', antialiased=True
)
fig.colorbar(contours, ax=ax, label="Probability Density")
ax.set_xlabel("ϕ (radians)")
ax.set_ylabel("ψ (radians)")
ax.set_title("ϕ–ψ Ramachandran Type Plot")
ax.set_aspect('equal')
plt.tight_layout()
plt.savefig('Probability')
plt.close(fig)

# Free energy
fig, ax = plt.subplots(figsize=(6, 5), dpi=120)
kB = 0.00199  # kcal/(mol·K)

F = 4.184 * (-np.log(hist_smooth + 1e-12) / (kB * 298.0))  # kJ/mol
F -= np.min(F)
levels = np.linspace(0, 80, 25)
linelevels = np.linspace(0, 80, 25)

cf = ax.contourf(F, levels=levels, cmap='plasma', extent=extent, origin='lower')
ax.contour(F, levels=linelevels, linewidths=1.0, colors='black',
           extent=extent, origin='lower')
fig.colorbar(cf, ax=ax, label="Free Energy [kJ/mol]")

ax.set_xlabel("ϕ (radians)")
ax.set_ylabel("ψ (radians)")
ax.set_title("ϕ–ψ Ramachandran Plot")
ax.set_aspect('equal')
plt.tight_layout()
plt.savefig('FreeEnergy')
plt.close(fig)

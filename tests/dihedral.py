import torch
import numpy as np
import math
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt

def compute_dihedrals(
    positions: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    """
    Compute signed dihedral angles from 4-atom indices and optionally shift them.

    Parameters
    ----------
    positions : torch.Tensor
        Tensor of shape (T, B, N, 3), atomic positions over time and batch.
    indices : torch.Tensor
        LongTensor of shape (M, 4), indices of atoms forming each dihedral.
    Returns
    -------
    phi : torch.Tensor
        Dihedral angles (T, B, M) in radians, in range [-π, π] + phase
    """
    # Gather the 4 positions per dihedral
    vecs = positions[:, :, indices]  # (T, B, M, 4, 3)
    r1, r2, r3, r4 = vecs[..., 0, :], vecs[..., 1, :], vecs[..., 2, :], vecs[..., 3, :]

    # Bond vectors
    b1 = r2 - r1
    b2 = r3 - r2
    b3 = r4 - r3

    # Normal vectors to the planes
    n1 = torch.nn.functional.normalize(torch.cross(b1, b2, dim=-1), dim=-1)
    n2 = torch.nn.functional.normalize(torch.cross(b2, b3, dim=-1), dim=-1)
    b2n = torch.nn.functional.normalize(b2, dim=-1)

    # Compute cosine and sine of the dihedral
    cos_phi = (n1 * n2).sum(dim=-1).clamp(-1.0, 1.0)
    sin_phi = (torch.cross(n1, n2, dim=-1) * b2n).sum(dim=-1)

    # Return signed angle with optional phase shift
    return torch.atan2(sin_phi, cos_phi)

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
        except:
            break

        i += 2  # Skip comment line

        coords = []
        frame_elements = []
        for j in range(n_atoms):
            parts = lines[i].split()
            frame_elements.append(parts[0])
            coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
            i += 1

        frames.append(np.array(coords))
        if not elements:
            elements = frame_elements

    # Convert to numpy array: (T, N, 3)
    positions = np.array(frames)
    return positions, elements


positions = torch.tensor(read_xyz_file('output.xyz')[0]).unsqueeze(dim=0)


bins = 300
extent = [-math.pi, math.pi, -math.pi, math.pi]
range_ = [extent[:2], extent[2:]]

indices = torch.tensor([[4, 6, 8, 14], [6, 8, 14, 16]])

# Compute and smooth histogram
dihedrals = compute_dihedrals(positions, indices)  # (T, M, 2)
data = dihedrals.reshape(-1, 2).cpu().numpy()
hist, _, _ = np.histogram2d(data[:, 0], data[:, 1], bins=bins, range=range_, density=True)
hist_smooth = gaussian_filter(hist, sigma=3.0)

# Create clean line contour plot
fig, ax = plt.subplots(figsize=(6, 5), dpi=120)
contours = ax.contour(
    hist_smooth, levels=50, linewidths=0.6, cmap='viridis',
    extent=extent, origin='lower', antialiased=True
)
cb = fig.colorbar(contours, ax=ax, label="Probability Density")

ax.set_xlabel("ϕ (radians)")
ax.set_ylabel("ψ (radians)")
ax.set_title("ϕ–ψ Ramachandran Type Plot")
ax.set_aspect('equal')
plt.tight_layout()
plt.savefig('Probability')
plt.close()

ig, ax = plt.subplots(figsize=(6, 5), dpi=120)

kB = 0.00199

# Compute the free energy
F = 4.184 * (-np.log(hist_smooth+1e-10) / (kB * 298))
F -= np.min(F)
levels = np.linspace(0, 80, 25)
linelevels = np.linspace(0, 80, 25)

# Filled contours
cf = ax.contourf(F, levels=levels, cmap='plasma', extent=extent, origin='lower')

# Optional: add contour lines on top
ax.contour(F, levels=linelevels, linewidths=1.5, colors='black', extent=extent, origin='lower')

# Colorbar
cb = fig.colorbar(cf, ax=ax, label="Free Energy [Kj/mol]")

# Labels and formatting
ax.set_xlabel("ϕ (radians)")
ax.set_ylabel("ψ (radians)")
ax.set_title("ϕ–ψ Ramachandran Plot")
ax.set_aspect('equal')
plt.tight_layout()
plt.savefig('FreeEnergy')
plt.close()
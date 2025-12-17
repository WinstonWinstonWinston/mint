import torch
import numpy as np
import math
from scipy.ndimage import gaussian_filter
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt
from tqdm.auto import tqdm


# ---------------------------------------------------------
# Device
# ---------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------
# I/O and utilities
# ---------------------------------------------------------

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


def element_ids(elements):
    """
    Map element symbols to integer IDs for masking.
    """
    unique = sorted(set(elements))
    mapping = {sym: i for i, sym in enumerate(unique)}
    return np.array([mapping[s] for s in elements], dtype=np.int64)


# ---------------------------------------------------------
# Batched Kabsch + Hungarian for one generated frame
# ---------------------------------------------------------

def batched_best_perm_for_frame(
    frame_coords: torch.Tensor,          # (N, 3), on device
    ref_subset_coords: torch.Tensor,     # (K, N, 3), on device
    elem_id_ref: torch.Tensor,           # (N,) long, on device
    elem_id_gen: torch.Tensor,           # (N,) long, on device
    large_penalty: float = 1e6,
):
    """
    For a single generated frame:

      - Align it to each reference in a subset (batched Kabsch on GPU).
      - Build K cost matrices (GPU).
      - Run Hungarian on CPU for each reference.
      - Return permutation with smallest cost.

    Permutation semantics: if perm[i] = j,
    then new_frame[i] = old_frame[j].
    """
    K, N, _ = ref_subset_coords.shape

    # Center coordinates
    frame_centered = frame_coords - frame_coords.mean(dim=0, keepdim=True)      # (N,3)
    refs_centered = ref_subset_coords - ref_subset_coords.mean(dim=1, keepdim=True)  # (K,N,3)

    # Cross-covariance H[k] = frame_centered^T @ refs_centered[k]
    H = torch.einsum("ni,knj->kij", frame_centered, refs_centered)  # (K,3,3)

    # SVD per reference
    U, S, Vh = torch.linalg.svd(H)  # H = U diag(S) Vh

    V = Vh.transpose(-2, -1)        # (K,3,3)
    UT = U.transpose(-2, -1)        # (K,3,3)

    # Provisional rotation
    R_ = V @ UT                     # (K,3,3)

    # Reflection correction
    detR = torch.det(R_)            # (K,)
    D = torch.eye(3, dtype=H.dtype, device=H.device).unsqueeze(0).repeat(K, 1, 1)
    D[:, 2, 2] = torch.where(detR < 0, -1.0, 1.0)
    R = V @ D @ UT                  # (K,3,3)

    # Align frame to each reference: P_aligned[k] = frame_centered @ R[k]
    P_exp = frame_centered.unsqueeze(0).expand(K, -1, -1)  # (K,N,3)
    P_aligned = torch.matmul(P_exp, R)                     # (K,N,3)
    Q = refs_centered                                      # (K,N,3)

    # Pairwise squared distances cost[k, i, j] =
    #  || Q[k,i] - P_aligned[k,j] ||^2
    diff = P_aligned[:, None, :, :] - Q[:, :, None, :]     # (K,N,N,3)
    cost = (diff ** 2).sum(dim=-1)                         # (K,N,N)

    # Element-type mask (same for all refs)
    mask = (elem_id_ref[:, None] == elem_id_gen[None, :])  # (N,N)
    mask = mask.to(device=cost.device)
    cost = cost + (~mask).unsqueeze(0) * large_penalty     # (K,N,N)

    # Move cost to CPU for Hungarian
    cost_np = cost.detach().cpu().numpy()

    best_perm = None
    best_cost = None

    for k in range(K):
        Ck = cost_np[k]
        row_ind, col_ind = linear_sum_assignment(Ck)
        total = Ck[row_ind, col_ind].sum()

        if (best_cost is None) or (total < best_cost):
            best_cost = total
            perm = np.zeros(N, dtype=np.int64)
            perm[row_ind] = col_ind
            best_perm = perm

    return best_perm, best_cost

# ---------------------------------------------------------
# Dihedral calculation
# ---------------------------------------------------------

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


# ---------------------------------------------------------
# Load datasets
# ---------------------------------------------------------

# Reference dataset: canonical indexing and atom types
ref_positions_np, ref_elements = read_xyz_file('dataset.xyz')  # (T_ref, N, 3)

# Generated dataset: possibly scrambled per frame
gen_positions_np, gen_elements = read_xyz_file('output.xyz')   # (T_gen, N, 3)

assert ref_positions_np.shape[1] == gen_positions_np.shape[1], "Atom counts differ."
assert sorted(ref_elements) == sorted(gen_elements), "Element sets differ."

T_ref, N, _ = ref_positions_np.shape
T_gen = gen_positions_np.shape[0]

# Torch tensors on device
ref_positions = torch.tensor(ref_positions_np, dtype=torch.float64, device=device)  # (T_ref, N, 3)
gen_positions = torch.tensor(gen_positions_np, dtype=torch.float64, device=device)  # (T_gen, N, 3)

# Element-type IDs
elem_id_ref_np = element_ids(ref_elements)
elem_id_gen_np = element_ids(gen_elements)
elem_id_ref = torch.tensor(elem_id_ref_np, dtype=torch.long, device=device)
elem_id_gen = torch.tensor(elem_id_gen_np, dtype=torch.long, device=device)

# ---------------------------------------------------------
# Random subset of reference frames
# ---------------------------------------------------------
subset_size = min(T_ref, 1000)  # tune as desired
rng = np.random.default_rng()
subset_indices_np = rng.choice(T_ref, size=subset_size, replace=False)
subset_indices = torch.tensor(subset_indices_np, dtype=torch.long, device=device)
ref_subset = ref_positions[subset_indices]  # (K, N, 3)

# ---------------------------------------------------------
# Per-frame matching with progress bar
# ---------------------------------------------------------
gen_positions_sorted_np = np.empty_like(gen_positions_np)

# for t in tqdm(range(T_gen), desc="Matching generated frames", unit="frame"):
#     frame_coords = gen_positions[t]  # (N,3) on device
#     best_perm, best_cost = batched_best_perm_for_frame(
#         frame_coords=frame_coords,
#         ref_subset_coords=ref_subset,
#         elem_id_ref=elem_id_ref,
#         elem_id_gen=elem_id_gen,
#         large_penalty=1e6,
#     )
#     gen_positions_sorted_np[t] = gen_positions_np[t, best_perm, :]
#     assert torch.all(elem_id_gen[best_perm] == elem_id_gen)
# gen_positions_np = gen_positions_sorted_np

# Back to Torch for dihedrals
positions = torch.tensor(gen_positions_np, dtype=torch.float32).unsqueeze(dim=1).to(device)  # (T_gen,1,N,3)
dataset_positions = torch.tensor(ref_positions_np, dtype=torch.float32).unsqueeze(dim=1).to(device)  # (T_ref,1,N,3)

# ---------------------------------------------------------
# Dihedral indices (defined in dataset.xyz indexing)
# ---------------------------------------------------------
indices = torch.tensor([[6, 8, 14, 16],
                        [4, 6, 8, 14]], dtype=torch.long, device=device)


# ---------------------------------------------------------
# Ramachandran + free energy
# ---------------------------------------------------------
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

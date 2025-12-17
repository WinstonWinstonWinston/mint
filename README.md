# Molecular Interpolant using Neural Transporters (MINT)

![Overview Diagram](MINTDesign.png)

## Environment setup (UMN/MSI)

This project was run on UMN MSI GPU nodes using an interactive SLURM allocation, the UMN module stack (GCC/OpenMPI/CUDA), and a micromamba environment. The provided `setup_umn.sh` script reproduces the *working* environment `mintenv` used for the paper experiments by installing pinned versions of the CUDA 11.8 PyTorch stack, PyG (torch-geometric + compiled extensions), and supporting scientific packages.

**Important note on reproducibility vs. minimal dependencies:**
- The environment created by `setup_umn.sh` is intentionally a "known-good" snapshot that matches a working UMN/MSI setup. It may include more packages than are strictly required by the current codebase. This is expected: the goal is to replicate the results reliably, not to minimize the dependency set.
- A pipreqs-generated `requirements.txt` is included for reference, but pipreqs can miss conditional/dynamic imports and does not capture the GPU-wheel index URLs needed for torch/PyG. For exact replication on UMN/MSI, use `setup_umn.sh`.

### 1) Install micromamba (one-time)

Run:
```bash
"${SHELL}" <(curl -L micro.mamba.pm/install.sh)
```

Verify:
```bash
micromamba --version
```

If your shell does not pick it up immediately, restart the shell.

### 2) Request a GPU (interactive)

Choose one of the following, depending on what is available/appropriate:

- **A40 (basic)**
```bash
srun -N1 --ntasks=1 --gres=gpu:a40:1 -p interactive-gpu -t 04:00:00 --pty bash
```

- **A40 (more CPU tasks; useful for heavier dataloading)**
```bash
srun -N1 --ntasks-per-node=8 --cpus-per-task=1 --gres=gpu:a40:1 -p interactive-gpu -t 08:00:00 --pty bash
```

- **V100 (example request)**
```bash
srun -N 1 -t 08:00:00 --ntasks-per-node=1 -p v100 --mem-per-cpu=20gb --gres=gpu:1 --pty bash
```

### 3) Load the compiler/MPI/CUDA modules

In the allocated shell:
```bash
conda deactivate
module load gcc/8.2.0
module load ompi/3.1.6/gnu-8.2.0
module load cuda/11.8.0-gcc-7.2.0-xqzqlf2
```

### 4) Create/install the pinned Python environment (one-time)

From the repo root (the directory containing `setup_umn.sh`):
```bash
chmod +x setup_umn.sh
./setup_umn.sh
```

Use the package versions pinned in `setup_umn.sh`. Beware that things may not run if you modify it too much.

### 5) Activate the environment (every session)

After you request a GPU and load modules:
```bash
micromamba activate mintenv
```

Recommended sanity checks:

- Confirm a GPU is visible:
```bash
nvidia-smi
```

- Confirm PyTorch sees CUDA:
```bash
python -c "import torch; print(torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

### 6) `requirements.txt`

A `pip freeze > requirements.txt` generated `requirements.txt` is included for reference. We recommend using `setup_umn.sh` instead.

## Obtaining a Model Checkpoint


## Reproducing Results

### Generate

Before running, you must obtain a trained model checkpoint and update the base filename in `generate.py` to point to it. The checkpoint needs to be stored within `mint/EEProjectResults`. You must modify the **base directory** used by the script/config so the paths resolve correctly on your system. This is located within `generate.py` in a string variable named `base`.

To obtain samples, `cd` into `mint/EEProjectResults` and run:
```bash
python generate.py
```

This script loads the model and integrates the stochastic interpolant forward in time to generate samples from the trained distribution. Outputs are written to `mint/EEProjectResults/output.xyz`. It will take 40-45 minutes roughly on an A40. 

You can view the resulting `.xyz` file with Ovito: https://www.ovito.org/.

To switch between SDE and ODE sampling, modify the relevant `generate_cfg` settings used by `generate.py` (SDE vs. ODE mode). Other such hyper parameters may be modified by inspecting the relevant configs (epsilon function, number of samples, return trajectory, etc.)

### Dihedral

This must be run **after** `generate` (i.e., after `output.xyz` has been produced). Running `dihedral` will create two plots in the output directory:

- `FreeEnergy.png` relative energy of folding as a function of the phi psi angle.
- `Probability.png` estimate probability of folding as a function of the phi psi angle.

### Train

Before running, you must modify the **base directory** used by the script/config so the paths resolve correctly on your system.  This is located within `train.py` in a string variable named `base`. Before running be sure you are signed into wandb within that console.

Training is launched from the same directory as `generate.py`. `cd` into `mint/EEProjectResults` and run:
```bash
python train.py
```
You can run this inline (interactive GPU session) or submit it using the provided SLURM script (which must be edited to match your resource/account/partition preferences). We reccomend the slurm method. It will take roughly 8 hours for 100 epochs of the default dataset sizes. The provided checkpoint ran for 10 hours.

Relevant model hyperparameters can be changed by inspecting the `EquivariantMINTModule` object and its associated configuration (i.e., the config fields that instantiate/parameterize the module).

During training, outputs are logged and saved to:
- Checkpoints: `base+"EEProjectResults/logs/hydra/ckpt"`
- Weights & Biases (wandb): `base+"EEProjectResults/logs/wandb"`

## Expected runtimes

Provide estimates for replication:
- Environment creation (`./setup_umn.sh`): 5 minutes
- Sampling for the paper results: 40 minutes
- Training from scratch: 10 hours

## Troubleshooting (common MSI issues)

- If you see `Killed` with no Python traceback, it is often an out-of-memory kill. Reduce memory usage (batch size, dataloader workers, caching) as needed.

- If CUDA is unavailable inside Python, re-check:
  1. You requested a GPU node
  2. The CUDA module is loaded
  3. You activated the correct micromamba environment
  4. The environment's torch (PyG stack) corresponds with the installed versions

from abc import abstractmethod
from torch import Tensor
import os
import pickle
import gzip
import zstandard as zstd
import torch
from torch.utils.data import Dataset
from omegaconf import DictConfig
import copy

class MINTDataset(Dataset):
    """ Abstract dataset class for mint
    ----------
    _dataset_cfg :  DictConfig
        stores all the relevant hydra yaml parameters for the datset configuration
    processed_data : list
        list of processed data, each element of the list corresponds to the result of _preprocess_one_*
    """
    def __init__(self, 
                 data_dir: str, 
                 data_proc_fname: str, 
                 data_proc_ext: str,
                 data_raw_fname: str, 
                 data_raw_ext: str,
                 split: str,
                 total_frames_train: int,
                 total_frames_test: int,
                 total_frames_valid: int,
                 lag: DictConfig,
                 normalize: DictConfig,
                 node_features: DictConfig,
                 augement_rotations: bool):
        
        self.split = split

        self.total_frames_train = total_frames_train
        self.total_frames_test = total_frames_test
        self.total_frames_valid = total_frames_valid

        self.lag = lag
        self.normalize = normalize
        self.node_features = node_features
        
        self.augment_rotations = augement_rotations

        # create path strings
        # dir + name + split + ext
        processed_path = data_dir+"/"+data_proc_fname+"_"+split+data_proc_ext
        raw_path = data_dir+"/"+data_raw_fname+"_"+split+data_raw_ext

        # check if processed data exists
        if os.path.exists(processed_path):
            pass
        else:
            # if not use preprocess from child 
            print(f"INFO:: No processed data found at {processed_path}... preprocessing data")
            self.processed_data = self._preprocess(raw_path, processed_path)

        # # processed data exists now, save it.
        # print(f"INFO:: Loading processed data from {processed_path}")
        # # with gzip.open(processed_path, 'rb') as f:
        # #     self.processed_data = pickle.load(f) # type: ignore

        # with open(processed_path, "rb") as raw:
        #     with zstd.ZstdDecompressor().stream_reader(raw) as f:
        #         self.processed_data = pickle.load(f)

# --------------------------------- preprocessing methods ---------------------------------
#       The methods in this section are built to preprocess a trajectory dataset.
#       Preprocessing should only occur once after cloning the repo.

    def _preprocess_traj_equilibrium(self, traj) -> tuple[Tensor, Tensor, Tensor]:
        """
        Expects traj to be a time-major axis tensor [T,N,3]. Preprocesses the trajectory into subsampled dataset.
        
        note: trajectory must be longer than lag.max

        D corresponds to the frames specified by total_frames_*

        Returns: a tuple of trajectory related objects
            - frames : [D, N, 3] (normalized based off config flags)
            - mean : [1]
            - std : [1]
        """

        # grab unique frames from dataset
        n_frames = getattr(self, f"total_frames_{self.split}")

        assert n_frames < len(traj)
        
        # shuffle the frames and grab a random subset
        frame_idx = torch.randperm(len(traj))[:n_frames]
        frames = torch.tensor(traj.center_coordinates().xyz[frame_idx])

        # compute normalization
        mean = frames.mean()
        std = frames.std()
        normalized_frames = (frames - mean)/std 

        if self.normalize.bool:
            return normalized_frames,mean,std

        return frames,mean,std
    
    def _preprocess_traj_nonequilibrium(self, traj) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """
        Expects traj to be a time-major axis tensor [T,N,3]. Preprocesses the trajectory into a time lagged dataset.
        
        lag.type in {stochastic, fixed} 
        lag.dist in {uniform, fixdisexped}

        note: trajectory must be longer than lag.max

        D corresponds to the frames specified by total_frames_*

        Returns: a tuple of trajectory related objects
            - frames_start : [D, N, 3] (normalized based off config flags)
            - frames_end : [D, N, 3] (normalized based off config flags)
            - mean : [self._dataset_cfg.lag.max, N, 3] or [1] based on time dependent flag 
            - std : [self._dataset_cfg.lag.max, N, 3] or [1] based on time dependent flag 
            - delta_idx : [D]

        """
        assert len(traj) > self.lag.max

        # grab unique frames from dataset
        n_frames = getattr(self, f"total_frames_{self.split}")

        start_frame_idx = torch.randperm(len(traj) - self.lag.max)[:n_frames]

        if self.lag.type == "stochastic":
            
            if self.lag.dist == "uniform":
                delta_idx = torch.randint(low=self.lag.min,high=self.lag.max,size=(n_frames,))
            if self.lag.dist == "disexp":
                delta_idx = torch.exp(torch.rand((n_frames,), dtype=torch.float32) * torch.log(torch.as_tensor(self.lag.max, dtype=torch.float32))).floor().to(torch.int64)
            
            else:
                raise NotImplementedError("Distribution type requested not implemented.")
            
        elif self.lag.type == "fixed":
            delta_idx =  self.lag.max*torch.ones_like(start_frame_idx)
        
        else:
            raise NotImplementedError("Lag type requested not implemented.")
        
        end_frame_idx = start_frame_idx + delta_idx
        
        frames_start = torch.tensor(traj.centered().xyz[start_frame_idx])
        frames_end = torch.tensor(traj.centered().xyz[end_frame_idx])
        frames = torch.cat([frames_start, frames_end], dim=0)

        std = frames.std()
        mean = torch.mean(frames)
        normalized_frames_start = (frames_start - mean)/std
        normalized_frames_end = (frames_end - mean)/std

        if self.normalize.bool:
            if self.normalize.t_dependent:
                MU, STD = self.lag_whitening_stats(traj, self.lag.max)

                normalized_frames_end = (frames_end - MU[delta_idx])/ STD[delta_idx]
                return normalized_frames_start, normalized_frames_end, MU, STD, delta_idx

            else:
                return normalized_frames_start, normalized_frames_end, mean, std, delta_idx
            
        return frames_start, frames_end, mean, std, delta_idx

    def lag_whitening_stats(self,x: torch.Tensor, t_max: int, unbiased: bool = True, eps: float = 1e-8):
        """
        x      : [T, *S] time-major tensor
        t_max  : largest lag to compute (clipped to T-1)
        unbiased: use sample std if True (Torch's default behavior)
        eps    : small additive to std for numerical safety (e.g., 1e-8)

        Returns:
        MU  : [t_max, *S]  where MU[t-1]  = mean over i of (x_{i+t} - x_i) at lag t
        STD : [t_max, *S]  where STD[t-1] = std  over i of (x_{i+t} - x_i) at lag t
        """
        T, *S = x.shape
        if T < 2:
            raise ValueError("x needs at least two time steps")
        t_max = max(1, min(int(t_max), T - 1))

        MU  = torch.empty((t_max, *S), dtype=x.dtype, device=x.device)
        STD = torch.empty((t_max, *S), dtype=x.dtype, device=x.device)

        for t in range(1, t_max + 1):
            delta = x[t:] - x[:-t]           # [T - t, *S]
            MU[t-1]  = delta.mean(dim=0)
            STD[t-1] = delta.std(dim=0, unbiased=unbiased) + eps

        return MU, STD
    
    @abstractmethod
    def _preprocess_node_features(self, **kwargs) -> dict:
        """
        Returns a dict with key value mapping of node features. Expects the values to be 
        tensors with major axis corresponding to number of atoms, like [N,...]. 
        """
        raise NotImplementedError
    
    def _preprocess_one_equilibrium(self, frame: Tensor, node_feats: dict) -> dict:
        """
        Combines the node features and frame to one object. 
        """
        batch = copy.deepcopy(node_feats)
        batch["x"] = frame
        return batch
    
    def _preprocess_one_nonequilibrium(self, frame_0: Tensor, frame_t: Tensor, t: Tensor, node_feats: dict) -> dict:
        """
        Combines the node features and frame to one object. 
        """
        batch = copy.deepcopy(node_feats)
        batch["ic"] = frame_0
        batch["x"] = frame_t
        batch["t"] = t
        return batch

    @abstractmethod
    def _preprocess(self, raw_path, processed_path):
        """
        Preprocess and save results into a compressed pickle file.
        Args:
            raw_path: Path to the prmtop file containing ADP trajectory.
            processed_path: Path to save the processed data (.pkl.gz).
        """
        raise NotImplementedError

# --------------------------------- dataset methods -----------------------------------
#       This class is meant to be a torch.utils.data Dataset type. These need lens and
#       get item methods. Other dataset like methods will be inhereted.

    def __len__(self):
        return len(self.processed_data)

    @abstractmethod
    def __getitem__(self, idx):
        """
        Returns: torch_geometric Data object with keys corresponding to a batch
        """
        raise NotImplementedError
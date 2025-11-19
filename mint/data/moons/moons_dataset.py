import torch
from sklearn.datasets import make_moons
from torch_geometric.data import Data
from torch import Tensor
from e3nn.o3 import Irreps
from torch.utils.data import Dataset

class MOONDataset(Dataset):
    def __init__(self,
                 split: str,
                 total_frames_train: int,
                 total_frames_test: int,
                 total_frames_valid: int):

        super().__init__()

        self.split = split
        self.total_frames_train = total_frames_train
        self.total_frames_test = total_frames_test
        self.total_frames_valid = total_frames_valid

        self.processed_data = self._preprocess()
        
        self.meta_keys = ["feature_keys","x_irrep","label_x_irrep","label_y_irrep","label_z_irrep"]
       
    def _preprocess(self):

        n_train = getattr(self, "total_frames_train")
        n_valid = getattr(self, "total_frames_valid")
        n_test = getattr(self, "total_frames_test")


        moon_noise = 0.1
        moon_scale = 1.3831
        moon_mean = 0.37491956297921425
        if self.split == 'train':
            N = n_train
        elif self.split == 'valid':
            N = n_valid
        elif self.split == 'test':
            N = n_test
        
        X_x, y_x = make_moons(N, noise=moon_noise, random_state=0)
        X_y, y_y = make_moons(N, noise=moon_noise, random_state=1)
        X_z, y_z = make_moons(N, noise=moon_noise, random_state=2)

        X_x = moon_scale*Tensor(X_x - moon_mean)
        X_y = moon_scale*Tensor(X_y - moon_mean)
        X_z = moon_scale*Tensor(X_z - moon_mean)

        y_x = Tensor(y_x)
        y_y = Tensor(y_y)
        y_z = Tensor(y_z)

        positions = torch.stack([X_x, X_y, X_z], dim=-1)

        data = [{   "x": positions[i],  # Tensor of shape [2, 3]
                    "label_x": y_x[i],  # scalar Tensor
                    "label_y": y_y[i],
                    "label_z": y_z[i],
                } for i in range(N)]

        return data   
        
    def __getitem__(self, idx):

        data = self.processed_data[idx]
    
        return Data(
                x=data['x'],
                label_x=data['label_x'],
                label_y=data['label_x'],
                label_z=data['label_x'],
            
                feature_keys = {"x","label_x","label_y","label_z"},

                x_irrep=Irreps("0e"),
                label_x_irrep=Irreps("0e"),
                label_y_irrep=Irreps("0e"),
                label_z_irrep=Irreps("0e"),
            )
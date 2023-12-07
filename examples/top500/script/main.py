import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

import sys
sys.path.append("../../")
from manifm.manifolds import FlatTorus

class Top500(Dataset):
    manifold = FlatTorus()
    dim = 2

    def __init__(self, root="./data", amino="General"):
        data = pd.read_csv(
            f"{root}/aggregated_angles.tsv",
            delimiter="\t",
            names=["source", "phi", "psi", "amino"],
        )

        amino_types = ["General", "Glycine", "Proline", "Pre-Pro"]
        assert amino in amino_types, f"amino type {amino} not implemented"

        data = data[data["amino"] == amino][["phi", "psi"]].values.astype("float32")
        self.data = torch.tensor(data % 360 * np.pi / 180)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    
train_set = Top500(amino="General")
train_loader = DataLoader(
    train_set, 128, shuffle=True, pin_memory=True, drop_last=True
)


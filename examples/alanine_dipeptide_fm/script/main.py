from omegaconf import OmegaConf
import mdtraj
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import sys

sys.path.append("/home/xqding/my_projects_on_github/MMFlow")
from manifm.model_pl import ManifoldFMLitModule
import pytorch_lightning as pl

topology = mdtraj.load_prmtop("./structure/alanine_dipeptide.prmtop")
traj = mdtraj.load_dcd(
    "./output/umbrella_sampling/traj_all.dcd", top=topology, stride=1
)

thetas = mdtraj.compute_dihedrals(traj, [[6, 8, 14, 16], [14, 8, 6, 4]])
thetas = thetas % 2 * np.pi
thetas = torch.from_numpy(thetas)

dataset = TensorDataset(thetas)
N = len(dataset)
N_val = N_test = N // 10
N_train = N - N_val - N_test
data_seed = 0
train_set, val_set, test_set = torch.utils.data.random_split(
    dataset,
    [N_train, N_val, N_test],
    generator=torch.Generator().manual_seed(data_seed),
)

cfg = OmegaConf.load("./conf.yaml")


def collate_fn(batch):
    return torch.stack([item[0] for item in batch])


train_loader = DataLoader(
    train_set,
    cfg.optim.batch_size,
    shuffle=True,
    pin_memory=True,
    drop_last=True,
    collate_fn=collate_fn,
)
val_loader = DataLoader(
    val_set,
    cfg.optim.val_batch_size,
    shuffle=False,
    pin_memory=True,
    collate_fn=collate_fn,
)
test_loader = DataLoader(
    test_set,
    cfg.optim.val_batch_size,
    shuffle=False,
    pin_memory=True,
    collate_fn=collate_fn,
)


model = ManifoldFMLitModule(cfg)
print(model)
trainer = pl.Trainer(
    max_steps=cfg.optim.num_iterations,
    val_check_interval=cfg.val_every,
    check_val_every_n_epoch=None,
    precision=cfg.get("precision", 32),
    gradient_clip_val=cfg.optim.grad_clip,
    num_sanity_val_steps=0,
)

trainer.fit(model, train_loader, val_loader)

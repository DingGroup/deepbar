import mdtraj
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import pickle
from sys import exit

top = mdtraj.load_prmtop("./structure/complex.prmtop")
traj = mdtraj.load_dcd("./output/trajectory.dcd", top=top, stride=1)

with open("./output/flexible_atom_indices.pkl", "rb") as f:
    flexible_atom_indices = pickle.load(f)

flexible_atom_indices = np.array(list(flexible_atom_indices))

x = traj.xyz[:, flexible_atom_indices, :]
mu = np.mean(x, axis=0)
std = np.std(x, axis=0) * 2
with open("./output/input_scale.pkl", "wb") as f:
    pickle.dump({"mu": mu, "std": std}, f)

x = (x - mu) / std
x = x.astype(np.float32)

dataset = TensorDataset(torch.from_numpy(x))
dataloader = DataLoader(dataset, batch_size=1024, shuffle=True)


class Denoiser(torch.nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, num_layers, sigma_data):
        super(Denoiser, self).__init__()

        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers

        ## the extra input dimension is for the sigma, the noise level
        self.embed = nn.Linear(input_dim + 1, embed_dim, bias=False)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=128,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )
        self.output_linear = nn.Linear(embed_dim, input_dim, bias=False)

        self.sigma_data = sigma_data

    def forward(self, t, x):
        while t.dim() < x.dim():
            t = t.unsqueeze(-1)
        sigma = t
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_in = 1.0 / torch.sqrt(sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / torch.sqrt(sigma**2 + self.sigma_data**2)
        c_noise = 1 / 4 * torch.log(sigma)

        out = c_skip * x

        x = c_in * x
        c_noise = torch.ones(list(x.shape)[0:-1] + [1], device = x.device) * c_noise
        x = torch.cat([x, c_noise], -1)
        x = self.embed(x)
        x = self.transformer_encoder(x)
        F = self.output_linear(x)

        out = out + c_out * F

        return out


net = Denoiser(
    input_dim=x.shape[-1], embed_dim=32, num_heads=8, num_layers=1, sigma_data=0.5
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = net.to(device)

optimizer = optim.Adam(net.parameters(), lr=1e-3)

for epoch in range(100):
    for x in dataloader:
        x = x[0]    
        x = x.to(device)
        t = torch.exp(torch.normal(-1.2, 1.2, size=(x.shape[0],1, 1))).to(device)
        noise = torch.randn_like(x) * t
        
        weight = (t**2 + net.sigma_ddata**2)/(t*net.sigma_data)**2
        loss = torch.mean(weight*(net(t, x + noise) - x)**2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"epoch {epoch}, loss {loss.item()}")
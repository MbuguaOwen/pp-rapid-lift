from __future__ import annotations
import torch, torch.nn as nn, torch.nn.functional as F

class TSEncoder(nn.Module):
    def __init__(self, in_ch: int, emb_dim: int=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, 64, 5, padding=2), nn.ReLU(),
            nn.Conv1d(64, 128, 5, padding=2), nn.ReLU(),
            nn.Conv1d(128, emb_dim, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
    def forward(self, x):        # x: (B, L, C)
        x = x.transpose(1,2)     # (B, C, L)
        h = self.net(x).squeeze(-1)
        return F.normalize(h, dim=1)

class ProjectionHead(nn.Module):
    def __init__(self, emb_dim: int=128, proj_dim: int=128):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(emb_dim, proj_dim), nn.ReLU(),
                                 nn.Linear(proj_dim, proj_dim))
    def forward(self, z):
        return F.normalize(self.mlp(z), dim=1)

def nt_xent(z1, z2, temperature=0.2):
    B = z1.size(0)
    z = torch.cat([z1, z2], dim=0)
    sim = torch.matmul(z, z.t()) / temperature
    mask = torch.eye(2*B, dtype=torch.bool, device=z.device)
    sim = sim.masked_fill(mask, -1e9)
    targets = torch.arange(B, device=z.device)
    loss = torch.nn.functional.cross_entropy(sim[:B, B:], targets) + torch.nn.functional.cross_entropy(sim[B:, :B], targets)
    return loss / 2


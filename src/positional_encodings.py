import torch
import torch.nn as nn

class PositionalEncodings(nn.Module):
    """
    > This Applies the Positional Encodings to the 'N' sampled points for all Rays before feeding into the MLP
    > Discussed in the Section-5.1: Positional encoding
    > We apply Positional Encodings because,
        Mapping the inputs to a higher
        dimensional space using high frequency functions before passing them to the MLP
        network enables better fitting of data that contains high frequency variation
    """
    def __init__(self, L: int):
        super().__init__()
        self.L = L
        self.register_buffer('freqs', torch.pow(2.0, torch.arange(self.L)))
        self.out_dim = 3 + 6 * self.L # because of concat (6L) and then attaching the original co-ordinates (+3)

    def forward(self, ray_samples):
        # ray_samples.shape = (B, R, N, 3)
        B, R, N, dir_dim = ray_samples.shape
        original_xyz = ray_samples
        ray_samples = ray_samples.unsqueeze(-2) # (B, R, N, 1, 3) 
        freqs = self.freqs.view(1, 1, 1, self.L, 1) # freq.shape = (1, 1, 1, self.L, 1)
        x = freqs * torch.pi * ray_samples # (B, R, N, L, 3)
        sin_pos = torch.sin(x) # (B, R, N, L, 3)
        cos_pos = torch.cos(x) # (B, R, N, L, 3)
        sin = sin_pos.view(B, R, N, self.L * dir_dim) # or we could also use [sin.flatten(start_dim = -2)]; same goes with cos
        cos = cos_pos.view(B, R, N, self.L * dir_dim)
        embeddings = torch.cat([original_xyz, sin, cos], dim = -1)
        return embeddings
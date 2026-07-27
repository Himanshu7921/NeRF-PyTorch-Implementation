import torch
import torch.nn as nn

class StratifiedSampler(nn.Module):

    """
    > This Implements the Ray Sampler from the Original Paper: NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis
    > Instead of Querying the MLP on all 640,000 Rays we Query the MLP on self.N points sampled uniformly from [t_f, t_n] / N bins

    > Discussed in the Section-4: Volume Rendering with Radiance Fields
    """
    def __init__(self, t_n, t_f, N: int):
        super().__init__()
        self.t_n = t_n
        self.t_f = t_f
        self.N = N
    
    def forward(self, ray):
        direction, origin = ray[0], ray[1]
        batch, n_rays, dir_dim = direction.shape # batch, n_pixels, 3 (x, y, z) direction
        bins = torch.linspace(self.t_n, self.t_f, self.N + 1, device = origin.device)
        lower = bins[:-1]
        upper = bins[1:]

        lower = lower.unsqueeze(0).unsqueeze(0).expand(batch, n_rays, self.N)
        upper = upper.unsqueeze(0).unsqueeze(0).expand(batch, n_rays, self.N)
        u = torch.rand(batch, n_rays, self.N, device = origin.device)
        t_vals = lower + (upper - lower) * u

        # ------------------------------------------------------------------------
        # direction.shape = (B, n_rays, 3)
        # origin.shape = (B, n_rays, 3)
        # t_vals.shape = (B, n_rays, self.N)

        # ------------------------------------------------------------------------
        # origin[:, :, None, :].shape = (B, n_rays, 1, 3)
        # direction[:, :, None, :].shape = (B, n_rays, 1, 3)
        # t_vals[..., :, None].shape = (B, n_rays, self.N, 1)

        # ------------------------------------------------------------------------
        points = (
            origin[:, :, None, :]
            + direction[:, :, None, :] * t_vals[..., :, None]
        )
        return points, t_vals # We need this for volume rendering
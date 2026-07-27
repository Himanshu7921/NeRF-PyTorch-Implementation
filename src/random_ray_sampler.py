import torch
import torch.nn as nn


class RandomRaySampler(nn.Module):
    """
    Randomly samples a subset of rays from the full set of image rays.

    Input:
        directions : (B, R, 3)
        origins    : (B, R, 3)

    Output:
        directions : (B, num_rays, 3)
        origins    : (B, num_rays, 3)
        indices    : (num_rays,)
    """

    def __init__(self, num_rays: int = 1024):
        super().__init__()
        self.num_rays = num_rays

    def forward(self, directions, origins, sample_all_rays = False):
        _, n_rays, _ = directions.shape

        if sample_all_rays:
            idx = torch.arange(
                n_rays,
                device=directions.device,
            )
        else:
            idx = torch.randperm(
                n_rays,
                device=directions.device
            )[:self.num_rays]

        directions = directions[:, idx]
        origins = origins[:, idx]

        return directions, origins, idx
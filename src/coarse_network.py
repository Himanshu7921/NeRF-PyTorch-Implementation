import torch
from mlp import MLP
import torch.nn as nn
from ray_generator import RayGenerator
from random_ray_sampler import RandomRaySampler
from stratified_sampler import StratifiedSampler
from positional_encodings import PositionalEncodings

class CoarseNetwork(nn.Module):
    """
    > This class implements the Coarse NeRF network proposed in
    "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis"
    (ECCV 2020).

    > The implementation follows the methodology described in:
        - Section 3: Neural Radiance Field Scene Representation
        - Section 4: Volume Rendering with Radiance Fields
        - Section 5.1: Positional Encoding

    > Responsibilities:
        - Generate camera rays from input images.
        - Randomly sample a subset of rays during training.
        - Perform stratified sampling to obtain 3D query points along each ray.
        - Apply Fourier positional encoding to sampled 3D locations and viewing directions.
        - Query the coarse NeRF MLP to predict:
            • Volume Density (σ)
            • View-dependent RGB Color

    > Pipeline:
        Input Images
            ↓
        Ray Generation
            ↓
        Random Ray Sampling (training only)
            ↓
        Stratified Sampling
            ↓
        Fourier Positional Encoding
            ↓
        Coarse NeRF MLP
            ├── Volume Density (σ)
            └── View-dependent RGB
            ↓
        Coarse Radiance Field (σ, RGB)

    > Note:
        This module implements only the coarse NeRF stage. The predicted coarse
        radiance field will later be consumed by the volume renderer to estimate
        transmittance weights, construct the importance sampling PDF, and generate
        additional samples for the Fine NeRF network described in Section 5.2
    """
    def __init__(self, t_n: int, t_f: int, n_points: int, num_rays: int, position_L: int, direction_L: int, hidden_dim: int, **kwargs):
        super().__init__()
        self.t_n = t_n
        self.t_f = t_f
        self.n_points = n_points
        self.hidden_dim = hidden_dim

        self.ray_generator = RayGenerator()
        self.sampler = RandomRaySampler(num_rays=num_rays)
        self.stratified_sampler = StratifiedSampler(t_n = self.t_n, t_f = self.t_f, N = self.n_points)
        self.position_pos_enc = PositionalEncodings(L = position_L)
        self.direction_pos_enc = PositionalEncodings(L = direction_L)
        self.mlp = MLP(position_dim = self.position_pos_enc.out_dim, direction_dim = self.direction_pos_enc.out_dim, hidden_dim = self.hidden_dim)

    def forward(self, batch, sample_all_rays = False):
        # Generate Rays
        direction, origin = self.ray_generator(batch)

        # Get Random Rays: Random Ray sampling is a training time optimization discussed in the NeRF paper
        img = batch["target_rgb"]
        # print(
        #     "Inside (CoarseNetwork.forward()) Whole image:",
        #     img.min().item(),
        #     img.max().item(),
        #     img.mean().item()
        # )
        if self.training:
            direction, origin, idx = self.sampler(direction, origin, sample_all_rays = sample_all_rays)

        outputs = self.render_rays(direction, origin)
        outputs["random_ray_idx"] = idx
        return outputs

    def render_rays(self, direction, origin):
        sampled_points, t_vals = self.stratified_sampler([direction, origin])
        sampled_directions = direction[:, :, None, :].expand(-1, -1, sampled_points.shape[2], -1) # <------------------ get the viewing directions

        # Encode both directions and positions
        encoded_sampled_points = self.position_pos_enc(sampled_points) # <-------------- Passing this to MLP
        encoded_sampled_directions = self.direction_pos_enc(sampled_directions) # <-------------- Passing this to MLP

        # pass this encoded positions and directions to the MLP
        sigma, rgb = self.mlp(encoded_sampled_points, encoded_sampled_directions)
        radiance_field = torch.cat([sigma, rgb], dim=-1) # [B, R, N, 4]

        # Get deltas from t_vals (later required for weight calculation)
        # t_vals = [x1, x2, x3, ...., x_n] | deltas = [x2- x1, x3 - x2, x4 - x3, ..., x_n - x_n-1, [??]]
        deltas = t_vals[:, :, 1:] - t_vals[:, :, :-1] # difference b/w each points # [B, R, N-1]
        last_delta = torch.full_like(deltas[:, :, :1], 1e10) # (B, R, 1)
        deltas = torch.cat([deltas, last_delta], dim=2) # (B, R, N)
        return {
            "radiance_field": radiance_field,
            "t_vals": t_vals,
            "deltas": deltas,
            "ray_origins": origin,
            "ray_directions": direction,
            "sigma": sigma,
            "rgb": rgb
        }
import torch
import torch.nn as nn
from positional_encodings import PositionalEncodings
from mlp import MLP

class FineNetwork(nn.Module):
    def __init__(self, position_L: int, direction_L: int, hidden_dim: int, **kwargs):
        super().__init__()

        self.hidden_dim = hidden_dim

        self.position_pos_enc = PositionalEncodings(L = position_L)
        self.direction_pos_enc = PositionalEncodings(L = direction_L)

        self.mlp = MLP(position_dim = self.position_pos_enc.out_dim, direction_dim = self.direction_pos_enc.out_dim, hidden_dim = self.hidden_dim)

    def forward(self, ray_origins, ray_directions, fine_t_vals):
        sampled_points = (
            ray_origins.unsqueeze(2)
            + ray_directions.unsqueeze(2)
            * fine_t_vals.unsqueeze(-1)
        )

        sampled_directions = ray_directions[:, :, None, :].expand(-1, -1, sampled_points.shape[2], -1) # <------------------ get the viewing directions
        
        # Encode both directions and positions
        encoded_sampled_points = self.position_pos_enc(sampled_points) # <-------------- Passing this to MLP
        encoded_sampled_directions = self.direction_pos_enc(sampled_directions) # <-------------- Passing this to MLP

        sigma, rgb = self.mlp(encoded_sampled_points, encoded_sampled_directions)

        # Get deltas from fine_t_vals (later required for weight calculation [Volume Rendering])
        deltas = fine_t_vals[:, :, 1:] - fine_t_vals[:, :, :-1] # difference b/w each points # [B, R, N-1]
        last_delta = torch.full_like(deltas[:, :, :1], 1e10) # (B, R, 1)
        deltas = torch.cat([deltas, last_delta], dim=2) # (B, R, N)
        # print(sampled_points.shape)
        # print(encoded_sampled_points.shape)
        # print(encoded_sampled_directions.shape)
        return {
            "t_vals": fine_t_vals,
            "deltas": deltas,
            "sigma": sigma,
            "rgb": rgb
        }
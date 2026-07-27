import torch
import torch.nn as nn

class VolumeRenderer(nn.Module):
    """
    > This class implements the differentiable Volume Renderer proposed in
    "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis"
    (ECCV 2020).

    > The implementation follows the methodology described in:
        - Section 4: Volume Rendering with Radiance Fields

    > Responsibilities:
        - Compute accumulated transmittance along each ray.
        - Estimate contribution weights for every sampled point.
        - Integrate the predicted radiance field into the final RGB color.
        - Return the normalized weights required for hierarchical importance sampling.

    > Note:
        This module performs differentiable volume rendering and provides both
        the rendered pixel colors and the sampling weights used by the
        ImportanceSampler.
    """
    def __init__(self, eps: float = 1e-10):
        super().__init__()
        self.eps = eps

    def forward(self, sigma, rgb, deltas, t_vals):
        # R = n_rays
        # N = n_points
        
        # ------------------------------------------------------------
        # deltas.shape = (B, R, N) | deltas.unsqueeze(-1).shape = (B, R, N, 1)
        # sigma.shape = (B, R, N, 1)
        # rgb.shape = (B, R, N, 3)
        # ------------------------------------------------------------
        deltas = deltas.unsqueeze(-1)
        sigma_delta = sigma * deltas # sigma_delta.shape = (B, R, N, 1) * (B, R, N, 1) = (B, R, N, 1)
        cumulative_sum = torch.cumsum(sigma_delta, dim = 2) # (B, R, N, 1)
        cumulative_sum = cumulative_sum[:, :, :-1, :] # get all but the last dim; shape = (B, R, N - 1, 1)
        zeros = torch.zeros_like(cumulative_sum[:, :, :1, :]) # (B, R, 1, 1)
        cumulative  = torch.cat([zeros, cumulative_sum], dim  = 2) # (B, R, N, 1)

        # Now get alpha, transmittance and weights
        alpha = 1 - torch.exp(-sigma_delta) # (B, R, N, 1)
        transmittance = torch.exp(-cumulative) # (B, R, N, 1)
        weights = transmittance * alpha # weights.shape = (B, R, N, 1) | (B, 1024, 64, 1) # This weights will then be used for rendering the final RGB and for constructing the PDF for hierarchical importance sampling

        # Normalize this weights for hierarchical importance sampling
        norm_weights = weights / (torch.sum(weights, dim = 2, keepdim = True) + self.eps)
        rendered_rgb = torch.sum(rgb * weights, dim  = 2)
        depth = torch.sum(weights * t_vals.unsqueeze(-1), dim = 2) # optional
        return {
            "rendered_rgb": rendered_rgb,
            "weights": weights,
            "norm_weights": norm_weights,
            "alpha": alpha,
            "transmittance": transmittance,
            "depth": depth
        }
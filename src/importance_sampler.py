import torch
import torch.nn as nn

class ImportanceSampler(nn.Module):
    """
    > This class implements the Hierarchical Importance Sampler proposed in
    "NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis"
    (ECCV 2020).

    > The implementation follows the methodology described in:
        - Section 5.2: Hierarchical Volume Sampling

    > Responsibilities:
        - Construct a probability density function (PDF) from the normalized
        transmittance weights produced by the Coarse Volume Renderer.
        - Perform inverse transform sampling to draw additional samples from
        regions with high accumulated density.
        - Generate a refined set of query points by combining the coarse and
        importance-sampled points.
        - Produce the samples that will be queried by the Fine NeRF network.

    > Pipeline:
        Coarse Radiance Field
            ↓
        Volume Rendering
            ↓
        Normalized Weights (PDF)
            ↓
        Inverse Transform Sampling
            ↓
        Additional Sampled Points
            ↓
        Fine NeRF Network

    > Note:
        This module implements hierarchical importance sampling only. The
        generated fine samples are later combined with the coarse samples and
        passed to the Fine NeRF network for refined radiance field prediction.

    Mathematically, importance sampling is just inverse transform sampling from a discrete probability distribution

    Steps:
        1. Construct the Probability Density Function (PDF) from the normalized weights
        2. Compute the Cumulative Distribution Function (CDF) from the PDF
        3. Draw new samples using inverse transform sampling over the CDF
        4. Linearly interpolate between neighboring t_vals to obtain the new importance-sampled t_vals
    """
    def __init__(self, eps: float = 1e-5, n_importance: int = 128):
        super().__init__()
        self.eps = eps
        self.n_importance = n_importance


    def _perform_linear_interpolation(self, u, t_vals, cdf, indices):
        # r = (u - cdf_low) / (cdf_upper - cdf_low)
        # t = t_low + r(t_upper - t_low)

        # ------------ Shapes ----------------------
        # t_vals.shape = (B, R, N)
        # cdf.shape = (B, R, N)
        # indices.shape = (B, R, self.n_importance)
        # -----------------------------------------

        n_points = t_vals.shape[-1]

        lower = torch.clamp(indices - 1, min = 0)
        upper = torch.clamp(indices, max = n_points - 1)


        cdf_lower = torch.gather(input = cdf, index = lower, dim = -1)
        cdf_upper = torch.gather(input = cdf, index = upper, dim = -1)

        t_lower = torch.gather(input = t_vals, index = lower, dim = -1)
        t_upper = torch.gather(input = t_vals, index = upper, dim = -1)

        r = (u - cdf_lower) / ((cdf_upper - cdf_lower) + self.eps)
        t_new = t_lower + r * (t_upper - t_lower)
        return t_new
    
    def forward(self, norm_weights: torch.Tensor, t_vals: torch.Tensor):
        # norm_weights.shape = (B, R, N, 1)
        # t_vals.shape = (B, R, N)
        pdf = norm_weights[:, :, 1:-1, 0] # reject the 1st and last bin, because of transmittance those bins will dominate # pdf.shape = (B, R, N-2)
        old_t_vals = t_vals
        t_vals = t_vals[:, :, 1:-1] # get the corresponding t_vals

        # normalize the pdf again
        pdf = pdf / (torch.sum(pdf, dim = 2, keepdim = True) + self.eps)
        cdf = torch.cumsum(pdf, dim = 2) # shape = (B, R, N-2)

        B, R, _ = cdf.shape
        u = torch.rand(
            B, R, self.n_importance,
            device=cdf.device,
            dtype=cdf.dtype
        )
        idx = torch.searchsorted(cdf, u)

        t_new = self._perform_linear_interpolation(u, t_vals, cdf, idx)
        return torch.sort(torch.cat([t_new, old_t_vals], dim = 2), dim = 2).values
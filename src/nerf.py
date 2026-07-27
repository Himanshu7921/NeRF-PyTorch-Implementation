import torch
import torch.nn as nn
from coarse_network import CoarseNetwork
from volume_renderer import VolumeRenderer
from importance_sampler import ImportanceSampler
from fine_network import FineNetwork


class NeRF(nn.Module):
    """
    > This class implements the complete Neural Radiance Field (NeRF) pipeline
    proposed in "NeRF: Representing Scenes as Neural Radiance Fields for View
    Synthesis" (ECCV 2020).

    > Responsibilities:
        - Evaluate the Coarse NeRF network to predict the radiance field.
        - Render the coarse predictions using volume rendering.
        - Perform hierarchical importance sampling to generate refined sample
        locations.
        - Pass the refined samples to the Fine NeRF network.

    > Pipeline:
        Input Rays
            ↓
        Coarse NeRF
            ↓
        Volume Rendering
            ↓
        Importance Sampling
            ↓
        Fine NeRF
    """
    def __init__(self, config):
        super().__init__()

        self.coarse_network = CoarseNetwork(
            t_n = config["t_n"],
            t_f = config["t_f"],
            n_points = config["n_points"],
            num_rays = config["num_rays"],
            position_L = config["position_L"],
            direction_L = config["direction_L"],
            hidden_dim = config["hidden_dim"]
        )
        self.volume_renderer = VolumeRenderer(eps = config["volume_renderer_eps"])
        self.importance_sampler = ImportanceSampler(eps = config["importance_sampler_eps"], n_importance = config["n_importance"])
        self.fine_network = FineNetwork(
            position_L = config["position_L"],
            direction_L = config["direction_L"],
            hidden_dim = config["hidden_dim"]
        )

    def forward(self, batch, sample_all_rays = False):
        # Coarse Network
        directions, origins = self.coarse_network.ray_generator(batch)

        if self.training:
            directions, origins, idx = self.coarse_network.sampler(
                directions = directions, origins = origins
            )

        outputs = self.render_rays(directions = directions, origins = origins)
        if self.training:
            outputs["coarse"]["random_ray_idx"] = idx
        return outputs


    def render_rays(self, directions, origins):
        # t = time.time()
        coarse_net = self.coarse_network.render_rays(
            direction = directions,
            origin = origins,
        )
        # print("Coarse:", time.time() - t)

        # t = time.time()
        coarse_render = self.volume_renderer(
            coarse_net["sigma"],
            coarse_net["rgb"],
            coarse_net["deltas"],
            coarse_net["t_vals"],
        )
        # print("Coarse Renderer:", time.time() - t)

        # Importance Sampling
        # t = time.time()
        fine_t_vals = self.importance_sampler(
            coarse_render["norm_weights"],
            coarse_net["t_vals"],
        )
        # print("Sampler:", time.time() - t)

        # Fine Network
        # t = time.time()
        fine_net = self.fine_network(
            ray_origins = origins,
            ray_directions = directions,
            fine_t_vals = fine_t_vals,
        )
        # print("Fine:", time.time() - t)

        # t = time.time()
        fine_render = self.volume_renderer(
            fine_net["sigma"],
            fine_net["rgb"],
            fine_net["deltas"],
            fine_net["t_vals"],
        )
        # print("Fine Renderer:", time.time() - t)
        # print(torch.all(fine_t_vals[:, :, 1:] >= fine_t_vals[:, :, :-1]))

        return {
            "coarse": {
                "rgb": coarse_render["rendered_rgb"],
                "depth": coarse_render["depth"],
                "weights": coarse_render["weights"],
                "alpha": coarse_render["alpha"],
                "transmittance": coarse_render["transmittance"],
                "sigma": coarse_net["sigma"],
            },
            "fine": {
                "rgb": fine_render["rendered_rgb"],
                "depth": fine_render["depth"],
                "weights": fine_render["weights"],
                "sigma": fine_net["sigma"],
                "alpha": fine_render["alpha"],
                "transmittance": fine_render["transmittance"],
            }
        }


    def render_image(self, batch, scale = 0.25):
        H = batch["image"].shape[-2]
        W = batch["image"].shape[-1]

        new_H = int(H * scale)
        new_W = int(W * scale)

        ray_directions, ray_origins = self.coarse_network.ray_generator(
            batch,
            height=new_H,
            width=new_W,
        )
        # print(ray_origins.shape)
        chunk_size = 1024
        rgb = []


        _, n_rays, _ = ray_origins.shape
        for i in range(0, n_rays, chunk_size):

            outputs = self.render_rays(
                origins = ray_origins[:, i:i+chunk_size],
                directions = ray_directions[:, i:i+chunk_size],
            )

            rgb.append(outputs["fine"]["rgb"])

        B = batch["image"].shape[0]
        rgb = torch.cat(rgb, dim=1)
        rgb = rgb.reshape(B, new_H, new_W, 3)
        return rgb
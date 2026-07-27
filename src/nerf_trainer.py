import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import Adam
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
from pathlib import Path

class Trainer:
    """
    Generic Trainer for the original NeRF implementation.

    Responsibilities
    ----------------
    - Forward pass
    - Loss computation
    - Backpropagation
    - Optimizer step
    - Checkpointing
    - Progress visualization
    - Optional Weights & Biases logging
    """

    def __init__(
        self,
        model,
        train_loader,
        config,
        device,
        use_wandb: bool = False,
        resume_from = None
    ):
        self.start_epoch = 1
        self.best_loss = float("inf")


        self.model = model.to(device)
        self.train_loader = train_loader
        self.device = device
        self.config = config

        self.optimizer = Adam(
            self.model.parameters(),
            lr=config["lr"],
            weight_decay=config.get("weight_decay", 0.0),
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=self.config["epochs"], 
            eta_min=self.config.get("lr_final", 5e-5)
        )
        self.use_wandb = use_wandb

        if resume_from is not None:
            self.load_checkpoint(resume_from)
            
        if self.use_wandb:
            import wandb
            wandb.init(
                project=config.get("wandb_project", "NeRF"),
                name=config.get("run_name"),
                config=config,
            )

            self.wandb = wandb
        self.save_dir = Path(config.get("save_dir", "checkpoints"))
        self.save_dir.mkdir(exist_ok=True)

    # Loss
    def compute_loss(self, outputs, target_rgb):
        B, _, H, W = target_rgb.shape

        # Flatten image: (B, 3, H, W) -> (B, H*W, 3)
        target_rgb = (
            target_rgb
            .permute(0, 2, 3, 1)      # (B, H, W, 3)
            .reshape(B, -1, 3)         # (B, H*W, 3)
        )

        # Get sampled ray indices
        ray_idx = outputs["coarse"]["random_ray_idx"]
        if ray_idx.dim() == 1:
            ray_idx = ray_idx.unsqueeze(0)

        _, N_rays = ray_idx.shape
        gather_idx = ray_idx.unsqueeze(-1).expand(B, N_rays, 3)

        # Gather target colors for selected rays -> (B, N_rays, 3)
        target_rgb_sampled = torch.gather(target_rgb, dim=1, index=gather_idx)
        # print(
        #     "Target sampled (inside compute_loss):",
        #     target_rgb_sampled.min().item(),
        #     target_rgb_sampled.max().item(),
        #     target_rgb_sampled.mean().item(),
        # )

        coarse_loss = F.mse_loss(
            outputs["coarse"]["rgb"],
            target_rgb_sampled,
        )

        fine_loss = F.mse_loss(
            outputs["fine"]["rgb"],
            target_rgb_sampled,
        )
        loss = coarse_loss + fine_loss
        return loss, coarse_loss, fine_loss

    # Train one epoch
    def train_one_epoch(self, epoch, sample_all_rays = False):
        self.model.train()
        epoch_loss = 0.0
        progress = tqdm(
            self.train_loader,
            desc=f"Epoch [{epoch}/{self.config['epochs']}]",
            dynamic_ncols=True,
            leave=False,
            colour="cyan",
            bar_format="{l_bar}{bar:20}{r_bar}",
        )

        for batch_idx, batch in enumerate(progress):
            target_rgb = batch["target_rgb"].to(self.device)

            batch = {
                k: v.to(self.device)
                if torch.is_tensor(v)
                else v
                for k, v in batch.items()
            }

            # Forward
            outputs = self.model(batch, sample_all_rays)

            # Loss
            loss, coarse_loss, fine_loss = self.compute_loss(
                outputs,
                target_rgb,
            )

            # Backprop
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Logging
            epoch_loss += loss.item()

            progress.set_postfix(
                loss=f"{loss.item():.6f}",
                coarse=f"{coarse_loss.item():.6f}",
                fine=f"{fine_loss.item():.6f}",
            )

            if self.use_wandb:
                coarse_weights = outputs["coarse"]["weights"]
                coarse_sigma = outputs["coarse"]["sigma"]          # if available
                coarse_rgb = outputs["coarse"]["rgb"]              # predicted colors before rendering
                coarse_weights_sum = coarse_weights.sum(dim = 2)
                coarse_alpha = outputs["coarse"]["alpha"]
                coarse_T = outputs["coarse"]["transmittance"]

                fine_weights = outputs["fine"]["weights"]
                fine_sigma = outputs["fine"]["sigma"]          # if available
                fine_rgb = outputs["fine"]["rgb"]              # predicted colors before rendering
                fine_weights_sum = fine_weights.sum(dim = 2)
                fine_alpha = outputs["fine"]["alpha"]
                fine_T = outputs["fine"]["transmittance"]

                self.wandb.log({
                        "train/loss": loss.item(),
                        "train/coarse_loss": coarse_loss.item(),
                        "train/fine_loss": fine_loss.item(),
                        
                        "coarse/weights/min": coarse_weights.min().item(),
                        "coarse/weights/max": coarse_weights.max().item(),
                        "coarse/weights/mean": coarse_weights.mean().item(),
                        "coarse/weights/std": coarse_weights.std().item(),
                        "coarse/weights_sum/mean": coarse_weights_sum.mean().item(),
                        "coarse/weights_sum/min": coarse_weights_sum.min().item(),
                        "coarse/weights_sum/max": coarse_weights_sum.max().item(),

                        "coarse/sigma/mean": coarse_sigma.mean().item(),
                        "coarse/sigma/max": coarse_sigma.max().item(),

                        "coarse/rgb/mean": coarse_rgb.mean().item(),
                        "coarse/rgb/max": coarse_rgb.max().item(),

                        # Alpha and Transmittance
                        "coarse/alpha/mean": coarse_alpha.mean().item(),
                        "coarse/alpha/max": coarse_alpha.max().item(),
                        "coarse/alpha/min": coarse_alpha.min().item(),

                        "coarse/transmittance/mean": coarse_T.mean().item(),
                        "coarse/transmittance/max": coarse_T.max().item(),
                        "coarse/transmittance/min": coarse_T.min().item(),

                        "fine/weights/min": fine_weights.min().item(),
                        "fine/weights/max": fine_weights.max().item(),
                        "fine/weights/mean": fine_weights.mean().item(),
                        "fine/weights/std": fine_weights.std().item(),
                        "fine/weights_sum/mean": fine_weights_sum.mean().item(),
                        "fine/weights_sum/min": fine_weights_sum.min().item(),
                        "fine/weights_sum/max": fine_weights_sum.max().item(),

                        "fine/sigma/mean": fine_sigma.mean().item(),
                        "fine/sigma/max": fine_sigma.max().item(),

                        "fine/rgb/mean": fine_rgb.mean().item(),
                        "fine/rgb/max": fine_rgb.max().item(),

                        # Alpha and Transmittance
                        "fine/alpha/mean": fine_alpha.mean().item(),
                        "fine/alpha/max": fine_alpha.max().item(),
                        "fine/alpha/min": fine_alpha.min().item(),

                        "fine/transmittance/mean": fine_T.mean().item(),
                        "fine/transmittance/max": fine_T.max().item(),
                        "fine/transmittance/min": fine_T.min().item(),
                    })
                    # Log expensive diagnostics every 50 batches
                if batch_idx % 50 == 0:
                    self.wandb.log({
                        "coarse/rgb/hist": self.wandb.Histogram(coarse_rgb.detach().cpu().flatten().numpy()),
                        "fine/rgb/hist": self.wandb.Histogram(fine_rgb.detach().cpu().flatten().numpy()),

                        "coarse/weights/hist": self.wandb.Histogram(
                            coarse_weights.detach().cpu().flatten().numpy()
                        ),
                        "coarse/sigma/hist": self.wandb.Histogram(
                            coarse_sigma.detach().cpu().flatten().numpy()
                        ),
                        "fine/weights/hist": self.wandb.Histogram(
                            fine_weights.detach().cpu().flatten().numpy()
                        ),
                        "fine/sigma/hist": self.wandb.Histogram(
                            fine_sigma.detach().cpu().flatten().numpy()
                        ),
                        "coarse/alpha/hist": self.wandb.Histogram(
                            coarse_alpha.detach().cpu().flatten().numpy()
                        ),
                        "coarse/transmittance/hist": self.wandb.Histogram(
                            coarse_T.detach().cpu().flatten().numpy()
                        ),
                        "fine/alpha/hist": self.wandb.Histogram(
                            fine_alpha.detach().cpu().flatten().numpy()
                        ),
                        "fine/transmittance/hist": self.wandb.Histogram(
                            fine_T.detach().cpu().flatten().numpy()
                        ),
                    })
        return epoch_loss / len(self.train_loader)

    # Checkpoint
    def save_checkpoint(self, epoch, best_loss):
        torch.save(
            {
                "epoch": epoch,
                "best_loss": best_loss,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict(),
                "config": self.config,
            },
            self.save_dir / f"epoch_{epoch}.pth",
        )

    def load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(
            checkpoint_path,
            map_location=self.device,
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.optimizer.load_state_dict(
            checkpoint["optimizer_state_dict"]
        )

        if "scheduler_state_dict" in checkpoint:
            self.scheduler.load_state_dict(
                checkpoint["scheduler_state_dict"]
            )

        self.start_epoch = checkpoint["epoch"] + 1
        self.best_loss = checkpoint.get(
            "best_loss",
            float("inf")
        )

        print("-" * 70)
        print(f"Resumed training from: {checkpoint_path}")
        print(f"Starting Epoch : {self.start_epoch}")
        print(f"Best Loss      : {self.best_loss:.6f}")
        print("-" * 70)

    # Fit
    def fit(self):
        print("-" * 80)
        print("NeRF Training")
        print("-" * 80)

        print(f"Device            : {self.device}")
        print(f"Run Name          : {self.config['run_name']}")
        print(f"Checkpoint Dir    : {self.config['save_dir']}")

        print("\n[Model]")
        print(f"  Hidden Dim      : {self.config['hidden_dim']}")
        print(f"  Position L      : {self.config['position_L']}")
        print(f"  Direction L     : {self.config['direction_L']}")

        print("\n[Sampling]")
        print(f"  Near Plane      : {self.config['t_n']}")
        print(f"  Far Plane       : {self.config['t_f']}")
        print(f"  Coarse Samples  : {self.config['n_points']}")
        print(f"  Fine Samples    : {self.config['n_importance']}")
        print(f"  Rays / Batch    : {self.config['num_rays']}")

        print("\n[Optimization]")
        print(f"  Epochs          : {self.config['epochs']}")
        print(f"  Learning Rate   : {self.config['lr']:.1e}")

        print("=" * 80)

        start = time.time()
        best_loss = self.best_loss
        for epoch in range(self.start_epoch, self.config["epochs"] + 1):
            train_loss = self.train_one_epoch(epoch)
            current_lr = self.scheduler.get_last_lr()[0]

            if epoch % 50 == 0:
                print(
                    f"[{epoch:03d}/{self.config['epochs']}] "
                    f"Train Loss : {train_loss:.6f} | LR: {current_lr:.6f}"
                )
            self.scheduler.step()
            if train_loss < best_loss:
                best_loss = train_loss
                self.save_checkpoint(epoch, best_loss)

            if epoch % 150 == 0:
                self.render_validation_image(epoch)
                print("Rendered one image and saved to rendering folder")

        elapsed = (time.time() - start) / 60

        print("=" * 70)
        print(f"Training Finished ({elapsed:.2f} min)")
        print(f"Best Loss : {best_loss:.6f}")
        print("=" * 70)

        if self.use_wandb:
            self.wandb.finish()


    @torch.no_grad()
    def render_validation_image(self, epoch):
        self.model.eval()
        batch = next(iter(self.train_loader))
        batch = {
            k: v.to(self.device) if torch.is_tensor(v) else v
            for k, v in batch.items()
        }

        pred = self.model.render_image(batch, scale = 0.25)
        # print(
        #     pred.min().item(),
        #     pred.max().item(),
        #     pred.mean().item()
        # )
        gt = batch["target_rgb"].permute(0, 2, 3, 1)
        # print(
        #     "\ngt (inside render_validation_image):",
        #     gt.min().item(),
        #     gt.max().item(),
        #     gt.mean().item(),
        # )
        
        pred = pred[0].cpu().numpy()
        gt = gt[0].cpu().numpy()

        render_dir = self.save_dir / "renders"
        render_dir.mkdir(exist_ok=True)

        fig, ax = plt.subplots(1, 2, figsize=(10, 5))

        ax[0].imshow(gt)
        ax[0].set_title("Ground Truth")
        ax[0].axis("off")

        ax[1].imshow(pred)
        ax[1].set_title(f"Prediction (Epoch {epoch})")
        ax[1].axis("off")

        plt.tight_layout()
        plt.savefig(render_dir / f"epoch_{epoch:03d}.png", dpi=200)
        plt.close(fig)

        self.model.train()
import torch
import argparse
from nerf import NeRF
from config import config
from torch.utils.data import DataLoader
from dataset_loader import NeRFSyntheticDataset
import matplotlib.pyplot as plt


def plot_img(pred, batch):
    B = pred.shape[0]

    imgs = pred.detach().cpu().clamp(0, 1)  # (B, H, W, 3)
    gts = batch["target_rgb"].detach().cpu().permute(0, 2, 3, 1)  # (B, H, W, 3)

    fig, axes = plt.subplots(B, 2, figsize=(8, 4 * B))

    if B == 1:
        axes = axes.reshape(1, 2)

    for i in range(B):
        axes[i, 0].imshow(gts[i])
        axes[i, 0].set_title(f"Ground Truth (Sample {i})")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(imgs[i])
        axes[i, 1].set_title(f"Prediction (Sample {i})")
        axes[i, 1].axis("off")

    plt.tight_layout()
    plt.savefig("output.png")
    plt.show()

def render_image(
    root_dir="data/nerf_synthetic/lego",
    checkpoint_pth="./checkpoints/epoch_500.pth",
    from_val=False,
    n_images=1,
    scale = 1.0,
):
    config["num_rays"] = args.num_rays
    config["n_points"] = args.n_points
    config["n_importance"] = args.n_importance

    model = NeRF(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    checkpoint = torch.load(
        checkpoint_pth,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()


    if not from_val:
        dataset = NeRFSyntheticDataset(
            root_dir = root_dir,
            split="test"
        )

        loader = DataLoader(
            dataset,
            batch_size = n_images,          
            shuffle=True
        )
        render_set = "validation"

    else:
        dataset = NeRFSyntheticDataset(
            root_dir = root_dir,
            split="val"
        )

        loader = DataLoader(
            dataset,
            batch_size = n_images,          
            shuffle = True
        )
        render_set = "test"


    batch = next(iter(loader))

    batch = {
        k: v.to(device) if torch.is_tensor(v) else v
        for k, v in batch.items()
    }

    print("=" * 30)
    print("NeRF Inference")
    print("=" * 30)
    print(f"Device          : {device}")
    print(f"Checkpoint      : {checkpoint_pth}")
    print(f"Dataset         : {root_dir}")
    print(f"Split           : {render_set}")
    print(f"Images          : {n_images}")
    print(f"Render Scale    : 1.0")
    print(f"Resolution      : {batch['image'].shape[-2]} x {batch['image'].shape[-1]}")
    print(f"Coarse Samples  : {config['n_points']}")
    print(f"Fine Samples    : {config['n_importance']}")
    print(f"Chunk Size      : {config['num_rays']} rays")
    print("=" * 30)

    with torch.no_grad():
        model.eval()
        pred = model.render_image(batch, scale=scale)

    plot_img(pred = pred, batch = batch)


def parse_args():
    parser = argparse.ArgumentParser(
        description="NeRF Inference"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./checkpoints/epoch_500.pth",
        help="Path to model checkpoint."
    )

    parser.add_argument(
        "--root_dir",
        type=str,
        default="data/nerf_synthetic/lego",
        help="Dataset root directory."
    )

    parser.add_argument(
        "--split",
        choices=["test", "val"],
        default="test",
        help="Dataset split to render."
    )

    parser.add_argument(
        "--n_images",
        type=int,
        default=1,
        help="Number of images to render."
    )

    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Rendering scale."
    )

    parser.add_argument(
        "--num_rays",
        type=int,
        default=config["num_rays"],
        help="Ray chunk size."
    )

    parser.add_argument(
        "--n_points",
        type=int,
        default=config["n_points"],
        help="Number of coarse samples."
    )

    parser.add_argument(
        "--n_importance",
        type=int,
        default=config["n_importance"],
        help="Number of fine samples."
    )

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    config["num_rays"] = args.num_rays
    config["n_points"] = args.n_points
    config["n_importance"] = args.n_importance

    render_image(
        root_dir=args.root_dir,
        checkpoint_pth=args.checkpoint,
        from_val=(args.split == "val"),
        n_images=args.n_images,
        scale=args.scale,
    )
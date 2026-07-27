import torch
from nerf import NeRF
from config import config
from nerf_trainer import Trainer
from torch.utils.data import DataLoader
from dataset_loader import NeRFSyntheticDataset

def main():
    # Get the Dataset to train on
    train_dataset = NeRFSyntheticDataset(
        root_dir="data/nerf_synthetic/lego",
        split="train"
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size = 2,          
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True,
    )

    # Prepare the Model
    nerf = NeRF(config = config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trainer = Trainer(
        model=nerf,
        train_loader=train_loader,
        config=config,
        device=device,
        use_wandb=True
    )
    trainer.fit()

if __name__ == "__main__":
    main()
import os
import json
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class NeRFSyntheticDataset(Dataset):
    def __init__(self, root_dir, split="train"):
        self.root_dir = root_dir
        self.split = split

        self.transform = transforms.ToTensor()
        json_path = os.path.join(root_dir, f"transforms_{split}.json")

        with open(json_path, "r") as f:
            metadata = json.load(f)

        self.camera_angle_x = metadata["camera_angle_x"]
        self.frames = metadata["frames"]

    def __len__(self):
        return len(self.frames)

    def __getitem__(self, idx):
        frame = self.frames[idx]
        image_path = os.path.join(
            self.root_dir,
            frame["file_path"] + ".png"
        )
        image = Image.open(image_path).convert("RGBA")
        image = self.transform(image)
        rgb = image[:3]
        alpha = image[3:4]

        # Composite onto a white background
        target_rgb = rgb * alpha + (1.0 - alpha)

        transform_matrix = torch.tensor(
            frame["transform_matrix"],
            dtype=torch.float32
        )

        sample = {
            "image": image,
            "target_rgb": target_rgb,
            "camera_angle_x": self.camera_angle_x,
            "camera_pose": transform_matrix,
        }
        return sample
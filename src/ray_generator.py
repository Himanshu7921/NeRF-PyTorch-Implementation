import torch
import torch.nn as nn

class RayGenerator(nn.Module):
    """
    > The Rays are generated entirely from the camera's intrinsic parameters (focal length, principal point, image size) and extrinsic parameters (camera pose in the world)
    > Each NeRF Dataset contains:
        - Image (RGBA img)
        - Camera Pose (transform_matrix)
        - Camera Field of View (camera_angle_x)
    > A Ray is Defined by 2 Things:
        - Origin (Camera Position)
        - Direction (ray pointing to the pixel position)
    So a Ray is Defined as,
        r(t) = o + t * d
    
    > To compute Direction:
        1. Convert pixel to Camera Co-ordinates by using focal length, image center and principal point
        2. Now convert this Camera Co-ordinates to world Camera Co-ordinates by using,
            - 4×4 Camera Pose Matrix
    
    > Now we have
        - Origin (world co-ordinates) [Shape = (H*W, 3)]
        - Direction (world co-ordinates) [Shape = (H*W, 3)]
        Per Pixel


    # Original paper
    --------------------------------------------------------------------------------------------
    x = (x, y, z)
    d = (θ, φ)
    ↓
    Implementation
    --------------------------------------------------------------------------------------------
    position  = (x, y, z)
    direction = (dx, dy, dz)


    So the Approach is simply:
        1. Compute Camera Intrinsics (This is helps us to convert from image pixel coordinates to camera coordinates)
        2. Generate Pixel Coordinates
        3. Convert Pixels to Camera Coordinates
        4. Normalize Directions
        5. Extract Camera Pose
        6. Rotate Directions into World Coordinates
        7. Generate Ray Origins
        8. Pair Every Origin with Every Direction
        9. Select Training Rays
    """
    def __init__(self):
        super().__init__()
        

    def forward(self, img_batch: dict, height = None, width = None):
        # Compute Camera Intrinsics
        # -------------------------------------------------------------------------------------------
        image = img_batch["image"] # Grab the image
        device = image.device
        dtype = image.dtype
        if height is None:
            H = image.shape[-2]
        else:
            H = height

        if width is None:
            W = image.shape[-1]
        else:
            W = width
        c_x, c_y = W / 2, H / 2 # image centers
        theta = img_batch["camera_angle_x"][0]
        f_x = W / ( 2 * torch.tan(theta / 2))
        f_y = f_x  # or we could use this, f_y = H / ( 2 * torch.tan(theta / 2))

        # Generate Pixel Coordinates
        # -------------------------------------------------------------------------------------------
        v, u = torch.meshgrid(
            torch.arange(H, device=device, dtype=dtype),
            torch.arange(W, device=device, dtype=dtype),
            indexing="ij"
        )
        u = u.reshape(-1).float()
        v = v.reshape(-1).float()

        # Convert Pixels to Camera Coordinates, with the help of computed camera Intrinsics
        # -------------------------------------------------------------------------------------------
        x, y = (u - c_x) / f_x, -(v - c_y) / f_y
        z = -torch.ones_like(x) # z.shape == x.shape and z.shape == y.shape
        camera_dirs = torch.stack([x, y, z], dim = -1)

        # Normalize Directions
        # -------------------------------------------------------------------------------------------
        camera_dirs = camera_dirs / torch.norm(
            camera_dirs,
            dim=-1,
            keepdim = True
        )

        # Rotate Directions into World Coordinates: Camera Coordinates ----> World Coordinates
        # -------------------------------------------------------------------------------------------
        camera_pose = img_batch["camera_pose"]
        R = camera_pose[:, :3, :3]
        T = camera_pose[:, :3, 3]
        world_dirs = camera_dirs @ R.transpose(-1, -2)
        ray_origins = T.unsqueeze(1).expand_as(world_dirs)

        return world_dirs, ray_origins

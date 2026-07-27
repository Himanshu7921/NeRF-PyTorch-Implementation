config = {
    # NeRF
    "t_n": 2.0,
    "t_f": 6.0,
    "n_points": 64,
    "num_rays": 1024,
    "position_L": 10,
    "direction_L": 4,
    "hidden_dim": 256,

    # Rendering
    "volume_renderer_eps": 1e-10,
    "importance_sampler_eps": 1e-5,
    "n_importance": 128, # changed from 64 to 128

    # Training
    "epochs": 500,
    "lr": 5e-4,
    "lr_decay_factor": 0.1,
    "weight_decay": 0.0,

    # Logging
    "save_dir": "checkpoints/model_02", # changed from checkpoint to checkpoint/model_02
    "wandb_project": "NeRF",
    "run_name": "nerf-lego"
}
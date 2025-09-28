
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import json
from argparse import Namespace
import math
import subprocess
import sys

# --- Required 3DGS/TCNN Module Imports ---
try:
    from scene import Scene
    from gaussian_renderer import render
    from scene.gaussian_model import GaussianModel
    import tinycudann as tcnn
except ImportError as e:
    print(f"Error: A required module is missing. {e}")
    print("Please ensure this script is run from the root of the 3DGS project, or that the project is in your PYTHONPATH.")
    exit(1)

# =====================================================================================
# --- 1. Master Configuration ---
# =====================================================================================
BASE_DATASET_DIR = "/home/chz/360_v2"
DATASET_NAMES = [
    "stump"
]

# This is the base directory where your PRE-TRAINED 3DGS models are located.
G3DGS_BASE_MODEL_DIR = "output"
NERF_BASE_OUTPUT_DIR = "nerf_outputs"

# --- NeRF Distillation Configuration ---
CHECKPOINT_ITERATION = 30000
MODEL_FILENAME = "NGP_final.ckpt"

# --- Training Hyperparameters for NeRF ---
NUM_STEPS = 60000
IMAGE_RESOLUTION_DIVIDER = 2
LEARNING_RATE = 1e-3
N_SAMPLES_PER_RAY = 128
BATCH_SIZE = 2048
DATA_SAMPLING_FRACTION = 1.0

device = "cuda" if torch.cuda.is_available() else "cpu"

# =====================================================================================
# --- NeRF Model and Helper Functions (Unchanged) ---
# =====================================================================================
class InstantNGPModel(nn.Module):
    def __init__(self, scene_center, scene_scale):
        super().__init__()
        self.scene_center = scene_center.to(device)
        self.scene_scale = scene_scale # This is a scalar
        encoding_config = {"otype": "HashGrid","n_levels": 16,"n_features_per_level": 2,"log2_hashmap_size": 19,"base_resolution": 16,"per_level_scale": 1.5}
        self.xyz_encoder = tcnn.Encoding(n_input_dims=3, encoding_config=encoding_config)
        self.sigma_mlp = tcnn.Network(n_input_dims=self.xyz_encoder.n_output_dims, n_output_dims=1 + 15, network_config={"otype": "FullyFusedMLP", "activation": "ReLU", "output_activation": "None", "n_neurons": 64, "n_hidden_layers": 1})
        dir_encoding_config = {"otype": "SphericalHarmonics", "degree": 4}
        self.dir_encoder = tcnn.Encoding(n_input_dims=3, encoding_config=dir_encoding_config)
        self.color_mlp = tcnn.Network(n_input_dims=15 + self.dir_encoder.n_output_dims, n_output_dims=3, network_config={"otype": "FullyFusedMLP", "activation": "ReLU", "output_activation": "Sigmoid", "n_neurons": 64, "n_hidden_layers": 2})

    def forward(self, pts, viewdirs):
        normalized_pts = ((pts - self.scene_center) / self.scene_scale) + 0.5
        xyz_features = self.xyz_encoder(normalized_pts.float())
        sigma_mlp_output = self.sigma_mlp(xyz_features)
        sigma = torch.nn.functional.relu(sigma_mlp_output[..., 0])
        geo_features = sigma_mlp_output[..., 1:]
        view_features = self.dir_encoder(viewdirs.float())
        color_input = torch.cat([geo_features, view_features], dim=-1)
        rgb = self.color_mlp(color_input)
        return sigma, rgb

def get_rays(H, W, focal_y, c2w):
    device = c2w.device
    i, j = torch.meshgrid(torch.linspace(0.5, W - 0.5, W, device=device), torch.linspace(0.5, H - 0.5, H, device=device), indexing='xy')
    dirs = torch.stack([(i - W * 0.5) / focal_y, -(j - H * 0.5) / focal_y, -torch.ones_like(i)], -1)
    rays_d = torch.sum(dirs[..., None, :] * c2w[:3, :3], -1)
    rays_d = rays_d / torch.norm(rays_d, dim=-1, keepdim=True)
    rays_o = c2w[:3, -1].expand(rays_d.shape)
    return rays_o, rays_d

def render_rays(nerf_model, rays_o, rays_d, n_samples, near, far):
    t_vals = torch.linspace(0., 1., steps=n_samples, device=device)
    z_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * t_vals)
    z_vals = z_vals.expand([rays_o.shape[0], n_samples])
    mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
    upper = torch.cat([mids, z_vals[..., -1:]], -1)
    lower = torch.cat([z_vals[..., :1], mids], -1)
    t_rand = torch.rand(z_vals.shape, device=device)
    z_vals = lower + (upper - lower) * t_rand
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
    pts_flat = pts.reshape(-1, 3)
    viewdirs = rays_d.unsqueeze(1).expand(-1, n_samples, -1)
    viewdirs_flat = viewdirs.reshape(-1, 3)
    sigma_flat, rgb_flat = nerf_model(pts_flat, viewdirs_flat)
    sigma = sigma_flat.view(rays_o.shape[0], n_samples)
    rgb = rgb_flat.view(rays_o.shape[0], n_samples, 3)
    deltas = z_vals[..., 1:] - z_vals[..., :-1]
    delta_inf = torch.full_like(deltas[..., :1], 1e10)
    deltas = torch.cat([deltas, delta_inf], -1)
    alpha = 1. - torch.exp(-torch.nn.functional.relu(sigma) * deltas)
    T = torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1), device=device), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    weights = alpha * T
    rgb_map = torch.sum(weights[..., None] * rgb, -2)
    acc_map = torch.sum(weights, -1)
    rgb_map = rgb_map + (1. - acc_map[..., None]) # White background
    return rgb_map

def build_gpu_cache(full_camera_pool, gaussians, pipe, background, divider, fraction):
    print(f"\n--- Building GPU Cache: [Resolution Divider: {divider}, Sampling Fraction: {fraction}] ---")
    cache_rays_o_list, cache_rays_d_list, cache_colors_list = [], [], []
    temp_camera_pool = []
    for cam_info in full_camera_pool:
        opt_cam_data = cam_info['opt_cam_data']
        view = cam_info['view']
        render_w, render_h = opt_cam_data['width'] // divider, opt_cam_data['height'] // divider
        render_fy = opt_cam_data['fy'] / divider
        temp_view = Namespace(**vars(view))
        temp_view.image_width, temp_view.image_height = render_w, render_h
        temp_view.FoVx = 2 * math.atan((render_w / 2) / (opt_cam_data['fx'] / divider))
        temp_view.FoVy = 2 * math.atan((render_h / 2) / render_fy)
        temp_camera_pool.append({'view': temp_view, 'c2w': cam_info['c2w'], 'H': render_h, 'W': render_w, 'focal_y': render_fy})
    with torch.no_grad():
        for cam_info in tqdm(temp_camera_pool, desc="Rendering and Sampling for cache"):
            view, c2w, H, W, focal_y = cam_info['view'], cam_info['c2w'], cam_info['H'], cam_info['W'], cam_info['focal_y']
            gt_image_tensor = render(view, gaussians, pipe, background)["render"]
            colors_full = gt_image_tensor.permute(1, 2, 0).reshape(-1, 3)
            rays_o_full, rays_d_full = get_rays(H, W, focal_y, c2w)
            rays_o_full, rays_d_full = rays_o_full.reshape(-1, 3), rays_d_full.reshape(-1, 3)
            num_samples = int(colors_full.shape[0] * fraction)
            sample_indices = torch.randperm(colors_full.shape[0], device=device)[:num_samples]
            cache_rays_o_list.append(rays_o_full[sample_indices])
            cache_rays_d_list.append(rays_d_full[sample_indices])
            cache_colors_list.append(colors_full[sample_indices])
    gpu_cache_rays_o = torch.cat(cache_rays_o_list, 0)
    gpu_cache_rays_d = torch.cat(cache_rays_d_list, 0)
    gpu_cache_colors = torch.cat(cache_colors_list, 0)
    print(f"✅ Cache created. Total rays in cache: {gpu_cache_rays_o.shape[0]}")
    return gpu_cache_rays_o, gpu_cache_rays_d, gpu_cache_colors

# =====================================================================================
# --- NeRF Distillation (Online Training) ---
# =====================================================================================
def train_nerf(original_dataset_path, g3dgs_model_path, nerf_output_dir):
    """
    Uses a pre-trained 3DGS model as a teacher to train an Instant-NGP NeRF model.
    """
    print("\n" + "="*80)
    print(f"--- Distilling NeRF for '{os.path.basename(original_dataset_path)}' ---")
    print(f"Teacher Model Path: {g3dgs_model_path}")
    print(f"NeRF Output Directory: {nerf_output_dir}")
    print("="*80)

    gaussians = GaussianModel(sh_degree=3)
    ply_path = os.path.join(g3dgs_model_path, "point_cloud", f"iteration_{CHECKPOINT_ITERATION}", "point_cloud.ply")
    gaussians.load_ply(ply_path)
    print(f"✅ 3DGS 'Teacher' model loaded.")

    with open(os.path.join(g3dgs_model_path, "cameras.json"), 'r') as f:
        optimized_cameras_data = json.load(f)
        
    args_for_scene = Namespace(source_path=original_dataset_path, model_path=g3dgs_model_path, images='images', resolution=-1, sh_degree=3, data_device=device, eval=True, depths="", train_test_exp=False)
    scene = Scene(args_for_scene, gaussians, load_iteration=CHECKPOINT_ITERATION, shuffle=False)
    train_cameras = scene.getTrainCameras()
    
    if not train_cameras:
        print(f"\nError: Loaded 0 training cameras for {os.path.basename(original_dataset_path)}.")
        return

    full_camera_pool = []
    all_c2w_poses = []
    coord_transform_matrix = torch.tensor([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]], dtype=torch.float32, device=device)
    for view in train_cameras:
        opt_cam_data = next((c for c in optimized_cameras_data if c['img_name'] == view.image_name), None)
        if not opt_cam_data: continue
        c2w_3dgs = torch.eye(4, device=device)
        c2w_3dgs[:3, :3] = torch.tensor(opt_cam_data['rotation'], dtype=torch.float32, device=device)
        c2w_3dgs[:3, 3] = torch.tensor(opt_cam_data['position'], dtype=torch.float32, device=device)
        c2w_nerf = c2w_3dgs @ coord_transform_matrix
        all_c2w_poses.append(c2w_nerf)
        full_camera_pool.append({'view': view, 'c2w': c2w_nerf, 'opt_cam_data': opt_cam_data})

    if not full_camera_pool:
        print(f"\nError: Created a pool of 0 camera views for {os.path.basename(original_dataset_path)} after matching.")
        return
    
    # ### MODIFIED: Compute scene bounds based ONLY on camera positions ###
    print("--- Computing scene bounds from camera positions ---")
    with torch.no_grad():
        # Get bounds from all camera positions
        cam_origins = torch.stack([c2w[:3, 3] for c2w in all_c2w_poses])
        min_bounds = torch.min(cam_origins, dim=0)[0]
        max_bounds = torch.max(cam_origins, dim=0)[0]
        
        scene_center = (min_bounds + max_bounds) / 2.0
        # Add a significant margin (50%) since we're only using cameras
        # to define the bounds. This helps ensure the scene content is included.
        scene_scale = (max_bounds - min_bounds).max() * 1.5
        
        # Ensure scale is not zero for single-camera cases or co-located cameras
        if scene_scale < 1e-6:
            scene_scale = 1.0

        print(f"Scene Center (from cameras): {scene_center.cpu().numpy()}")
        print(f"Scene Scale (from cameras): {scene_scale.item():.4f}")

    # Use sub-sampling on the point cloud to compute near/far bounds to prevent memory errors
    print("--- Computing near/far bounds from camera poses and scene points ---")
    with torch.no_grad():
        all_points = gaussians.get_xyz
        
        max_points_for_bounds = 100000 
        if all_points.shape[0] > max_points_for_bounds:
            print(f"Sub-sampling {max_points_for_bounds} points (out of {all_points.shape[0]}) for near/far calculation.")
            perm = torch.randperm(all_points.shape[0], device=device)
            indices = perm[:max_points_for_bounds]
            points_subset = all_points[indices]
        else:
            points_subset = all_points

        all_depths = []
        for c2w in all_c2w_poses:
            cam_origin = c2w[:3, 3]
            cam_forward = -c2w[:3, 2]
            depths = torch.sum((points_subset - cam_origin) * cam_forward, dim=-1)
            depths = depths[depths > 0]
            all_depths.append(depths)
        
        all_depths_tensor = torch.cat(all_depths)
        near_bound = torch.quantile(all_depths_tensor, 0.01).item()
        far_bound = torch.quantile(all_depths_tensor, 0.99).item()
        near_bound = max(near_bound, 0.01)
        print(f"Dynamic Near Bound: {near_bound:.4f}, Far Bound: {far_bound:.4f}")

    print(f"✅ Created a pool of {len(full_camera_pool)} total available camera views for distillation.")

    nerf_model = InstantNGPModel(scene_center=scene_center, scene_scale=scene_scale).to(device)
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(nerf_model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.99), eps=1e-15)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99995)
    
    os.makedirs(nerf_output_dir, exist_ok=True)
    
    background = torch.tensor([1, 1, 1], dtype=torch.float32, device=device)
    pipe = Namespace(debug=False, antialiasing=False, compute_cov3D_python=False, convert_SHs_python=False)
    
    gpu_cache_rays_o, gpu_cache_rays_d, gpu_cache_colors = build_gpu_cache(full_camera_pool, gaussians, pipe, background, divider=IMAGE_RESOLUTION_DIVIDER, fraction=DATA_SAMPLING_FRACTION)

    print("\n🚀 Starting NeRF distillation training...")
    pbar = tqdm(total=NUM_STEPS, desc=f"Training NeRF for {os.path.basename(original_dataset_path)}")
    for step in range(NUM_STEPS):
        select_indices = torch.randint(0, gpu_cache_rays_o.shape[0], (BATCH_SIZE,), device=device)
        rays_o_batch, rays_d_batch, target_colors = gpu_cache_rays_o[select_indices], gpu_cache_rays_d[select_indices], gpu_cache_colors[select_indices]
        predicted_colors = render_rays(nerf_model, rays_o_batch, rays_d_batch, n_samples=N_SAMPLES_PER_RAY, near=near_bound, far=far_bound)
        loss = loss_fn(predicted_colors, target_colors)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        pbar.set_postfix(loss=f"{loss.item():.6f}", lr=f"{scheduler.get_last_lr()[0]:.1e}")
        pbar.update(1)
    pbar.close()
    
    final_model_path = os.path.join(nerf_output_dir, MODEL_FILENAME)
    save_data = {'model_state_dict': nerf_model.state_dict(), 'scene_center': scene_center, 'scene_scale': scene_scale}
    torch.save(save_data, final_model_path)
    print(f"\n--- ✅ NeRF Distillation Training Complete ---")
    print(f"🎉 Final NeRF model saved to: {final_model_path}")

# =====================================================================================
# --- Main Orchestrator ---
# =====================================================================================
if __name__ == "__main__":
    print("🚀🚀🚀 Starting Instant-NGP Distillation Pipeline from Pre-trained 3DGS 🚀🚀🚀")
    for dataset_name in DATASET_NAMES:
        print("\n" + "#"*80)
        print(f"### PROCESSING DATASET: {dataset_name.upper()} ###")
        print("#"*80)
        current_dataset_path = os.path.join(BASE_DATASET_DIR, dataset_name)
        current_3dgs_model_path = os.path.join(G3DGS_BASE_MODEL_DIR, f"{dataset_name}_3dgs_model")
        current_nerf_output_path = os.path.join(NERF_BASE_OUTPUT_DIR, f"{dataset_name}_nerf_output")
        if not os.path.isdir(current_dataset_path):
            print(f"⚠️ Warning: Dataset source directory not found for '{dataset_name}'. Skipping.")
            print(f"  (Checked path: {current_dataset_path})")
            continue
        if not os.path.isdir(current_3dgs_model_path):
            print(f"⚠️ Warning: Pre-trained 3DGS model directory not found for '{dataset_name}'. Skipping.")
            print(f"  (Checked path: {current_3dgs_model_path})")
            continue
        train_nerf(
            original_dataset_path=current_dataset_path,
            g3dgs_model_path=current_3dgs_model_path,
            nerf_output_dir=current_nerf_output_path
        )
        print(f"\n✅✅✅ Successfully finished distilling NeRF for dataset: {dataset_name} ✅✅✅")
    print("\n🎉🎉🎉 All datasets have been processed. Pipeline finished! 🎉🎉🎉")

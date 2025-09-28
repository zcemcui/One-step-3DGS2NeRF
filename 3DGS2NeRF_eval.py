import os
import torch
import json
import argparse
import math
import numpy as np
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import imageio
import plyfile
import time

# --- 1. Configuration Area ---
# Please confirm all paths and parameters here

# List of scenes to evaluate (The script will automatically check if paths exist and skip invalid entries like .txt files)
SCENES_TO_EVALUATE = [
    'stump',
]

# Parent directory paths for different components
DATASET_PARENT_DIR = "/home/chz/360_v2"
G3DGS_PARENT_DIR = "/home/chz/3DGS/gaussian-splatting/output"
NERF_PARENT_DIR = "/home/chz/3DGS/gaussian-splatting/nerf_outputs"
EVAL_PARENT_DIR = "eval_renders_maxresolution" # Render results will be saved in this folder within the current directory

# Default parameters
DOWNSCALE_FACTOR = 8
G3DGS_ITERATION = 30000
CHUNK_SIZE = 1024 * 4
N_SAMPLES = 64


# --- 2. Import Dependencies ---
try:
    from piqa import PSNR, SSIM, LPIPS
    import tinycudann as tcnn
    from scene import Scene, GaussianModel
    from gaussian_renderer import render as render_3dgs
    from utils.system_utils import searchForMaxIteration
    print("✅ All dependencies loaded successfully.")
except ImportError as e:
    print(f"❌ Dependency loading failed: {e}")
    print("Please ensure all required libraries (piqa, tinycudann, etc.) are installed and the script is located in the 3DGS project root directory.")
    exit(1)


# --- 3. NeRF Model and Rendering Functions (No change) ---
class InstantNGPModel(torch.nn.Module):
    def __init__(self, bounding_box_size=4.0):
        super().__init__()
        self.bounding_box_size = bounding_box_size
        encoding_config = {"otype": "HashGrid", "n_levels": 16, "n_features_per_level": 4, "log2_hashmap_size": 19, "base_resolution": 16, "per_level_scale": 1.5}
        self.xyz_encoder = tcnn.Encoding(n_input_dims=3, encoding_config=encoding_config)
        self.sigma_mlp = tcnn.Network(n_input_dims=self.xyz_encoder.n_output_dims, n_output_dims=1 + 15, network_config={"otype": "FullyFusedMLP", "activation": "ReLU", "output_activation": "None", "n_neurons": 64, "n_hidden_layers": 1})
        self.dir_encoder = tcnn.Encoding(n_input_dims=3, encoding_config={"otype": "SphericalHarmonics", "degree": 4})
        self.color_mlp = tcnn.Network(n_input_dims=15 + self.dir_encoder.n_output_dims, n_output_dims=3, network_config={"otype": "FullyFusedMLP", "activation": "ReLU", "output_activation": "Sigmoid", "n_neurons": 64, "n_hidden_layers": 2})
    def forward(self, pts, viewdirs):
        normalized_pts = (pts / self.bounding_box_size) + 0.5
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

def get_dynamic_near_far(camera_position, points, percentile_min=5.0, percentile_max=95.0):
    dists = torch.norm(points - camera_position, dim=-1)
    near_plane = torch.quantile(dists, percentile_min / 100.0)
    far_plane = torch.quantile(dists, percentile_max / 100.0)
    return near_plane.item() * 0.9, far_plane.item() * 1.1

def render_rays_nerf(nerf_model, rays_o, rays_d, near, far, n_samples):
    device = rays_o.device
    t_vals = torch.linspace(0., 1., steps=n_samples, device=device)
    z_vals = near * (1. - t_vals) + far * t_vals
    z_vals = z_vals.expand([rays_o.shape[0], n_samples])
    mids = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
    upper = torch.cat([mids, z_vals[..., -1:]], -1)
    lower = torch.cat([z_vals[..., :1], mids], -1)
    t_rand = torch.rand(z_vals.shape, device=device)
    z_vals = lower + (upper - lower) * t_rand
    pts = rays_o[..., None, :] + rays_d[..., None, :] * z_vals[..., :, None]
    sigma_flat, rgb_flat = nerf_model(pts.reshape(-1, 3), rays_d.unsqueeze(1).expand(-1, n_samples, -1).reshape(-1, 3))
    sigma = sigma_flat.view(rays_o.shape[0], n_samples)
    rgb = rgb_flat.view(rays_o.shape[0], n_samples, 3)
    deltas = z_vals[..., 1:] - z_vals[..., :-1]
    deltas = torch.cat([deltas, torch.full_like(deltas[..., :1], 1e10)], -1)
    alpha = 1. - torch.exp(-torch.nn.functional.relu(sigma) * deltas)
    T = torch.cumprod(torch.cat([torch.ones((alpha.shape[0], 1), device=device), 1.-alpha + 1e-10], -1), -1)[:, :-1]
    weights = alpha * T
    rgb_map = torch.sum(weights[..., None] * rgb, -2)
    acc_map = torch.sum(weights, -1)
    rgb_map = rgb_map + (1. - acc_map[..., None])
    return rgb_map

# --- 4. Define Execution Functions for Three Stages ---

def run_stage1_render_3dgs(args):
    print("\n--- STAGE 1: Rendering 3DGS (Ground Truth) Images ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(args.gt_folder).mkdir(parents=True, exist_ok=True)
    gaussians = GaussianModel(sh_degree=3)
    iteration = args.g3dgs_iteration or searchForMaxIteration(os.path.join(args.g3dgs_model_path, "point_cloud"))
    scene_args = argparse.Namespace(
        source_path=args.dataset_path, model_path=args.g3dgs_model_path, images="images",
        depths="", train_test_exp=False, resolution=-1, data_device=device, eval=True)
    scene = Scene(scene_args, gaussians, load_iteration=iteration)
    ply_path = os.path.join(args.g3dgs_model_path, "point_cloud", f"iteration_{iteration}", "point_cloud.ply")
    gaussians.load_ply(ply_path)
    test_cameras = scene.getTestCameras()
    with open(os.path.join(args.g3dgs_model_path, "cameras.json"), 'r') as f:
        optimized_cams_data = {cam['img_name']: cam for cam in json.load(f)}
    bg_color = torch.tensor([1,1,1], dtype=torch.float32, device=device)
    pipe = argparse.Namespace(debug=False, antialiasing=False, compute_cov3D_python=False, convert_SHs_python=False)
    with torch.no_grad():
        for view in tqdm(test_cameras, desc="Stage 1: Rendering 3DGS"):
            if view.image_name not in optimized_cams_data: continue
            cam_data = optimized_cams_data[view.image_name]
            render_w, render_h = cam_data['width']//args.downscale_factor, cam_data['height']//args.downscale_factor
            render_fx, render_fy = cam_data['fx']/args.downscale_factor, cam_data['fy']/args.downscale_factor
            view.image_width, view.image_height = render_w, render_h
            view.FoVx, view.FoVy = 2*math.atan(render_w/2/render_fx), 2*math.atan(render_h/2/render_fy)
            c2w = torch.eye(4, device=device)
            c2w[:3, :3], c2w[:3, 3] = torch.tensor(cam_data['rotation'], device=device, dtype=torch.float32), torch.tensor(cam_data['position'], device=device, dtype=torch.float32)
            view.world_view_transform = torch.inverse(c2w).transpose(0, 1)
            image_tensor = render_3dgs(view, gaussians, pipe, bg_color)["render"]
            image_name = f"{Path(view.image_name).stem}.png"
            imageio.imwrite(os.path.join(args.gt_folder, image_name), (image_tensor.permute(1,2,0).cpu().numpy()*255).astype('uint8'))
    print(f"✅ Stage 1 complete. GT images saved to: {args.gt_folder}")

def run_stage2_render_nerf(args):
    print("\n--- STAGE 2: Rendering NeRF (Prediction) Images ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    Path(args.pred_folder).mkdir(parents=True, exist_ok=True)
    nerf_checkpoint = torch.load(args.nerf_model_path, map_location=device)
    nerf_model = InstantNGPModel(bounding_box_size=nerf_checkpoint.get('bounding_box_size', 20.0)).to(device)
    nerf_model.load_state_dict(nerf_checkpoint['model_state_dict'])
    nerf_model.eval()
    nerf_coord_transform = torch.tensor([[1.,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1.]], dtype=torch.float32, device=device)
    iteration = args.g3dgs_iteration or searchForMaxIteration(os.path.join(args.g3dgs_model_path, "point_cloud"))
    ply_path = os.path.join(args.g3dgs_model_path, "point_cloud", f"iteration_{iteration}", "point_cloud.ply")
    plydata = plyfile.PlyData.read(ply_path)
    points_np = np.stack([plydata['vertex'][c] for c in ['x','y','z']], -1)
    scene_points = torch.from_numpy(points_np).float().to(device)
    scene_args = argparse.Namespace(
        source_path=args.dataset_path, model_path=args.g3dgs_model_path, images="images",
        depths="", train_test_exp=False, resolution=-1, data_device=device, eval=True)
    dummy_gaussians = GaussianModel(sh_degree=3)
    scene = Scene(scene_args, dummy_gaussians, shuffle=False)
    test_cameras = scene.getTestCameras()
    with open(os.path.join(args.g3dgs_model_path, "cameras.json"), 'r') as f:
        optimized_cams_data = {cam['img_name']: cam for cam in json.load(f)}
    with torch.no_grad():
        for view in tqdm(test_cameras, desc="Stage 2: Rendering NeRF"):
            if view.image_name not in optimized_cams_data: continue
            cam_data = optimized_cams_data[view.image_name]
            render_w, render_h = cam_data['width']//args.downscale_factor, cam_data['height']//args.downscale_factor
            render_fy = cam_data['fy']/args.downscale_factor
            c2w_3dgs = torch.eye(4, device=device)
            c2w_3dgs[:3, :3], c2w_3dgs[:3, 3] = torch.tensor(cam_data['rotation'], device=device, dtype=torch.float32), torch.tensor(cam_data['position'], device=device, dtype=torch.float32)
            c2w_nerf = c2w_3dgs @ nerf_coord_transform
            rays_o_full, rays_d_full = get_rays(render_h, render_w, render_fy, c2w_nerf)
            near, far = get_dynamic_near_far(c2w_nerf[:3, 3], scene_points)
            all_pixels = []
            for i in range(0, render_h*render_w, args.chunk_size):
                rays_o, rays_d = rays_o_full.reshape(-1,3)[i:i+args.chunk_size], rays_d_full.reshape(-1,3)[i:i+args.chunk_size]
                pixels = render_rays_nerf(nerf_model, rays_o, rays_d, near=near, far=far, n_samples=args.n_samples)
                all_pixels.append(pixels)
            image_tensor = torch.cat(all_pixels, 0).view(render_h, render_w, 3)
            image_name = f"{Path(view.image_name).stem}.png"
            Image.fromarray((image_tensor.cpu().clamp(0,1)*255).byte().numpy()).save(os.path.join(args.pred_folder, image_name))
    print(f"✅ Stage 2 complete. Prediction images saved to: {args.pred_folder}")

def run_stage3_compare(args):
    print("\n--- STAGE 3: Comparing Images and Calculating Metrics ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    psnr_metric, ssim_metric, lpips_metric = PSNR().to(device), SSIM().to(device), LPIPS(network='alex').to(device)
    psnr_scores, ssim_scores, lpips_scores = [], [], []
    gt_images = sorted(list(Path(args.gt_folder).glob("*.png")))
    if not gt_images:
        print(f"❌ ERROR: No .png images found in the GT folder {args.gt_folder}.")
        return None
    for gt_path in tqdm(gt_images, desc="Stage 3: Comparing Images"):
        pred_path = os.path.join(args.pred_folder, gt_path.name)
        if not os.path.exists(pred_path): continue
        gt_image = (torch.from_numpy(np.array(Image.open(gt_path).convert("RGB"))).float().to(device)/255.0).permute(2,0,1)
        pred_image = (torch.from_numpy(np.array(Image.open(pred_path).convert("RGB"))).float().to(device)/255.0).permute(2,0,1)
        gt_batch, pred_batch = gt_image.unsqueeze(0), pred_image.unsqueeze(0)
        psnr_scores.append(psnr_metric(pred_batch, gt_batch).item())
        ssim_scores.append(ssim_metric(pred_batch, gt_batch).item())
        lpips_scores.append(lpips_metric(pred_batch, gt_batch).item())
    if not psnr_scores:
        print("❌ ERROR: No images were evaluated. Please check if filenames match in both folders.")
        return None
    
    avg_psnr = np.mean(psnr_scores)
    avg_ssim = np.mean(ssim_scores)
    avg_lpips = np.mean(lpips_scores)
    
    print(f"✅ Stage 3 complete for scene.")
    return {"PSNR": avg_psnr, "SSIM": avg_ssim, "LPIPS": avg_lpips}

# --- 5. Final Report Generation and Saving Function ---
def process_and_save_results(all_results):
    # --- Generate Summary Report String ---
    report_lines = []
    header1 = "="*80
    header2 = " " * 28 + "FINAL EVALUATION SUMMARY"
    
    report_lines.append("\n\n" + header1)
    report_lines.append(header2)
    report_lines.append(header1)
    report_lines.append(f"{'Scene':<15} | {'PSNR ↑':>10} | {'SSIM ↑':>10} | {'LPIPS ↓':>10}")
    report_lines.append("-"*80)
    
    avg_psnr, avg_ssim, avg_lpips = [], [], []
    
    for scene, metrics in all_results.items():
        psnr, ssim, lpips = metrics['PSNR'], metrics['SSIM'], metrics['LPIPS']
        report_lines.append(f"{scene:<15} | {psnr:>10.4f} | {ssim:>10.4f} | {lpips:>10.4f}")
        avg_psnr.append(psnr)
        avg_ssim.append(ssim)
        avg_lpips.append(lpips)
        
    average_results = {}
    if len(avg_psnr) > 1:
        report_lines.append("-"*80)
        avg_p, avg_s, avg_l = np.mean(avg_psnr), np.mean(avg_ssim), np.mean(avg_lpips)
        report_lines.append(f"{'Average':<15} | {avg_p:>10.4f} | {avg_s:>10.4f} | {avg_l:>10.4f}")
        average_results = {"PSNR": avg_p, "SSIM": avg_s, "LPIPS": avg_l}

    report_lines.append("="*80)
    summary_string = "\n".join(report_lines)

    # --- Print to Terminal ---
    print(summary_string)
    
    # --- Save Results to File ---
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    
    # Save .txt plain text report
    txt_filename = f"evaluation_summary_{timestamp}.txt"
    with open(txt_filename, 'w', encoding='utf-8') as f:
        f.write(summary_string)
    print(f"\n✅ Summary report saved to: {txt_filename}")

    # Save .json structured data
    json_filename = f"evaluation_results_{timestamp}.json"
    data_to_save = {
        "run_timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "scenes_evaluated": list(all_results.keys()),
        "per_scene_results": all_results,
        "average_results": average_results
    }
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(data_to_save, f, indent=4, ensure_ascii=False)
    print(f"✅ Detailed JSON results saved to: {json_filename}")

# --- 6. Main Execution Logic ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fully automated evaluation pipeline for all datasets.")
    parser.add_argument('--stage', type=str, default='all', choices=['render_3dgs', 'render_nerf', 'compare', 'all'], help="The pipeline stage to execute. 'all' runs everything.")
    args = parser.parse_args()
    
    all_results = {}
    start_time = time.time()

    for scene_name in SCENES_TO_EVALUATE:
        print(f"\n{'='*25} PROCESSING SCENE: {scene_name.upper()} {'='*25}")
        
        scene_args = argparse.Namespace(
            dataset_path=os.path.join(DATASET_PARENT_DIR, scene_name),
            g3dgs_model_path=os.path.join(G3DGS_PARENT_DIR, f"{scene_name}_3dgs_model"),
            nerf_model_path=os.path.join(NERF_PARENT_DIR, f"{scene_name}_nerf_output", "NGP_final.ckpt"),
            gt_folder=os.path.join(EVAL_PARENT_DIR, scene_name, "3dgs_gt"),
            pred_folder=os.path.join(EVAL_PARENT_DIR, scene_name, "nerf_pred"),
            downscale_factor=DOWNSCALE_FACTOR,
            g3dgs_iteration=G3DGS_ITERATION,
            chunk_size=CHUNK_SIZE,
            n_samples=N_SAMPLES
        )

        if not os.path.exists(scene_args.dataset_path) or \
           not os.path.exists(scene_args.g3dgs_model_path) or \
           not os.path.exists(scene_args.nerf_model_path):
            print(f"⚠️  WARNING: Scene '{scene_name}' is missing necessary input files/directories. Skipping this scene.")
            print(f"  - Checking dataset path: {scene_args.dataset_path}")
            print(f"  - Checking 3DGS model path: {scene_args.g3dgs_model_path}")
            print(f"  - Checking NeRF model path: {scene_args.nerf_model_path}")
            continue

        try:
            if args.stage in ['all', 'render_3dgs']:
                run_stage1_render_3dgs(scene_args)
            if args.stage in ['all', 'render_nerf']:
                run_stage2_render_nerf(scene_args)
            if args.stage in ['all', 'compare']:
                result = run_stage3_compare(scene_args)
                if result:
                    all_results[scene_name] = result
        except Exception as e:
            print(f"❌❌❌ CRITICAL ERROR occurred while processing scene '{scene_name}': {e}")
            print(f"Skipping this scene and moving to the next.")
            import traceback
            traceback.print_exc()

    if all_results:
        process_and_save_results(all_results)
    
    total_time = time.time() - start_time
    print(f"\n🎉 Entire process finished successfully, Total time: {total_time // 60:.0f} min {total_time % 60:.2f} sec.")

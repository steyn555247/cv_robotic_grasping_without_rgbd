import os
import trimesh
import yaml
import numpy as np
import cv2
import torch
import pyrender

from PIL import Image
from estimater import Any6D

from foundationpose.Utils import get_bounding_box, visualize_frame_results, calculate_chamfer_distance_gt_mesh, align_mesh_to_coordinate
import nvdiffrast.torch as dr
import argparse
from pytorch_lightning import seed_everything

from sam2_instantmesh import *

glctx = dr.RasterizeCudaContext()

def patch_estimator_for_memory(est):
    if hasattr(est, 'scorer'):   
        if hasattr(est.scorer, 'cfg') and 'crop_ratio' in est.scorer.cfg:
            est.scorer.cfg['crop_ratio'] = 1.1  # 1.5→1.1
    return est

if __name__=='__main__':

    seed_everything(0)

    parser = argparse.ArgumentParser(description="Set experiment name and paths")
    parser.add_argument("--ycb_model_path", type=str, default="/home/miruware/ssd_4tb/dataset/ho3d/YCB_Video_Models", help="Path to the YCB Video Models")
    parser.add_argument("--obj", type=str, default="mustard", help="Object name")
    parser.add_argument("--img_to_3d", action="store_true", help="Running with InstantMesh+SAM2")
    parser.add_argument("--demo_path", type=str, default="demo_data", help="Path to demo data")
    parser.add_argument("--max_size", type=int, default=320, help="Max image dimension (320 for T4 GPU)")
    args = parser.parse_args()

    ycb_model_path = args.ycb_model_path
    img_to_3d = args.img_to_3d
    obj = args.obj

    results = []
    demo_path = f'demo_data_{obj}'
    mesh_path = os.path.join(demo_path, f'{obj}.obj')

    save_path = f'results/demo_{obj}'
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    depth_scale = 1000.0
    color_original = cv2.cvtColor(cv2.imread(os.path.join(demo_path, 'color.png')), cv2.COLOR_BGR2RGB)
    depth_image_raw = cv2.imread(os.path.join(demo_path, 'depth.png'), cv2.IMREAD_ANYDEPTH).astype(np.float32)
    
    h_orig, w_orig = color_original.shape[:2]
    
    # Removed outliers
    p1, p99 = np.percentile(depth_image_raw, [1, 99])
    depth_clipped = np.clip(depth_image_raw, p1, p99)
    
    # convert disparity to depth
    epsilon = 1e-6
    depth_inv = 1.0 / (depth_clipped + epsilon)
    
    # Normalize to [0,1] then scale to 0-5m range
    depth_norm = (depth_inv - depth_inv.min()) / (depth_inv.max() - depth_inv.min() + epsilon)
    depth_image_original = depth_norm * 5000.0
    depth_image_original = np.clip(depth_image_original, 100, 10000).astype(np.float32)
    
    label_file = np.load(os.path.join(demo_path, 'labels_filtered.npz'), allow_pickle=True)
    label = label_file['arr_0']
    label = label.item()
    obj_num = 0
    mask_original = label['segmentation'].astype(np.bool_)
    
    intrinsic_path = f"{demo_path}/836212060125_640x480.yml"
    with open(intrinsic_path, 'r') as file:
        data = yaml.load(file, Loader=yaml.FullLoader)

    intrinsic_original = np.array([
        [data["depth"]["fx"], 0.0, data["depth"]["ppx"]], 
        [0.0, data["depth"]["fy"], data["depth"]["ppy"]], 
        [0.0, 0.0, 1.0]
    ])
    
    MAX_SIZE = args.max_size
    h, w = color_original.shape[:2]
    
    if max(h, w) > MAX_SIZE:
        scale = MAX_SIZE / max(h, w)
        new_h = int(h * scale)
        new_w = int(w * scale)
        # Make dimensions divisible by 32 (required for neural networks)
        new_h = (new_h // 32) * 32
        new_w = (new_w // 32) * 32
        
        print(f"  Resizing: {w}x{h} → {new_w}x{new_h}")
        print(f"  Scale factor: {scale:.4f}")
        print(f"  Memory savings: ~{(1-scale**2)*100:.0f}%")
        
        # Resize images
        color = cv2.resize(color_original, (new_w, new_h), interpolation=cv2.INTER_AREA)
        depth_image = cv2.resize(depth_image_original, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask_original.astype(np.uint8), (new_w, new_h), interpolation=cv2.INTER_NEAREST).astype(bool)
        
        # Scale intrinsics to match resized images
        intrinsic = intrinsic_original.copy()
        intrinsic[0, 0] *= scale  # fx
        intrinsic[1, 1] *= scale  # fy
        intrinsic[0, 2] *= scale  # cx
        intrinsic[1, 2] *= scale  # cy
        
        print(f"  ✓ Scaled K: fx={intrinsic[0,0]:.1f}, fy={intrinsic[1,1]:.1f}")
        
        # Save data for later use
        metadata = {
            'image_scale': scale,  # Renamed for clarity
            'original_size': (w_orig, h_orig),
            'resized_size': (new_w, new_h),
            'max_size': MAX_SIZE
        }
        np.save(os.path.join(save_path, f'{obj}_resize_metadata.npy'), metadata)
        
        # Save both original and scaled intrinsics
        np.savetxt(os.path.join(save_path, f'K_original.txt'), intrinsic_original)
        np.savetxt(os.path.join(save_path, f'K_scaled.txt'), intrinsic)
                
    else:
        print(f"  Images already small enough: {w}x{h}")
        color = color_original.copy()
        depth_image = depth_image_original.copy()
        mask = mask_original.copy()
        intrinsic = intrinsic_original.copy()
        scale = 1.0
        new_w, new_h = w, h
        
        metadata = {
            'image_scale': 1.0,
            'original_size': (w_orig, h_orig),
            'resized_size': (w, h),
            'max_size': MAX_SIZE
        }
        np.save(os.path.join(save_path, f'{obj}_resize_metadata.npy'), metadata)
        np.savetxt(os.path.join(save_path, f'K_original.txt'), intrinsic_original)
    
    # Normalize depth for FoundationPose
    depth = np.zeros_like(depth_image, dtype=np.float32)
    cv2.normalize(depth_image, depth, 0, 1, cv2.NORM_MINMAX)
    depth = np.clip(depth, 0.0, 1.0).astype(np.float32)
    
    Image.fromarray(color).save(os.path.join(save_path, 'color_resized.png'))

    if img_to_3d:
        print(f"\n[STEP 6] Generating 3D mesh with InstantMesh...")
        cmin, rmin, cmax, rmax = get_bounding_box(mask).astype(np.int32)
        input_box = np.array([cmin, rmin, cmax, rmax])[None, :]
        mask = running_sam_box(color, input_box)

        input_image = preprocess_image(color, mask, save_path, obj)
        images = diffusion_image_generation(save_path, save_path, obj, input_image=input_image)
        instant_mesh_process(images, save_path, obj)

        mesh = trimesh.load(os.path.join(save_path, f'mesh_{obj}.obj'))
        mesh = align_mesh_to_coordinate(mesh)
        mesh.export(os.path.join(save_path, f'center_mesh_{obj}.obj'))
        mesh = trimesh.load(os.path.join(save_path, f'center_mesh_{obj}.obj'))
        
    else:
        mesh = trimesh.load(mesh_path)
    
    est = Any6D(symmetry_tfs=None, mesh=mesh, debug_dir=save_path, debug=2)
    est = patch_estimator_for_memory(est)

    # Save final intrinsics used for pose estimation
    np.savetxt(os.path.join(save_path, f'K.txt'), intrinsic)

    # Clear GPU memory before pose estimation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        allocated = torch.cuda.memory_allocated() / 1024**3
        print(f"  GPU memory before: {allocated:.2f} GB\n")
    
    pred_pose = est.register_any6d(K=intrinsic, rgb=color,depth=depth,ob_mask=mask,iteration=5,name=f'demo')
    
    print("POSE ESTIMATION SUCCESSFUL!")

    print(f"  Center of image: {est.get_tf_to_centered_mesh()}")
    print(f"  Translation bbox: {est.guess_translation_bounding_box(depth_image, mask, K=intrinsic)}")
    print(f"  Center of object: {est.guess_translation(depth_image, mask, K=intrinsic)}")
    
    np.savetxt(os.path.join(save_path, f'{obj}_initial_pose.txt'), pred_pose)
    
    est.mesh.export(os.path.join(save_path, f'final_mesh_{obj}.obj'))
    
    all_poses = est.poses
    all_scores = est.scores
    
    np.save(os.path.join(save_path, f'{obj}_poses.npy'), all_poses.cpu().numpy())
    np.save(os.path.join(save_path, f'{obj}_scores.npy'), all_scores.cpu().numpy())
    
    mesh_scale = est.mesh_scale
    # Saving mesh_scale
    np.save(os.path.join(save_path, f'{obj}_mesh_scale.npy'), mesh_scale)
    
    metadata_path = os.path.join(save_path, f'{obj}_resize_metadata.npy')
    if os.path.exists(metadata_path):
        metadata = np.load(metadata_path, allow_pickle=True).item()
        metadata['mesh_scale'] = mesh_scale
        np.save(metadata_path, metadata)
    
    results.append({
        'Object': obj,
        'Object_Number': obj_num,
    })
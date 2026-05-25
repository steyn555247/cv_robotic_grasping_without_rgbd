import cv2
import numpy as np
from tqdm import tqdm
import time

# =============================================================================
# HELPER FUNCTION: ADJUST GRASP COORDINATES AFTER CROPPING
# =============================================================================

def adjust_grasp_coordinates(grasps, crop_bbox):
    """
    Adjust ground truth grasp coordinates after cropping.

    Args:
        grasps: List of grasp dictionaries with 'center', 'angle', 'width', 'height', 'corners'
        crop_bbox: (x, y, w, h) bounding box of the crop

    Returns:
        List of adjusted grasp dictionaries
    """
    x_offset, y_offset, crop_w, crop_h = crop_bbox
    adjusted_grasps = []

    for grasp in grasps:
        # Get center coordinates
        center_x, center_y = grasp['center']

        # Adjust center coordinates
        new_center_x = center_x - x_offset
        new_center_y = center_y - y_offset

        # Only keep grasps that are within the cropped region
        if 0 <= new_center_x < crop_w and 0 <= new_center_y < crop_h:
            # Adjust all corner points
            adjusted_corners = []
            all_corners_valid = True

            for corner in grasp['corners']:
                adj_corner_x = corner[0] - x_offset
                adj_corner_y = corner[1] - y_offset
                adjusted_corners.append((adj_corner_x, adj_corner_y))

            # Create adjusted grasp dictionary
            adjusted_grasp = {
                'center': (new_center_x, new_center_y),
                'angle': grasp['angle'],
                'angle_deg': grasp['angle_deg'],
                'width': grasp['width'],
                'height': grasp['height'],
                'corners': np.array(adjusted_corners)
            }
            adjusted_grasps.append(adjusted_grasp)

    return adjusted_grasps


# =============================================================================
# EVALUATION WITH WHITE SURFACE CROPPING
# =============================================================================

print("=" * 80)
print("CORNELL EVALUATION WITH WHITE SURFACE CROPPING")
print("=" * 80)
print(f"Pipeline: Crop White Surface → Mask → Find COG → Detect Grasps")
print(f"White Crop Enabled: {ENABLE_WHITE_CROP}")
print("=" * 80)

# =============================================================================
# STEP 1: PRECOMPUTE DEPTH WITH CROPPING
# =============================================================================

cached_data = []
crop_stats = {
    'total': 0,
    'cropped_success': 0,
    'no_surface_detected': 0,
    'no_grasps_in_crop': 0,
    'fallback_to_original': 0
}

start_time = time.time()

for sample in tqdm(samples, desc="Computing depth maps with cropping"):
    try:
        # Load RGB image
        rgb = cv2.imread(sample["rgb_path"])
        if rgb is None:
            continue
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        crop_stats['total'] += 1
        grasps_to_use = sample["grasps"]

        # Crop white surface if enabled
        if ENABLE_WHITE_CROP:
            cropped_rgb, crop_bbox, crop_success = crop_white_surface(
                rgb,
                lower_white=WHITE_LOWER,
                upper_white=WHITE_UPPER,
                min_area_ratio=MIN_SURFACE_AREA
            )

            if crop_success:
                # Adjust ground truth grasps to cropped coordinates
                adjusted_grasps = adjust_grasp_coordinates(sample["grasps"], crop_bbox)

                if len(adjusted_grasps) == 0:
                    # No grasps in cropped region - fall back to original
                    crop_stats['no_grasps_in_crop'] += 1
                    crop_stats['fallback_to_original'] += 1
                else:
                    # Successfully cropped with valid grasps
                    rgb = cropped_rgb
                    grasps_to_use = adjusted_grasps
                    crop_stats['cropped_success'] += 1
            else:
                # White surface detection failed - use original
                crop_stats['no_surface_detected'] += 1
                crop_stats['fallback_to_original'] += 1

        # Compute depth on (possibly cropped) RGB
        depth, _ = depth_model.predict(rgb)

        cached_data.append({
            "rgb": rgb,
            "depth": depth,
            "grasps": grasps_to_use,
        })

    except Exception as e:
        print(f"\nError processing {sample.get('rgb_path', 'unknown')}: {e}")
        import traceback
        traceback.print_exc()
        continue

depth_time = time.time() - start_time

print(f"\n✓ Successfully cached {len(cached_data)} / {len(samples)} samples")
if ENABLE_WHITE_CROP:
    print(f"\nCropping Statistics:")
    print(f"  Total samples:           {crop_stats['total']}")
    print(f"  Successfully cropped:    {crop_stats['cropped_success']} ({crop_stats['cropped_success']/crop_stats['total']*100:.1f}%)")
    print(f"  No surface detected:     {crop_stats['no_surface_detected']}")
    print(f"  No grasps in crop:       {crop_stats['no_grasps_in_crop']}")
    print(f"  Fallback to original:    {crop_stats['fallback_to_original']}")
print(f"\nDepth computation time: {depth_time:.1f}s")

# =============================================================================
# STEP 2: RUN GRASP DETECTION ON CACHED DATA
# =============================================================================

detector = GraspDetector(
    BEST_PARAMS["w_edge"],
    BEST_PARAMS["w_depth"],
    BEST_PARAMS["w_cog"],
)

top1_successes = 0
top5_successes = 0
any_successes = 0
ious = []
angle_diffs = []
total = 0

start_time = time.time()

for data in tqdm(cached_data, desc="Evaluating grasp detection"):
    try:
        # Run grasp detection (cropping already done in Step 1)
        grasps, _ = detector.process(
            rgb=data["rgb"],
            depth=data["depth"],
            n_grasps=BEST_PARAMS["num_grasps"],
            pct=BEST_PARAMS["depth_percentile"],
            mult=BEST_PARAMS["candidate_multiplier"],
            min_l=BEST_PARAMS["min_grasp_length"],
            max_l=BEST_PARAMS["max_grasp_length"],
            algo=BEST_PARAMS["ray_algorithm"],
            boost=BEST_PARAMS["cog_boost"],
            grad_src=BEST_PARAMS["gradient_source"],
            enable_crop=False  # Already cropped in Step 1
        )

        # Convert to GraspCandidate format for evaluation
        predictions = [
            GraspCandidate(
                x=g.x,
                y=g.y,
                angle=g.angle,
                width=g.width,
                height=g.height,
            )
            for g in grasps
        ]

        # Evaluate predictions against ground truth
        eval_result = evaluator.evaluate_image(predictions, data["grasps"])

        total += 1
        if eval_result.get("top1_success", False):
            top1_successes += 1
        if eval_result.get("top5_success", False):
            top5_successes += 1
        if eval_result.get("any_success", False):
            any_successes += 1

        ious.append(eval_result.get("avg_iou", 0.0))
        angle_diffs.append(eval_result.get("avg_angle_diff", 0.0))

    except Exception as e:
        print(f"\nError evaluating sample, skipping: {e}")
        import traceback
        traceback.print_exc()
        continue

eval_time = time.time() - start_time

# =============================================================================
# STEP 3: PRINT RESULTS
# =============================================================================

if total == 0:
    print("\n❌ No samples were successfully evaluated.")
else:
    top1_acc = top1_successes / total * 100.0
    top5_acc = top5_successes / total * 100.0
    any_acc = any_successes / total * 100.0
    mean_iou = float(np.mean(ious)) if ious else 0.0
    mean_angle = float(np.mean(angle_diffs)) if angle_diffs else 0.0

    print("\n" + "=" * 80)
    print("FINAL RESULTS WITH WHITE SURFACE CROPPING")
    print("=" * 80)
    print(f"Images evaluated      : {total}")
    print(f"Top-1 accuracy        : {top1_acc:.2f}%")
    print(f"Top-5 accuracy        : {top5_acc:.2f}%")
    print(f"Any-success accuracy  : {any_acc:.2f}%")
    print(f"Average IoU           : {mean_iou:.4f}")
    print(f"Average angle diff    : {mean_angle:.2f} degrees")
    print(f"Evaluation time       : {eval_time:.1f}s")
    print(f"Total time            : {depth_time + eval_time:.1f}s")
    print("=" * 80)

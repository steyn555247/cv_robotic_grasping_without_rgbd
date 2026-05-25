import cv2
import numpy as np
from tqdm import tqdm
import time

# =============================================================================
# MANUAL CROP EVALUATION
# Horizontal: 100-500 (width=400)
# Vertical: 150-450 (height=300)
# =============================================================================

def manual_crop(image, crop_bbox):
    """Manually crop image to specified region."""
    x, y, w, h = crop_bbox
    img_h, img_w = image.shape[:2]
    x = max(0, min(x, img_w - 1))
    y = max(0, min(y, img_h - 1))
    w = min(w, img_w - x)
    h = min(h, img_h - y)
    cropped = image[y:y+h, x:x+w].copy()
    return cropped, (x, y, w, h), True


def adjust_grasp_coordinates(grasps, crop_bbox):
    """Adjust ground truth grasp coordinates after cropping."""
    x_offset, y_offset, crop_w, crop_h = crop_bbox
    adjusted_grasps = []

    for grasp in grasps:
        center_x, center_y = grasp['center']
        new_center_x = center_x - x_offset
        new_center_y = center_y - y_offset

        if 0 <= new_center_x < crop_w and 0 <= new_center_y < crop_h:
            adjusted_corners = []
            for corner in grasp['corners']:
                adj_corner = (corner[0] - x_offset, corner[1] - y_offset)
                adjusted_corners.append(adj_corner)

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
# CONFIGURATION
# =============================================================================

# MANUAL CROP REGION
# Horizontal: 100 to 500 (x=100, width=400)
# Vertical: 150 to 450 (y=150, height=300)
MANUAL_CROP_BBOX = (100, 150, 400, 300)

ENABLE_MANUAL_CROP = True

print("=" * 80)
print("CORNELL EVALUATION WITH MANUAL CROP")
print("=" * 80)
print(f"Manual Crop Enabled: {ENABLE_MANUAL_CROP}")
x, y, w, h = MANUAL_CROP_BBOX
print(f"Crop Region:")
print(f"  Horizontal: {x} to {x+w} (width={w})")
print(f"  Vertical:   {y} to {y+h} (height={h})")
print(f"  Format: x={x}, y={y}, width={w}, height={h}")
print("=" * 80)

# =============================================================================
# STEP 1: CROP + DEPTH COMPUTATION
# =============================================================================

cached_data_manual = []
crop_stats = {
    'total': 0,
    'cropped': 0,
    'skipped_no_grasps': 0,
    'grasps_removed': 0,
    'grasps_kept': 0
}

print("\nStep 1: Manual cropping and depth computation...")
start_time = time.time()

for sample in tqdm(samples, desc="Manual crop + depth"):
    try:
        # Load RGB
        rgb = cv2.imread(sample["rgb_path"])
        if rgb is None:
            continue
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        crop_stats['total'] += 1
        grasps_to_use = sample["grasps"]
        crop_info = {'cropped': False}

        # MANUAL CROP
        if ENABLE_MANUAL_CROP:
            cropped_rgb, crop_bbox, success = manual_crop(rgb, MANUAL_CROP_BBOX)

            if success:
                # Adjust ground truth
                original_grasp_count = len(sample["grasps"])
                adjusted_grasps = adjust_grasp_coordinates(sample["grasps"], crop_bbox)

                grasps_removed = original_grasp_count - len(adjusted_grasps)
                crop_stats['grasps_removed'] += grasps_removed
                crop_stats['grasps_kept'] += len(adjusted_grasps)

                if len(adjusted_grasps) > 0:
                    rgb = cropped_rgb
                    grasps_to_use = adjusted_grasps
                    crop_stats['cropped'] += 1
                    crop_info = {
                        'cropped': True,
                        'crop_bbox': crop_bbox,
                        'grasps_removed': grasps_removed
                    }
                else:
                    # All grasps outside crop region - skip this sample
                    crop_stats['skipped_no_grasps'] += 1
                    continue

        # Compute depth
        depth, _ = depth_model.predict(rgb)

        cached_data_manual.append({
            "rgb": rgb,
            "depth": depth,
            "grasps": grasps_to_use,
            "crop_info": crop_info,
            "original_path": sample["rgb_path"]
        })

    except Exception as e:
        print(f"\nError: {sample.get('rgb_path', 'unknown')}: {e}")
        continue

depth_time = time.time() - start_time

print(f"\n✓ Processed {len(cached_data_manual)}/{len(samples)} samples")
if ENABLE_MANUAL_CROP:
    print(f"\n{'='*80}")
    print("MANUAL CROP STATISTICS")
    print(f"{'='*80}")
    print(f"Total samples:               {crop_stats['total']}")
    print(f"Successfully cropped:        {crop_stats['cropped']} ({crop_stats['cropped']/crop_stats['total']*100:.1f}%)")
    print(f"Skipped (no grasps in crop): {crop_stats['skipped_no_grasps']}")
    print(f"Samples with data:           {len(cached_data_manual)}")
    print(f"Grasps kept:                 {crop_stats['grasps_kept']}")
    print(f"Grasps removed (outside):    {crop_stats['grasps_removed']}")
    print(f"{'='*80}")

print(f"\nDepth computation time: {depth_time:.1f}s")

# =============================================================================
# STEP 2: GRASP DETECTION
# =============================================================================

print("\nStep 2: Running grasp detection on manually cropped images...")

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

for data in tqdm(cached_data_manual, desc="Evaluating"):
    try:
        # Run grasp detection
        grasps, info = detector.process(
            rgb=data["rgb"],
            depth=data["depth"],
            n_grasps=BEST_PARAMS["num_grasps"],
            pct=BEST_PARAMS["depth_percentile"],
            mult=BEST_PARAMS["candidate_multiplier"],
            min_l=BEST_PARAMS["min_grasp_length"],
            max_l=BEST_PARAMS["max_grasp_length"],
            algo=BEST_PARAMS["ray_algorithm"],
            boost=BEST_PARAMS["cog_boost"],
            grad_src=BEST_PARAMS["gradient_source"]
        )

        # Convert to GraspCandidate
        predictions = [
            GraspCandidate(x=g.x, y=g.y, angle=g.angle, width=g.width, height=g.height)
            for g in grasps
        ]

        # Evaluate
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

        # Store for visualization
        data["eval_result"] = eval_result
        data["predictions"] = predictions
        data["detection_info"] = info

    except Exception as e:
        print(f"\nError: {e}")
        continue

eval_time = time.time() - start_time

# =============================================================================
# STEP 3: RESULTS
# =============================================================================

if total == 0:
    print("\n❌ No samples evaluated.")
else:
    top1_acc = top1_successes / total * 100.0
    top5_acc = top5_successes / total * 100.0
    any_acc = any_successes / total * 100.0
    mean_iou = float(np.mean(ious)) if ious else 0.0
    mean_angle = float(np.mean(angle_diffs)) if angle_diffs else 0.0

    print("\n" + "=" * 80)
    print("RESULTS WITH MANUAL CROP")
    print("=" * 80)
    print(f"Crop Region: Horizontal 100-500, Vertical 150-450")
    print(f"  (x=100, y=150, width=400, height=300)")
    print("-" * 80)
    print(f"Images evaluated          : {total}")
    print(f"Top-1 accuracy            : {top1_acc:.2f}%")
    print(f"Top-5 accuracy            : {top5_acc:.2f}%")
    print(f"Any-success accuracy      : {any_acc:.2f}%")
    print(f"Average IoU               : {mean_iou:.4f}")
    print(f"Average angle difference  : {mean_angle:.2f}°")
    print("-" * 80)
    print(f"Evaluation time           : {eval_time:.1f}s")
    print(f"Total time                : {depth_time + eval_time:.1f}s")
    print("=" * 80)

# Store for visualization
cached_data = cached_data_manual

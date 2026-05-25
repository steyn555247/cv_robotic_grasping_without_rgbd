import cv2
import numpy as np
from tqdm import tqdm
import time

# =============================================================================
# STRICT WHITE SURFACE CROPPING FUNCTION
# =============================================================================

def crop_white_surface_strict(image, white_threshold=210, min_area_ratio=0.20,
                               aggressive_clean=True):
    """
    Strictly crop to only the white surface and objects.
    Removes ALL background, desk edges, shadows, etc.
    """
    h, w = image.shape[:2]

    # STRICT white detection - all RGB channels must be >= threshold
    r, g, b = image[:,:,0], image[:,:,1], image[:,:,2]
    white_mask = ((r >= white_threshold) &
                  (g >= white_threshold) &
                  (b >= white_threshold)).astype(np.uint8) * 255

    # Grayscale must also agree
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray_mask = (gray >= white_threshold).astype(np.uint8) * 255
    combined_mask = cv2.bitwise_and(white_mask, gray_mask)

    # Aggressive morphological cleaning
    if aggressive_clean:
        kernel_large = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
        kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_large, iterations=4)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel_small, iterations=3)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_large, iterations=2)
    else:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=2)

    # Find largest white region
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return image, (0, 0, w, h), False

    largest_contour = max(contours, key=cv2.contourArea)
    contour_area = cv2.contourArea(largest_contour)

    if contour_area < h * w * min_area_ratio:
        return image, (0, 0, w, h), False

    # Get tight bounding box with minimal padding
    x, y, bw, bh = cv2.boundingRect(largest_contour)
    padding_x = max(3, int(bw * 0.01))
    padding_y = max(3, int(bh * 0.01))

    x = max(0, x - padding_x)
    y = max(0, y - padding_y)
    bw = min(w - x, bw + 2 * padding_x)
    bh = min(h - y, bh + 2 * padding_y)

    cropped_image = image[y:y+bh, x:x+bw].copy()
    return cropped_image, (x, y, bw, bh), True


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
# STRICT CROPPING EVALUATION
# =============================================================================

# Configuration
ENABLE_STRICT_CROPPING = True
WHITE_THRESHOLD = 210        # Higher = stricter (removes more background)
MIN_AREA_RATIO = 0.20        # Minimum 20% of image must be white
AGGRESSIVE_CLEAN = True      # Use aggressive morphological operations

print("=" * 80)
print("CORNELL EVALUATION WITH STRICT WHITE SURFACE CROPPING")
print("=" * 80)
print(f"Strict Cropping: {ENABLE_STRICT_CROPPING}")
print(f"White Threshold: {WHITE_THRESHOLD} (higher = stricter)")
print(f"Min Area Ratio: {MIN_AREA_RATIO*100:.0f}%")
print(f"Aggressive Clean: {AGGRESSIVE_CLEAN}")
print("=" * 80)
print("\nGoal: Remove ALL background, keep ONLY white surface + objects")
print("Pipeline: Strict Crop → Compute Depth → Mask → COG → Detect Grasps")
print("=" * 80)

# =============================================================================
# STEP 1: STRICT CROPPING + DEPTH COMPUTATION
# =============================================================================

cached_data_strict = []
crop_stats = {
    'total': 0,
    'cropped_success': 0,
    'no_white_surface': 0,
    'no_grasps_after_crop': 0,
    'used_original': 0,
    'avg_size_reduction': []
}

print("\nStep 1: Strict cropping and depth computation...")
start_time = time.time()

for sample in tqdm(samples, desc="Strict cropping + depth"):
    try:
        # Load RGB
        rgb = cv2.imread(sample["rgb_path"])
        if rgb is None:
            continue
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        crop_stats['total'] += 1
        original_size = rgb.shape[1] * rgb.shape[0]
        grasps_to_use = sample["grasps"]
        crop_info = {'cropped': False}

        # STRICT CROPPING
        if ENABLE_STRICT_CROPPING:
            cropped_rgb, crop_bbox, crop_success = crop_white_surface_strict(
                rgb,
                white_threshold=WHITE_THRESHOLD,
                min_area_ratio=MIN_AREA_RATIO,
                aggressive_clean=AGGRESSIVE_CLEAN
            )

            if crop_success:
                # Adjust ground truth
                adjusted_grasps = adjust_grasp_coordinates(sample["grasps"], crop_bbox)

                if len(adjusted_grasps) == 0:
                    # No grasps remain - use original
                    crop_stats['no_grasps_after_crop'] += 1
                    crop_stats['used_original'] += 1
                else:
                    # Success! Use cropped image
                    rgb = cropped_rgb
                    grasps_to_use = adjusted_grasps
                    crop_stats['cropped_success'] += 1

                    # Track size reduction
                    cropped_size = cropped_rgb.shape[1] * cropped_rgb.shape[0]
                    reduction = (1 - cropped_size / original_size) * 100
                    crop_stats['avg_size_reduction'].append(reduction)

                    crop_info = {
                        'cropped': True,
                        'crop_bbox': crop_bbox,
                        'original_size': (sample["rgb_path"], original_size),
                        'cropped_size': cropped_size,
                        'reduction_pct': reduction
                    }
            else:
                # No white surface detected
                crop_stats['no_white_surface'] += 1
                crop_stats['used_original'] += 1

        # Compute depth
        depth, _ = depth_model.predict(rgb)

        cached_data_strict.append({
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

# Print cropping statistics
print(f"\n✓ Processed {len(cached_data_strict)}/{len(samples)} samples")
if ENABLE_STRICT_CROPPING:
    print(f"\n{'='*80}")
    print("STRICT CROPPING STATISTICS")
    print(f"{'='*80}")
    print(f"Total samples:              {crop_stats['total']}")
    print(f"Successfully cropped:       {crop_stats['cropped_success']} ({crop_stats['cropped_success']/crop_stats['total']*100:.1f}%)")
    print(f"No white surface detected:  {crop_stats['no_white_surface']} ({crop_stats['no_white_surface']/crop_stats['total']*100:.1f}%)")
    print(f"No grasps after crop:       {crop_stats['no_grasps_after_crop']}")
    print(f"Used original (fallback):   {crop_stats['used_original']} ({crop_stats['used_original']/crop_stats['total']*100:.1f}%)")

    if crop_stats['avg_size_reduction']:
        avg_reduction = np.mean(crop_stats['avg_size_reduction'])
        print(f"\nAverage image size reduction: {avg_reduction:.1f}%")
        print(f"  (Higher = more background removed)")
    print(f"{'='*80}")

print(f"\nDepth computation time: {depth_time:.1f}s")

# =============================================================================
# STEP 2: GRASP DETECTION ON STRICTLY CROPPED DATA
# =============================================================================

print("\nStep 2: Running grasp detection on strictly cropped images...")

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

for data in tqdm(cached_data_strict, desc="Evaluating on cropped images"):
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
    print("RESULTS WITH STRICT WHITE SURFACE CROPPING")
    print("=" * 80)
    print(f"Images evaluated          : {total}")
    print(f"Images actually cropped   : {crop_stats['cropped_success']} ({crop_stats['cropped_success']/total*100:.1f}%)")
    print("-" * 80)
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
cached_data = cached_data_strict

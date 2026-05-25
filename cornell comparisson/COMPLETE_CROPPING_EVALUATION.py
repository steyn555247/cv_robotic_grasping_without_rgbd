import cv2
import numpy as np
from tqdm import tqdm
import time

# =============================================================================
# WHITE SURFACE CROPPING FUNCTION
# =============================================================================

def crop_white_surface(image, lower_white=(180, 180, 180), upper_white=(255, 255, 255),
                       min_area_ratio=0.15, show_debug=False):
    """
    Detect and crop the white surface from an image, removing background noise.

    Args:
        image: RGB image (numpy array)
        lower_white: Lower bound for white color in RGB
        upper_white: Upper bound for white color in RGB
        min_area_ratio: Minimum area ratio of contour to image
        show_debug: If True, print debug info

    Returns:
        cropped_image: Cropped RGB image containing only the white surface
        crop_bbox: (x, y, w, h) - bounding box of the crop
        crop_success: True if cropping succeeded, False otherwise
    """
    h, w = image.shape[:2]

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Create mask for white regions using both RGB and grayscale
    rgb_mask = cv2.inRange(image, np.array(lower_white), np.array(upper_white))
    gray_mask = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)[1]

    # Combine masks
    combined_mask = cv2.bitwise_or(rgb_mask, gray_mask)

    # Morphological operations to clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        if show_debug:
            print("  No white surface detected (no contours)")
        return image, (0, 0, w, h), False

    # Find the largest contour (assume it's the white surface)
    largest_contour = max(contours, key=cv2.contourArea)
    contour_area = cv2.contourArea(largest_contour)
    image_area = h * w

    # Check if contour is large enough
    if contour_area < image_area * min_area_ratio:
        if show_debug:
            print(f"  Surface too small: {contour_area/image_area:.1%} < {min_area_ratio:.1%}")
        return image, (0, 0, w, h), False

    # Get bounding box
    x, y, bw, bh = cv2.boundingRect(largest_contour)

    # Add small padding (2% of width/height)
    padding_x = max(5, int(bw * 0.02))
    padding_y = max(5, int(bh * 0.02))

    x = max(0, x - padding_x)
    y = max(0, y - padding_y)
    bw = min(w - x, bw + 2 * padding_x)
    bh = min(h - y, bh + 2 * padding_y)

    # Crop the image
    cropped_image = image[y:y+bh, x:x+bw].copy()

    if show_debug:
        print(f"  Cropped: {w}x{h} -> {bw}x{bh} ({contour_area/image_area:.1%} of area)")
        print(f"  Bbox: x={x}, y={y}, w={bw}, h={bh}")

    return cropped_image, (x, y, bw, bh), True


def adjust_grasp_coordinates(grasps, crop_bbox):
    """
    Adjust ground truth grasp coordinates after cropping.
    """
    x_offset, y_offset, crop_w, crop_h = crop_bbox
    adjusted_grasps = []

    for grasp in grasps:
        # Get center coordinates
        center_x, center_y = grasp['center']

        # Adjust center
        new_center_x = center_x - x_offset
        new_center_y = center_y - y_offset

        # Only keep grasps within cropped region
        if 0 <= new_center_x < crop_w and 0 <= new_center_y < crop_h:
            # Adjust all corners
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
# EVALUATION WITH WHITE SURFACE CROPPING
# =============================================================================

# Configuration
ENABLE_CROPPING = True  # Set to False to disable cropping
SHOW_CROP_DEBUG = False  # Set to True to see cropping details

print("=" * 80)
print("CORNELL EVALUATION WITH WHITE SURFACE CROPPING")
print("=" * 80)
print(f"Cropping Enabled: {ENABLE_CROPPING}")
print(f"Pipeline: Crop White Surface → Compute Depth → Mask → Find COG → Detect Grasps")
print("=" * 80)

# =============================================================================
# STEP 1: PRECOMPUTE DEPTH WITH WHITE SURFACE CROPPING
# =============================================================================

cached_data_cropped = []
crop_stats = {
    'total': 0,
    'cropped_success': 0,
    'no_surface': 0,
    'no_grasps_after_crop': 0,
    'used_original': 0
}

print("\nStep 1: Processing images with white surface cropping...")
start_time = time.time()

for sample in tqdm(samples, desc="Cropping and computing depth"):
    try:
        # Load RGB image
        rgb = cv2.imread(sample["rgb_path"])
        if rgb is None:
            continue
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        crop_stats['total'] += 1
        original_rgb = rgb.copy()
        grasps_to_use = sample["grasps"]
        crop_info = {'cropped': False}

        # CROP WHITE SURFACE
        if ENABLE_CROPPING:
            cropped_rgb, crop_bbox, crop_success = crop_white_surface(
                rgb,
                lower_white=(180, 180, 180),
                upper_white=(255, 255, 255),
                min_area_ratio=0.15,
                show_debug=SHOW_CROP_DEBUG
            )

            if crop_success:
                # Adjust ground truth grasps
                adjusted_grasps = adjust_grasp_coordinates(sample["grasps"], crop_bbox)

                if len(adjusted_grasps) == 0:
                    # No grasps in cropped region - use original
                    crop_stats['no_grasps_after_crop'] += 1
                    crop_stats['used_original'] += 1
                    # Use original image
                else:
                    # Successfully cropped with valid grasps
                    rgb = cropped_rgb
                    grasps_to_use = adjusted_grasps
                    crop_stats['cropped_success'] += 1
                    crop_info = {
                        'cropped': True,
                        'crop_bbox': crop_bbox,
                        'original_size': (original_rgb.shape[1], original_rgb.shape[0]),
                        'cropped_size': (cropped_rgb.shape[1], cropped_rgb.shape[0])
                    }
            else:
                # White surface detection failed
                crop_stats['no_surface'] += 1
                crop_stats['used_original'] += 1

        # Compute depth on (possibly cropped) RGB
        depth, _ = depth_model.predict(rgb)

        cached_data_cropped.append({
            "rgb": rgb,
            "depth": depth,
            "grasps": grasps_to_use,
            "crop_info": crop_info,
            "original_path": sample["rgb_path"]
        })

    except Exception as e:
        print(f"\nError processing {sample.get('rgb_path', 'unknown')}: {e}")
        continue

depth_time = time.time() - start_time

print(f"\n✓ Processed {len(cached_data_cropped)} / {len(samples)} samples")
if ENABLE_CROPPING:
    print(f"\nCropping Statistics:")
    print(f"  Total samples:           {crop_stats['total']}")
    print(f"  Successfully cropped:    {crop_stats['cropped_success']} ({crop_stats['cropped_success']/crop_stats['total']*100:.1f}%)")
    print(f"  No surface detected:     {crop_stats['no_surface']} ({crop_stats['no_surface']/crop_stats['total']*100:.1f}%)")
    print(f"  No grasps after crop:    {crop_stats['no_grasps_after_crop']}")
    print(f"  Used original (total):   {crop_stats['used_original']} ({crop_stats['used_original']/crop_stats['total']*100:.1f}%)")
print(f"\nDepth computation time: {depth_time:.1f}s")

# =============================================================================
# STEP 2: RUN GRASP DETECTION ON CROPPED DATA
# =============================================================================

print("\nStep 2: Running grasp detection on processed images...")

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

for data in tqdm(cached_data_cropped, desc="Evaluating grasp detection"):
    try:
        # Run grasp detection on (possibly cropped) image
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

        # Convert to GraspCandidate format
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

        # Store evaluation result for visualization
        data["eval_result"] = eval_result
        data["predictions"] = predictions
        data["detection_info"] = info

    except Exception as e:
        print(f"\nError evaluating sample, skipping: {e}")
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
    print("RESULTS WITH WHITE SURFACE CROPPING")
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

    if ENABLE_CROPPING:
        print(f"\nCropping Impact:")
        print(f"  Images actually cropped: {crop_stats['cropped_success']}/{total} ({crop_stats['cropped_success']/total*100:.1f}%)")
        print("=" * 80)

# Store for visualization
cached_data = cached_data_cropped  # This is needed for visualization scripts

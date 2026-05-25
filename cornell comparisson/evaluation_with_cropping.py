import cv2
import numpy as np
from tqdm import tqdm
import time

# =============================================================================
# WHITE SURFACE CROPPING FUNCTIONS
# =============================================================================

def crop_white_surface(image, lower_white=(200, 200, 200), upper_white=(255, 255, 255),
                       min_area_ratio=0.1):
    """
    Detect and crop the white surface from an image, removing background noise.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Create mask for white regions
    rgb_mask = cv2.inRange(image, np.array(lower_white), np.array(upper_white))
    gray_mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)[1]
    combined_mask = cv2.bitwise_or(rgb_mask, gray_mask)

    # Morphological operations to clean up
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=2)

    # Find largest contour
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        return image, (0, 0, image.shape[1], image.shape[0])

    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Check minimum area
    image_area = image.shape[0] * image.shape[1]
    contour_area = cv2.contourArea(largest_contour)

    if contour_area < image_area * min_area_ratio:
        return image, (0, 0, image.shape[1], image.shape[0])

    # Add small padding
    padding_x = int(w * 0.02)
    padding_y = int(h * 0.02)
    x = max(0, x - padding_x)
    y = max(0, y - padding_y)
    w = min(image.shape[1] - x, w + 2 * padding_x)
    h = min(image.shape[0] - y, h + 2 * padding_y)

    cropped_image = image[y:y+h, x:x+w]
    return cropped_image, (x, y, w, h)


def adjust_grasp_coordinates(grasps, crop_bbox):
    """
    Adjust ground truth grasp coordinates after cropping.
    """
    x_offset, y_offset, crop_w, crop_h = crop_bbox
    adjusted_grasps = []

    for grasp in grasps:
        # Adjust coordinates
        new_x = grasp.x - x_offset
        new_y = grasp.y - y_offset

        # Only keep grasps that are within the cropped region
        if 0 <= new_x < crop_w and 0 <= new_y < crop_h:
            # Create adjusted grasp
            adjusted_grasp = type(grasp)(
                x=new_x,
                y=new_y,
                angle=grasp.angle,
                width=grasp.width,
                height=grasp.height if hasattr(grasp, 'height') else 20.0
            )
            adjusted_grasps.append(adjusted_grasp)

    return adjusted_grasps


# =============================================================================
# EVALUATION WITH WHITE SURFACE CROPPING
# =============================================================================

print("=" * 80)
print("FULL CORNELL EVALUATION WITH WHITE SURFACE CROPPING")
print("=" * 80)

# Step 1: Precompute depth for all samples WITH cropping
cached_data_cropped = []
skipped_no_surface = 0
skipped_no_grasps = 0
start = time.time()

for sample in tqdm(samples, desc="Computing depth maps with cropping"):
    try:
        rgb = cv2.imread(sample["rgb_path"])
        if rgb is None:
            continue
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        # CROP THE WHITE SURFACE
        cropped_rgb, crop_bbox = crop_white_surface(rgb)

        # Skip if cropping failed (no surface detected)
        if crop_bbox == (0, 0, rgb.shape[1], rgb.shape[0]) and cropped_rgb.shape == rgb.shape:
            skipped_no_surface += 1
            continue

        # Adjust ground truth grasps
        adjusted_grasps = adjust_grasp_coordinates(sample["grasps"], crop_bbox)

        # Skip if no grasps remain in the cropped region
        if len(adjusted_grasps) == 0:
            skipped_no_grasps += 1
            continue

        # Compute depth on CROPPED image
        depth, _ = depth_model.predict(cropped_rgb)

        cached_data_cropped.append({
            "rgb": cropped_rgb,
            "depth": depth,
            "grasps": adjusted_grasps,
            "original_path": sample["rgb_path"],
            "crop_bbox": crop_bbox
        })
    except Exception as e:
        print(f"\nError processing {sample.get('rgb_path', 'unknown')}: {e}")
        continue

print(f"✓ Cached {len(cached_data_cropped)} / {len(samples)} samples")
print(f"  Skipped (no surface): {skipped_no_surface}")
print(f"  Skipped (no grasps in crop): {skipped_no_grasps}")
print(f"Depth precomputation time: {time.time() - start:.1f}s")

# Step 2: Run detector on cropped samples
detector_cropped = GraspDetector(
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

start = time.time()

for data in tqdm(cached_data_cropped, desc="Evaluating with cropped images"):
    try:
        grasps, _ = detector_cropped.process(
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
        continue

elapsed = time.time() - start

# Print results
if total == 0:
    print("No samples were successfully evaluated.")
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
    print(f"Total eval time       : {elapsed:.1f}s")
    print("=" * 80)

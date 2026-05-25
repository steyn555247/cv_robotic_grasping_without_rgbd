import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

# =============================================================================
# STRICT WHITE SURFACE CROPPING - ONLY WHITE + OBJECTS
# =============================================================================

def crop_white_surface_strict(image, white_threshold=200, min_area_ratio=0.20,
                               aggressive_clean=True, show_debug=False):
    """
    Strictly crop to only the white surface and objects on it.
    Removes ALL background, desk edges, shadows, etc.

    Args:
        image: RGB image (numpy array)
        white_threshold: Minimum brightness for white (200-220 recommended)
        min_area_ratio: Minimum area ratio (0.2 = 20% of image)
        aggressive_clean: If True, uses more aggressive morphological operations
        show_debug: If True, print debug info

    Returns:
        cropped_image: Cropped RGB image containing ONLY white surface
        crop_bbox: (x, y, w, h) - bounding box of the crop
        crop_success: True if cropping succeeded
    """
    h, w = image.shape[:2]

    if show_debug:
        print(f"  Original: {w}x{h}")

    # STEP 1: Create strict white mask
    # Use all 3 channels - must be white in ALL channels
    r, g, b = image[:,:,0], image[:,:,1], image[:,:,2]

    # All channels must be above threshold (strict white)
    white_mask = ((r >= white_threshold) &
                  (g >= white_threshold) &
                  (b >= white_threshold)).astype(np.uint8) * 255

    # Also check grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray_mask = (gray >= white_threshold).astype(np.uint8) * 255

    # Both must agree it's white
    combined_mask = cv2.bitwise_and(white_mask, gray_mask)

    if show_debug:
        print(f"  White pixels: {np.count_nonzero(combined_mask)}/{h*w} ({np.count_nonzero(combined_mask)/(h*w)*100:.1f}%)")

    # STEP 2: Aggressive morphological cleaning
    if aggressive_clean:
        # Large kernel to remove small noise
        kernel_large = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
        # Small kernel for fine details
        kernel_small = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

        # Close large gaps first
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_large, iterations=4)
        # Remove small noise
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel_small, iterations=3)
        # Fill any remaining holes
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel_large, iterations=2)
    else:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=2)

    # STEP 3: Find the largest white region (the table surface)
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        if show_debug:
            print(f"  ❌ No white regions found")
        return image, (0, 0, w, h), False

    # Get largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    contour_area = cv2.contourArea(largest_contour)
    image_area = h * w

    if show_debug:
        print(f"  Largest region: {contour_area} ({contour_area/image_area*100:.1f}%)")

    # Check minimum area
    if contour_area < image_area * min_area_ratio:
        if show_debug:
            print(f"  ❌ Region too small: {contour_area/image_area*100:.1f}% < {min_area_ratio*100:.1f}%")
        return image, (0, 0, w, h), False

    # STEP 4: Get tight bounding box - NO padding (strict crop)
    x, y, bw, bh = cv2.boundingRect(largest_contour)

    # Optional: Very minimal padding (just 1% to avoid edge artifacts)
    padding_x = max(3, int(bw * 0.01))
    padding_y = max(3, int(bh * 0.01))

    x = max(0, x - padding_x)
    y = max(0, y - padding_y)
    bw = min(w - x, bw + 2 * padding_x)
    bh = min(h - y, bh + 2 * padding_y)

    # STEP 5: Crop strictly
    cropped_image = image[y:y+bh, x:x+bw].copy()

    if show_debug:
        print(f"  ✓ Cropped: {w}x{h} → {bw}x{bh}")
        print(f"  Reduction: {(1 - (bw*bh)/(w*h))*100:.1f}%")

    return cropped_image, (x, y, bw, bh), True


def adjust_grasp_coordinates(grasps, crop_bbox):
    """Adjust ground truth grasp coordinates after cropping."""
    x_offset, y_offset, crop_w, crop_h = crop_bbox
    adjusted_grasps = []

    for grasp in grasps:
        center_x, center_y = grasp['center']
        new_center_x = center_x - x_offset
        new_center_y = center_y - y_offset

        # Keep grasps within cropped region
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
# TEST STRICT CROPPING ON SINGLE IMAGE
# =============================================================================

print("=" * 80)
print("TESTING STRICT WHITE SURFACE CROPPING")
print("=" * 80)
print("Goal: Remove ALL background, keep ONLY white surface + objects")
print("=" * 80)

# Test on first few samples
test_indices = [0, 1, 2, 100, 200]

for test_idx in test_indices:
    if test_idx >= len(samples):
        continue

    test_sample = samples[test_idx]
    print(f"\n{'='*80}")
    print(f"Sample {test_idx + 1}: {test_sample['rgb_path']}")
    print('-' * 80)

    # Load image
    test_img = cv2.imread(test_sample["rgb_path"])
    test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)

    # Test different thresholds
    thresholds = [200, 210, 220]

    fig, axes = plt.subplots(2, len(thresholds) + 1, figsize=(20, 10))

    # Show original
    axes[0, 0].imshow(test_img)
    axes[0, 0].set_title('ORIGINAL', fontsize=12, fontweight='bold')
    axes[0, 0].axis('off')

    axes[1, 0].axis('off')
    info_text = f"Sample {test_idx + 1}\n\n"
    info_text += f"Size: {test_img.shape[1]}x{test_img.shape[0]}\n"
    info_text += f"Grasps: {len(test_sample['grasps'])}\n\n"
    info_text += "Testing different\nwhite thresholds:"
    axes[1, 0].text(0.2, 0.5, info_text, fontsize=11, family='monospace',
                   verticalalignment='center', transform=axes[1, 0].transAxes)

    # Test each threshold
    for i, threshold in enumerate(thresholds):
        print(f"\nThreshold {threshold}:")
        cropped, bbox, success = crop_white_surface_strict(
            test_img,
            white_threshold=threshold,
            min_area_ratio=0.20,
            aggressive_clean=True,
            show_debug=True
        )

        # Show cropped result
        if success:
            axes[0, i+1].imshow(cropped)
            x, y, bw, bh = bbox
            title = f'Threshold={threshold}\n✓ {bw}x{bh}'
            color = 'green'

            # Draw bbox on original
            axes[1, i+1].imshow(test_img)
            rect = plt.Rectangle((x, y), bw, bh, fill=False, edgecolor='lime', linewidth=3)
            axes[1, i+1].add_patch(rect)
        else:
            axes[0, i+1].imshow(test_img)
            title = f'Threshold={threshold}\n✗ FAILED'
            color = 'red'

            axes[1, i+1].imshow(test_img)

        axes[0, i+1].set_title(title, fontsize=11, fontweight='bold', color=color)
        axes[0, i+1].axis('off')
        axes[1, i+1].set_title('Crop Region', fontsize=10)
        axes[1, i+1].axis('off')

    plt.suptitle(f"Strict Cropping Test - Sample {test_idx + 1}",
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

print("\n" + "=" * 80)
print("RECOMMENDATION:")
print("  Choose the threshold that:")
print("  1. Removes ALL background/desk edges")
print("  2. Keeps the complete white surface")
print("  3. Doesn't cut into the objects")
print("\n  Typical best value: 205-215")
print("=" * 80)

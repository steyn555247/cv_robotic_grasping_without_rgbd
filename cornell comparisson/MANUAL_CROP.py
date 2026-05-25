import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

# =============================================================================
# MANUAL CROP - DEFINE YOUR OWN CROP REGION
# =============================================================================

def manual_crop(image, crop_bbox):
    """
    Manually crop image to specified region.

    Args:
        image: RGB image
        crop_bbox: (x, y, width, height) - the region to keep

    Returns:
        cropped_image: Cropped image
        crop_bbox: Same bbox for consistency
        success: Always True (manual crop always succeeds)
    """
    x, y, w, h = crop_bbox

    # Ensure crop is within image bounds
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
# FIND THE RIGHT CROP REGION INTERACTIVELY
# =============================================================================

print("=" * 80)
print("MANUAL CROP - FIND THE RIGHT REGION")
print("=" * 80)
print("\nFirst, let's look at a few sample images to determine the crop region")
print("=" * 80)

# Show several samples to find the best crop region
test_indices = [0, 1, 2, 50, 100, 200, 400]

for idx in test_indices[:3]:  # Show first 3
    if idx >= len(samples):
        continue

    sample = samples[idx]
    img = cv2.imread(sample["rgb_path"])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    print(f"\nSample {idx}: {sample['rgb_path']}")
    print(f"Image size: {img.shape[1]} x {img.shape[0]} (width x height)")

    # Show image with grid overlay to help identify crop region
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    ax.imshow(img)
    ax.set_title(f'Sample {idx} - Image size: {img.shape[1]}x{img.shape[0]}',
                 fontsize=14, fontweight='bold')

    # Add grid lines to help identify coordinates
    ax.grid(True, alpha=0.3, color='yellow', linewidth=1)

    # Add coordinate markers every 50 pixels
    for x in range(0, img.shape[1], 50):
        ax.axvline(x, color='yellow', alpha=0.2, linewidth=0.5)
        if x % 100 == 0:
            ax.text(x, 10, str(x), color='yellow', fontsize=8,
                   bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

    for y in range(0, img.shape[0], 50):
        ax.axhline(y, color='yellow', alpha=0.2, linewidth=0.5)
        if y % 100 == 0:
            ax.text(10, y, str(y), color='yellow', fontsize=8,
                   bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

    ax.axis('on')
    plt.tight_layout()
    plt.show()

print("\n" + "=" * 80)
print("INSTRUCTIONS:")
print("=" * 80)
print("Look at the images above and identify the crop region that:")
print("  1. Includes the ENTIRE white table surface")
print("  2. Excludes brown background on the sides")
print("  3. Excludes any desk edges or non-white areas")
print("\nNote the coordinates where:")
print("  - X starts (left edge of white table)")
print("  - Y starts (top edge of white table)")
print("  - X ends (right edge of white table)")
print("  - Y ends (bottom edge of white table)")
print("=" * 80)

# =============================================================================
# DEFINE YOUR CROP REGION HERE
# =============================================================================

# MODIFY THESE VALUES based on what you see in the images above
# Format: (x, y, width, height)
#
# Example: If white table starts at x=50, y=30 and ends at x=550, y=450:
#   x = 50
#   y = 30
#   width = 550 - 50 = 500
#   height = 450 - 30 = 420
#   MANUAL_CROP_BBOX = (50, 30, 500, 420)

MANUAL_CROP_BBOX = (50, 30, 540, 420)  # ADJUST THESE VALUES!

print(f"\nCurrent crop region: x={MANUAL_CROP_BBOX[0]}, y={MANUAL_CROP_BBOX[1]}, "
      f"width={MANUAL_CROP_BBOX[2]}, height={MANUAL_CROP_BBOX[3]}")
print(f"This means: crop from ({MANUAL_CROP_BBOX[0]}, {MANUAL_CROP_BBOX[1]}) to "
      f"({MANUAL_CROP_BBOX[0]+MANUAL_CROP_BBOX[2]}, {MANUAL_CROP_BBOX[1]+MANUAL_CROP_BBOX[3]})")

# =============================================================================
# TEST THE CROP REGION
# =============================================================================

print("\n" + "=" * 80)
print("TESTING YOUR CROP REGION")
print("=" * 80)

for idx in test_indices[:5]:  # Test on 5 samples
    if idx >= len(samples):
        continue

    sample = samples[idx]
    img = cv2.imread(sample["rgb_path"])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Apply manual crop
    cropped, bbox, success = manual_crop(img, MANUAL_CROP_BBOX)

    # Show before/after
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Original with crop box
    axes[0].imshow(img)
    x, y, w, h = bbox
    rect = plt.Rectangle((x, y), w, h, fill=False, edgecolor='lime', linewidth=3)
    axes[0].add_patch(rect)
    axes[0].set_title(f'Original ({img.shape[1]}x{img.shape[0]})\nGreen box = crop region',
                     fontsize=12, fontweight='bold')
    axes[0].axis('off')

    # Cropped result
    axes[1].imshow(cropped)
    axes[1].set_title(f'Cropped Result ({cropped.shape[1]}x{cropped.shape[0]})',
                     fontsize=12, fontweight='bold', color='green')
    axes[1].axis('off')

    # Info
    axes[2].axis('off')
    reduction = (1 - (cropped.shape[0]*cropped.shape[1])/(img.shape[0]*img.shape[1])) * 100
    info_text = f"""
Sample {idx}

Original: {img.shape[1]} x {img.shape[0]}
Cropped:  {cropped.shape[1]} x {cropped.shape[0]}

Crop Box:
  x: {x}
  y: {y}
  width: {w}
  height: {h}

Size Reduction: {reduction:.1f}%

Grasps: {len(sample['grasps'])}

Does this look good?
- White table fully visible?
- No brown background?
- Objects not cut off?
"""
    axes[2].text(0.1, 0.5, info_text, fontsize=11, family='monospace',
                verticalalignment='center', transform=axes[2].transAxes,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.suptitle(f"Manual Crop Test - Sample {idx}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.show()

print("\n" + "=" * 80)
print("If the crop looks good, use MANUAL_CROP_EVALUATION.py")
print("If not, adjust MANUAL_CROP_BBOX values above and re-run this cell")
print("=" * 80)

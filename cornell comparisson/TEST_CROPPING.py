import cv2
import numpy as np
import matplotlib.pyplot as plt

# =============================================================================
# TEST WHITE SURFACE CROPPING ON A SINGLE IMAGE
# =============================================================================

def crop_white_surface_test(image, lower_white=(180, 180, 180), upper_white=(255, 255, 255),
                            min_area_ratio=0.15):
    """Test version with detailed output."""
    h, w = image.shape[:2]
    print(f"Original image size: {w} x {h}")

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Create mask for white regions
    rgb_mask = cv2.inRange(image, np.array(lower_white), np.array(upper_white))
    gray_mask = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)[1]
    combined_mask = cv2.bitwise_or(rgb_mask, gray_mask)

    print(f"Mask pixels: {np.count_nonzero(combined_mask)} / {h*w} ({np.count_nonzero(combined_mask)/(h*w)*100:.1f}%)")

    # Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"Found {len(contours)} contours")

    if len(contours) == 0:
        print("❌ No contours found!")
        return image, (0, 0, w, h), False, combined_mask

    # Get largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    contour_area = cv2.contourArea(largest_contour)
    image_area = h * w

    print(f"Largest contour area: {contour_area} ({contour_area/image_area*100:.1f}% of image)")

    if contour_area < image_area * min_area_ratio:
        print(f"❌ Contour too small! {contour_area/image_area*100:.1f}% < {min_area_ratio*100:.1f}%")
        return image, (0, 0, w, h), False, combined_mask

    # Get bounding box
    x, y, bw, bh = cv2.boundingRect(largest_contour)
    print(f"Bounding box: x={x}, y={y}, w={bw}, h={bh}")

    # Add padding
    padding_x = max(5, int(bw * 0.02))
    padding_y = max(5, int(bh * 0.02))

    x = max(0, x - padding_x)
    y = max(0, y - padding_y)
    bw = min(w - x, bw + 2 * padding_x)
    bh = min(h - y, bh + 2 * padding_y)

    print(f"With padding: x={x}, y={y}, w={bw}, h={bh}")

    # Crop
    cropped = image[y:y+bh, x:x+bw].copy()
    print(f"✓ Cropped to: {bw} x {bh}")
    print(f"Size reduction: {(1 - (bw*bh)/(w*h))*100:.1f}%")

    return cropped, (x, y, bw, bh), True, combined_mask


# Test on first sample
print("=" * 80)
print("TESTING WHITE SURFACE CROPPING")
print("=" * 80)

test_sample = samples[0]
print(f"\nTesting on: {test_sample['rgb_path']}")
print("-" * 80)

# Load image
test_img = cv2.imread(test_sample["rgb_path"])
test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)

# Test cropping
cropped, bbox, success, mask = crop_white_surface_test(test_img)

print("-" * 80)
print(f"Cropping {'SUCCEEDED ✓' if success else 'FAILED ✗'}")
print("=" * 80)

# Visualize
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Original
axes[0, 0].imshow(test_img)
axes[0, 0].set_title('Original Image', fontsize=14, fontweight='bold')
if success:
    x, y, w, h = bbox
    rect = plt.Rectangle((x, y), w, h, fill=False, edgecolor='lime', linewidth=3)
    axes[0, 0].add_patch(rect)
    axes[0, 0].text(x+10, y+30, 'Crop Region', color='lime', fontsize=12,
                   bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
axes[0, 0].axis('off')

# Mask
axes[0, 1].imshow(mask, cmap='gray')
axes[0, 1].set_title('White Surface Mask', fontsize=14, fontweight='bold')
axes[0, 1].axis('off')

# Cropped result
if success:
    axes[1, 0].imshow(cropped)
    axes[1, 0].set_title(f'Cropped Result ({cropped.shape[1]}x{cropped.shape[0]})',
                        fontsize=14, fontweight='bold', color='green')
else:
    axes[1, 0].imshow(test_img)
    axes[1, 0].set_title('Cropping Failed - Showing Original',
                        fontsize=14, fontweight='bold', color='red')
axes[1, 0].axis('off')

# Comparison
axes[1, 1].axis('off')
info_text = f"""
CROPPING TEST RESULT
{'='*40}

Status: {'✓ SUCCESS' if success else '✗ FAILED'}

Original Size: {test_img.shape[1]} x {test_img.shape[0]}
{'Cropped Size: ' + str(cropped.shape[1]) + ' x ' + str(cropped.shape[0]) if success else 'No crop applied'}

{f'Crop BBox: ({bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]})' if success else ''}
{f'Size Reduction: {(1 - (bbox[2]*bbox[3])/(test_img.shape[1]*test_img.shape[0]))*100:.1f}%' if success else ''}

Ground Truth Grasps: {len(test_sample['grasps'])}

{'='*40}
"""

axes[1, 1].text(0.1, 0.5, info_text, fontsize=12, family='monospace',
               verticalalignment='center', transform=axes[1, 1].transAxes,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.tight_layout()
plt.show()

print("\nTo adjust cropping sensitivity:")
print("  - Lower min_area_ratio (e.g., 0.10) to be more aggressive")
print("  - Adjust lower_white threshold (e.g., (170, 170, 170)) to detect less-bright whites")

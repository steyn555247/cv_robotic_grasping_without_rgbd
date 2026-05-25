import cv2
import numpy as np
import matplotlib.pyplot as plt

def crop_white_surface(image, lower_white=(200, 200, 200), upper_white=(255, 255, 255),
                       min_area_ratio=0.1, show_debug=False):
    """
    Detect and crop the white surface from an image, removing background noise.

    Args:
        image: RGB image (numpy array)
        lower_white: Lower bound for white color in RGB (default: (200, 200, 200))
        upper_white: Upper bound for white color in RGB (default: (255, 255, 255))
        min_area_ratio: Minimum area ratio of contour to image (default: 0.1)
        show_debug: If True, display debug visualization

    Returns:
        cropped_image: Cropped RGB image containing only the white surface
        crop_bbox: (x, y, w, h) - bounding box of the crop
        mask: Binary mask of the white surface
    """
    # Convert to grayscale for better thresholding
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    # Create mask for white regions
    # Use both RGB thresholding and grayscale
    rgb_mask = cv2.inRange(image, np.array(lower_white), np.array(upper_white))
    gray_mask = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)[1]

    # Combine masks (white should be bright in both)
    combined_mask = cv2.bitwise_or(rgb_mask, gray_mask)

    # Morphological operations to clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    combined_mask = cv2.morphologyEx(combined_mask, cv2.MORPH_OPEN, kernel, iterations=2)

    # Fill holes in the mask
    contours, _ = cv2.findContours(combined_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        # No white surface found, return original image
        print("Warning: No white surface detected, returning original image")
        return image, (0, 0, image.shape[1], image.shape[0]), combined_mask

    # Find the largest contour (assuming it's the white surface)
    largest_contour = max(contours, key=cv2.contourArea)

    # Get bounding box
    x, y, w, h = cv2.boundingRect(largest_contour)

    # Check if the contour is large enough
    image_area = image.shape[0] * image.shape[1]
    contour_area = cv2.contourArea(largest_contour)

    if contour_area < image_area * min_area_ratio:
        print(f"Warning: Detected surface too small ({contour_area/image_area:.1%} of image), returning original")
        return image, (0, 0, image.shape[1], image.shape[0]), combined_mask

    # Add some padding (5% of width/height)
    padding_x = int(w * 0.02)
    padding_y = int(h * 0.02)

    x = max(0, x - padding_x)
    y = max(0, y - padding_y)
    w = min(image.shape[1] - x, w + 2 * padding_x)
    h = min(image.shape[0] - y, h + 2 * padding_y)

    # Crop the image
    cropped_image = image[y:y+h, x:x+w]

    if show_debug:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        axes[0, 0].imshow(image)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')

        axes[0, 1].imshow(combined_mask, cmap='gray')
        axes[0, 1].set_title('White Surface Mask')
        axes[0, 1].axis('off')

        # Draw bounding box on original
        img_with_bbox = image.copy()
        cv2.rectangle(img_with_bbox, (x, y), (x+w, y+h), (0, 255, 0), 3)
        axes[1, 0].imshow(img_with_bbox)
        axes[1, 0].set_title('Detected Crop Region')
        axes[1, 0].axis('off')

        axes[1, 1].imshow(cropped_image)
        axes[1, 1].set_title('Cropped Result')
        axes[1, 1].axis('off')

        plt.tight_layout()
        plt.show()

    return cropped_image, (x, y, w, h), combined_mask


def adjust_grasp_coordinates(grasps, crop_bbox):
    """
    Adjust grasp coordinates after cropping.

    Args:
        grasps: List of grasp rectangles (each has x, y coordinates)
        crop_bbox: (x, y, w, h) - the crop bounding box

    Returns:
        adjusted_grasps: List of grasps with coordinates adjusted for the cropped image
    """
    x_offset, y_offset, _, _ = crop_bbox

    adjusted_grasps = []
    for grasp in grasps:
        # Create a copy of the grasp
        adjusted_grasp = type(grasp).__new__(type(grasp))

        # Copy all attributes
        for attr in dir(grasp):
            if not attr.startswith('_') and not callable(getattr(grasp, attr)):
                setattr(adjusted_grasp, attr, getattr(grasp, attr))

        # Adjust x and y coordinates
        adjusted_grasp.x = grasp.x - x_offset
        adjusted_grasp.y = grasp.y - y_offset

        # Keep the grasp only if it's still within the cropped region
        if adjusted_grasp.x >= 0 and adjusted_grasp.y >= 0:
            adjusted_grasps.append(adjusted_grasp)

    return adjusted_grasps


# Test the cropping function on a sample image
if __name__ == "__main__":
    # Test on first sample
    if len(samples) > 0:
        test_sample = samples[0]
        test_img = cv2.imread(test_sample["rgb_path"])
        test_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)

        print("Testing white surface detection...")
        cropped, bbox, mask = crop_white_surface(test_img, show_debug=True)
        print(f"Original size: {test_img.shape[:2]}")
        print(f"Cropped size: {cropped.shape[:2]}")
        print(f"Crop bbox: x={bbox[0]}, y={bbox[1]}, w={bbox[2]}, h={bbox[3]}")

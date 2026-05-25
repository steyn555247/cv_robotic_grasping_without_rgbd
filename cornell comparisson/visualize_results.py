import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import random

# =============================================================================
# VISUALIZATION: SHOW GRASP DETECTION RESULTS
# =============================================================================

def draw_grasp_rectangle(ax, grasp, color='green', label=None, linewidth=2):
    """
    Draw a grasp rectangle on the given axis.

    Args:
        ax: Matplotlib axis
        grasp: Either a Candidate object or dict with corners
        color: Color for the rectangle
        label: Optional label
        linewidth: Line width
    """
    if hasattr(grasp, 'get_corners'):
        # It's a Candidate object
        corners = grasp.get_corners()
    elif isinstance(grasp, dict) and 'corners' in grasp:
        # It's a dictionary
        corners = grasp['corners']
    else:
        return

    polygon = Polygon(corners, fill=False, edgecolor=color, linewidth=linewidth, label=label)
    ax.add_patch(polygon)

    # Draw center point
    if hasattr(grasp, 'x'):
        center = (grasp.x, grasp.y)
    elif 'center' in grasp:
        center = grasp['center']
    else:
        center = np.mean(corners, axis=0)

    ax.plot(center[0], center[1], 'o', color=color, markersize=5)


def visualize_single_result(rgb, depth, predictions, ground_truths,
                            eval_result, crop_info=None, save_path=None):
    """
    Visualize a single grasp detection result.

    Args:
        rgb: RGB image
        depth: Depth map
        predictions: List of predicted grasps (Candidate objects)
        ground_truths: List of ground truth grasps (dicts)
        eval_result: Evaluation metrics dict
        crop_info: Optional dict with cropping information
        save_path: Optional path to save the figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1. Original RGB with ground truth
    axes[0, 0].imshow(rgb)
    axes[0, 0].set_title('RGB Image with Ground Truth Grasps', fontsize=14, fontweight='bold')
    for i, gt in enumerate(ground_truths):
        draw_grasp_rectangle(axes[0, 0], gt, color='blue',
                           label='Ground Truth' if i == 0 else None, linewidth=2)
    axes[0, 0].legend(loc='upper right')
    axes[0, 0].axis('off')

    # 2. Depth map
    axes[0, 1].imshow(depth, cmap='viridis')
    axes[0, 1].set_title('Depth Map', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    cbar = plt.colorbar(axes[0, 1].images[0], ax=axes[0, 1], fraction=0.046, pad=0.04)
    cbar.set_label('Depth', rotation=270, labelpad=15)

    # 3. Predictions vs Ground Truth
    axes[1, 0].imshow(rgb)
    axes[1, 0].set_title('Predicted Grasps vs Ground Truth', fontsize=14, fontweight='bold')

    # Draw ground truth in blue
    for i, gt in enumerate(ground_truths):
        draw_grasp_rectangle(axes[1, 0], gt, color='blue',
                           label='Ground Truth' if i == 0 else None, linewidth=2)

    # Draw predictions in green (success) or red (failure)
    for i, pred in enumerate(predictions):
        if i == 0:  # Top-1 prediction
            color = 'lime' if eval_result.get('top1_success', False) else 'red'
            label = f"Top-1 ({'Success' if eval_result.get('top1_success', False) else 'Fail'})"
            draw_grasp_rectangle(axes[1, 0], pred, color=color, label=label, linewidth=3)
        else:
            color = 'green'
            label = 'Other Predictions' if i == 1 else None
            draw_grasp_rectangle(axes[1, 0], pred, color=color, label=label, linewidth=2)

    axes[1, 0].legend(loc='upper right')
    axes[1, 0].axis('off')

    # 4. Metrics and Info
    axes[1, 1].axis('off')

    info_text = "EVALUATION METRICS\n" + "="*40 + "\n\n"
    info_text += f"Top-1 Success:     {'✓ YES' if eval_result.get('top1_success', False) else '✗ NO'}\n"
    info_text += f"Top-5 Success:     {'✓ YES' if eval_result.get('top5_success', False) else '✗ NO'}\n"
    info_text += f"Any Success:       {'✓ YES' if eval_result.get('any_success', False) else '✗ NO'}\n\n"
    info_text += f"Average IoU:       {eval_result.get('avg_iou', 0.0):.4f}\n"
    info_text += f"Angle Diff:        {eval_result.get('avg_angle_diff', 0.0):.2f}°\n\n"
    info_text += f"Predictions:       {len(predictions)}\n"
    info_text += f"Ground Truths:     {len(ground_truths)}\n\n"

    if crop_info:
        info_text += "CROPPING INFO\n" + "="*40 + "\n\n"
        info_text += f"Cropped:           {'✓ YES' if crop_info.get('cropped', False) else '✗ NO'}\n"
        if crop_info.get('cropped', False):
            bbox = crop_info.get('crop_bbox', (0, 0, 0, 0))
            info_text += f"Crop BBox:         ({bbox[0]}, {bbox[1]}, {bbox[2]}, {bbox[3]})\n"
            orig_size = crop_info.get('original_size', (0, 0))
            crop_size = crop_info.get('cropped_size', (0, 0))
            info_text += f"Original Size:     {orig_size[0]} x {orig_size[1]}\n"
            info_text += f"Cropped Size:      {crop_size[0]} x {crop_size[1]}\n"

    info_text += "\n" + "="*40 + "\n"
    info_text += f"Image Size:        {rgb.shape[1]} x {rgb.shape[0]}\n"

    axes[1, 1].text(0.1, 0.5, info_text, fontsize=12, family='monospace',
                   verticalalignment='center', transform=axes[1, 1].transAxes)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved visualization to {save_path}")

    plt.show()


def visualize_batch_results(cached_data, detector, evaluator, num_samples=5,
                            successful_only=False, failed_only=False, random_selection=True):
    """
    Visualize multiple grasp detection results.

    Args:
        cached_data: List of cached samples with rgb, depth, grasps
        detector: GraspDetector instance
        evaluator: GraspEvaluator instance
        num_samples: Number of samples to visualize
        successful_only: Only show successful detections
        failed_only: Only show failed detections
        random_selection: Randomly select samples (otherwise sequential)
    """
    print("=" * 80)
    print(f"VISUALIZING {num_samples} GRASP DETECTION RESULTS")
    print("=" * 80)

    # Select samples to visualize
    if random_selection:
        indices = random.sample(range(len(cached_data)), min(num_samples, len(cached_data)))
    else:
        indices = list(range(min(num_samples, len(cached_data))))

    shown = 0

    for idx in indices:
        if shown >= num_samples:
            break

        data = cached_data[idx]

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
                grad_src=BEST_PARAMS["gradient_source"],
                enable_crop=False  # Already cropped
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

            # Filter by success/failure if requested
            if successful_only and not eval_result.get('top1_success', False):
                continue
            if failed_only and eval_result.get('top1_success', False):
                continue

            # Visualize
            print(f"\nSample {idx + 1}/{len(cached_data)}")
            print(f"Top-1: {'✓' if eval_result.get('top1_success', False) else '✗'} | "
                  f"IoU: {eval_result.get('avg_iou', 0.0):.4f} | "
                  f"Angle: {eval_result.get('avg_angle_diff', 0.0):.2f}°")

            visualize_single_result(
                rgb=data["rgb"],
                depth=data["depth"],
                predictions=predictions,
                ground_truths=data["grasps"],
                eval_result=eval_result,
                crop_info=info if 'cropped' in info else None
            )

            shown += 1

        except Exception as e:
            print(f"\nError visualizing sample {idx}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print(f"\n✓ Visualized {shown} samples")


# =============================================================================
# MAIN VISUALIZATION
# =============================================================================

print("\n" + "=" * 80)
print("GRASP DETECTION VISUALIZATION")
print("=" * 80)
print("\nOptions:")
print("1. Random samples (mix of success/failure)")
print("2. Successful detections only")
print("3. Failed detections only")
print("4. Sequential samples")
print("=" * 80)

# Show 5 random samples (mix of success and failure)
visualize_batch_results(
    cached_data=cached_data,
    detector=detector,
    evaluator=evaluator,
    num_samples=5,
    successful_only=False,
    failed_only=False,
    random_selection=True
)

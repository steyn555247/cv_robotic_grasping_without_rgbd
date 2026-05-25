import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import random

# =============================================================================
# VISUALIZE GRASP DETECTION RESULTS
# =============================================================================

def draw_grasp(ax, grasp, color='green', label=None, linewidth=2, show_center=True):
    """Draw a grasp rectangle on axis."""
    # Get corners
    if hasattr(grasp, 'get_corners'):
        corners = grasp.get_corners()
        center = (grasp.x, grasp.y)
    elif isinstance(grasp, dict) and 'corners' in grasp:
        corners = grasp['corners']
        center = grasp['center']
    else:
        return

    # Draw rectangle
    polygon = Polygon(corners, fill=False, edgecolor=color, linewidth=linewidth, label=label)
    ax.add_patch(polygon)

    # Draw center
    if show_center:
        ax.plot(center[0], center[1], 'o', color=color, markersize=6)


def show_result(sample_idx, cached_data, detector, evaluator):
    """Show grasp detection result for a single sample."""

    data = cached_data[sample_idx]

    # Run detection
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
        enable_crop=False
    )

    # Convert predictions
    predictions = [
        GraspCandidate(x=g.x, y=g.y, angle=g.angle, width=g.width, height=g.height)
        for g in grasps
    ]

    # Evaluate
    eval_result = evaluator.evaluate_image(predictions, data["grasps"])

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))

    # 1. RGB with Ground Truth
    axes[0, 0].imshow(data["rgb"])
    axes[0, 0].set_title('Ground Truth Grasps', fontsize=16, fontweight='bold')
    for i, gt in enumerate(data["grasps"]):
        draw_grasp(axes[0, 0], gt, color='blue', label='Ground Truth' if i == 0 else None, linewidth=3)
    if data["grasps"]:
        axes[0, 0].legend(loc='upper right', fontsize=12)
    axes[0, 0].axis('off')

    # 2. Depth Map
    axes[0, 1].imshow(data["depth"], cmap='viridis')
    axes[0, 1].set_title('Depth Map', fontsize=16, fontweight='bold')
    axes[0, 1].axis('off')

    # 3. Predicted Grasps
    axes[1, 0].imshow(data["rgb"])
    axes[1, 0].set_title('Predicted Grasps', fontsize=16, fontweight='bold')
    for i, pred in enumerate(predictions):
        if i == 0:  # Top-1
            color = 'lime' if eval_result.get('top1_success') else 'red'
            label = f"Top-1 ({'SUCCESS' if eval_result.get('top1_success') else 'FAIL'})"
            draw_grasp(axes[1, 0], pred, color=color, label=label, linewidth=4)
        else:
            draw_grasp(axes[1, 0], pred, color='yellow', label='Top-5' if i == 1 else None, linewidth=2)
    if predictions:
        axes[1, 0].legend(loc='upper right', fontsize=12)
    axes[1, 0].axis('off')

    # 4. Overlay: Predictions + Ground Truth
    axes[1, 1].imshow(data["rgb"])
    axes[1, 1].set_title('Overlay: Predictions (Green/Red) + Ground Truth (Blue)', fontsize=16, fontweight='bold')

    # Ground truth in blue
    for i, gt in enumerate(data["grasps"]):
        draw_grasp(axes[1, 1], gt, color='blue', label='Ground Truth' if i == 0 else None, linewidth=2)

    # Predictions in green/red
    for i, pred in enumerate(predictions):
        if i == 0:
            color = 'lime' if eval_result.get('top1_success') else 'red'
            label = f"Top-1 ({'✓' if eval_result.get('top1_success') else '✗'})"
            draw_grasp(axes[1, 1], pred, color=color, label=label, linewidth=3)
        else:
            draw_grasp(axes[1, 1], pred, color='green', linewidth=1.5, show_center=False)

    axes[1, 1].legend(loc='upper right', fontsize=12)
    axes[1, 1].axis('off')

    # Add metrics text
    metrics_text = (
        f"Sample: {sample_idx + 1}/{len(cached_data)}\n\n"
        f"TOP-1 SUCCESS: {'✓ YES' if eval_result.get('top1_success') else '✗ NO'}\n"
        f"TOP-5 SUCCESS: {'✓ YES' if eval_result.get('top5_success') else '✗ NO'}\n"
        f"ANY SUCCESS:   {'✓ YES' if eval_result.get('any_success') else '✗ NO'}\n\n"
        f"IoU:          {eval_result.get('avg_iou', 0.0):.4f}\n"
        f"Angle Diff:   {eval_result.get('avg_angle_diff', 0.0):.2f}°\n\n"
        f"Predictions:  {len(predictions)}\n"
        f"Ground Truth: {len(data['grasps'])}"
    )

    fig.text(0.02, 0.02, metrics_text, fontsize=13, family='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    plt.show()

    return eval_result


# =============================================================================
# SHOW MULTIPLE RESULTS
# =============================================================================

def show_multiple_results(num_samples=5, mode='random'):
    """
    Show multiple grasp detection results.

    Args:
        num_samples: Number of samples to show
        mode: 'random', 'success', 'failure', or 'sequential'
    """
    print("=" * 80)
    print(f"SHOWING {num_samples} GRASP DETECTION RESULTS ({mode.upper()} mode)")
    print("=" * 80)

    if mode == 'random':
        indices = random.sample(range(len(cached_data)), min(num_samples, len(cached_data)))
    else:
        indices = range(min(num_samples, len(cached_data)))

    shown = 0
    for idx in indices:
        if shown >= num_samples:
            break

        try:
            print(f"\n{'='*80}")
            print(f"Showing sample {idx + 1}/{len(cached_data)}")
            print(f"{'='*80}")

            eval_result = show_result(idx, cached_data, detector, evaluator)

            # Filter by mode
            if mode == 'success' and not eval_result.get('top1_success'):
                continue
            if mode == 'failure' and eval_result.get('top1_success'):
                continue

            shown += 1

        except Exception as e:
            print(f"Error visualizing sample {idx}: {e}")
            continue

    print(f"\n✓ Showed {shown} results")


# =============================================================================
# RUN VISUALIZATION
# =============================================================================

print("\n" + "=" * 80)
print("GRASP DETECTION RESULTS VISUALIZATION")
print("=" * 80)

# Show 5 random samples
show_multiple_results(num_samples=5, mode='random')

# Uncomment to show specific modes:
# show_multiple_results(num_samples=5, mode='success')  # Only successful detections
# show_multiple_results(num_samples=5, mode='failure')  # Only failed detections

# Or show a specific sample:
# show_result(sample_idx=0, cached_data=cached_data, detector=detector, evaluator=evaluator)

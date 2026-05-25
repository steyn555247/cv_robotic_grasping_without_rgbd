import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import random

# =============================================================================
# VISUALIZE GRASP DETECTION WITH MASK AND PIPELINE STAGES
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


def show_result_with_mask(sample_idx, cached_data, detector, evaluator):
    """Show complete grasp detection pipeline including mask generation."""

    data = cached_data[sample_idx]

    # Run detection and capture info
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

    # Create visualization with 3 rows x 3 columns
    fig = plt.figure(figsize=(20, 16))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    # Row 1: Input images
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(data["rgb"])
    ax1.set_title('1. RGB Input Image', fontsize=14, fontweight='bold')
    ax1.axis('off')

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(data["depth"], cmap='viridis')
    ax2.set_title('2. Depth Map', fontsize=14, fontweight='bold')
    ax2.axis('off')

    ax3 = fig.add_subplot(gs[0, 2])
    ax3.imshow(data["rgb"])
    for i, gt in enumerate(data["grasps"]):
        draw_grasp(ax3, gt, color='blue', label='Ground Truth' if i == 0 else None, linewidth=3)
    if data["grasps"]:
        ax3.legend(loc='upper right', fontsize=10)
    ax3.set_title('3. Ground Truth Grasps', fontsize=14, fontweight='bold')
    ax3.axis('off')

    # Row 2: Mask and processing stages
    ax4 = fig.add_subplot(gs[1, 0])
    if 'mask' in info:
        ax4.imshow(info['mask'], cmap='gray')
        ax4.set_title('4. Generated Mask (Key Step!)', fontsize=14, fontweight='bold', color='red')
    else:
        ax4.text(0.5, 0.5, 'Mask not available', ha='center', va='center')
        ax4.set_title('4. Mask', fontsize=14, fontweight='bold')
    ax4.axis('off')

    ax5 = fig.add_subplot(gs[1, 1])
    if 'edges' in info:
        ax5.imshow(info['edges'], cmap='gray')
        ax5.set_title('5. Edge Detection', fontsize=14, fontweight='bold')
    else:
        ax5.text(0.5, 0.5, 'Edges not available', ha='center', va='center')
        ax5.set_title('5. Edges', fontsize=14, fontweight='bold')
    ax5.axis('off')

    ax6 = fig.add_subplot(gs[1, 2])
    if 'depth_grad' in info:
        ax6.imshow(info['depth_grad'], cmap='hot')
        ax6.set_title('6. Depth Gradients', fontsize=14, fontweight='bold')
    else:
        ax6.text(0.5, 0.5, 'Gradients not available', ha='center', va='center')
        ax6.set_title('6. Depth Gradients', fontsize=14, fontweight='bold')
    ax6.axis('off')

    # Row 3: Results and overlay
    ax7 = fig.add_subplot(gs[2, 0])
    # Show mask with CoG point
    if 'mask' in info:
        ax7.imshow(info['mask'], cmap='gray')
        if 'cog' in info:
            cog_x, cog_y = info['cog']
            ax7.plot(cog_x, cog_y, 'r*', markersize=20, label='Center of Gravity')
            ax7.legend(loc='upper right', fontsize=10)
        ax7.set_title('7. Mask + CoG (Red Star)', fontsize=14, fontweight='bold')
    else:
        ax7.text(0.5, 0.5, 'Mask not available', ha='center', va='center')
        ax7.set_title('7. Mask + CoG', fontsize=14, fontweight='bold')
    ax7.axis('off')

    ax8 = fig.add_subplot(gs[2, 1])
    ax8.imshow(data["rgb"])
    for i, pred in enumerate(predictions):
        if i == 0:  # Top-1
            color = 'lime' if eval_result.get('top1_success') else 'red'
            label = f"Top-1 ({'SUCCESS' if eval_result.get('top1_success') else 'FAIL'})"
            draw_grasp(ax8, pred, color=color, label=label, linewidth=4)
        else:
            draw_grasp(ax8, pred, color='yellow', label='Top-5' if i == 1 else None, linewidth=2)
    if predictions:
        ax8.legend(loc='upper right', fontsize=10)
    ax8.set_title('8. Predicted Grasps', fontsize=14, fontweight='bold')
    ax8.axis('off')

    ax9 = fig.add_subplot(gs[2, 2])
    ax9.imshow(data["rgb"])
    # Ground truth in blue
    for i, gt in enumerate(data["grasps"]):
        draw_grasp(ax9, gt, color='blue', label='Ground Truth' if i == 0 else None, linewidth=2)
    # Predictions
    for i, pred in enumerate(predictions):
        if i == 0:
            color = 'lime' if eval_result.get('top1_success') else 'red'
            label = f"Top-1 ({'✓' if eval_result.get('top1_success') else '✗'})"
            draw_grasp(ax9, pred, color=color, label=label, linewidth=3)
        else:
            draw_grasp(ax9, pred, color='green', linewidth=1.5, show_center=False)
    # Add CoG if available
    if 'cog' in info:
        cog_x, cog_y = info['cog']
        ax9.plot(cog_x, cog_y, 'r*', markersize=15, label='CoG')
    ax9.legend(loc='upper right', fontsize=10)
    ax9.set_title('9. Final Overlay (Blue=GT, Green/Red=Pred)', fontsize=14, fontweight='bold')
    ax9.axis('off')

    # Add detailed metrics text at bottom
    metrics_text = (
        f"SAMPLE: {sample_idx + 1}/{len(cached_data)}  |  "
        f"TOP-1: {'✓ SUCCESS' if eval_result.get('top1_success') else '✗ FAIL'}  |  "
        f"IoU: {eval_result.get('avg_iou', 0.0):.4f}  |  "
        f"Angle Diff: {eval_result.get('avg_angle_diff', 0.0):.2f}°  |  "
        f"Predictions: {len(predictions)}  |  "
        f"Ground Truth: {len(data['grasps'])}\n"
    )

    # Add pipeline info
    pipeline_text = "PIPELINE INFO: "
    if 'mask_area' in info:
        pipeline_text += f"Mask Area: {info['mask_area']} px  |  "
    if 'num_contour_points' in info:
        pipeline_text += f"Contour Points: {info['num_contour_points']}  |  "
    if 'num_evaluated' in info:
        pipeline_text += f"Candidates Evaluated: {info['num_evaluated']}  |  "
    if 'num_valid' in info:
        pipeline_text += f"Valid Grasps: {info['num_valid']}"

    metrics_text += pipeline_text

    fig.text(0.5, 0.02, metrics_text, fontsize=12, family='monospace', ha='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.9))

    plt.suptitle(f"GRASP DETECTION PIPELINE VISUALIZATION - Sample {sample_idx + 1}",
                fontsize=18, fontweight='bold', y=0.98)

    plt.show()

    return eval_result


# =============================================================================
# SHOW MULTIPLE RESULTS
# =============================================================================

def show_multiple_results(num_samples=5, mode='random'):
    """
    Show multiple grasp detection results with full pipeline visualization.

    Args:
        num_samples: Number of samples to show
        mode: 'random', 'success', 'failure', or 'sequential'
    """
    print("=" * 80)
    print(f"SHOWING {num_samples} GRASP DETECTION RESULTS WITH MASKS ({mode.upper()} mode)")
    print("=" * 80)
    print("\nVisualization includes:")
    print("  - Input RGB and Depth")
    print("  - Generated Mask (the key evaluation component)")
    print("  - Edge Detection and Depth Gradients")
    print("  - Center of Gravity (CoG)")
    print("  - Predicted and Ground Truth Grasps")
    print("=" * 80)

    if mode == 'random':
        indices = random.sample(range(len(cached_data)), min(num_samples, len(cached_data)))
    else:
        indices = range(min(num_samples, len(cached_data)))

    shown = 0
    results_summary = []

    for idx in indices:
        if shown >= num_samples:
            break

        try:
            print(f"\n{'='*80}")
            print(f"Processing sample {idx + 1}/{len(cached_data)}")
            print(f"{'='*80}")

            eval_result = show_result_with_mask(idx, cached_data, detector, evaluator)

            # Filter by mode
            if mode == 'success' and not eval_result.get('top1_success'):
                continue
            if mode == 'failure' and eval_result.get('top1_success'):
                continue

            results_summary.append({
                'idx': idx,
                'success': eval_result.get('top1_success'),
                'iou': eval_result.get('avg_iou', 0.0),
                'angle': eval_result.get('avg_angle_diff', 0.0)
            })

            shown += 1

        except Exception as e:
            print(f"Error visualizing sample {idx}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Print summary
    print(f"\n{'='*80}")
    print(f"VISUALIZATION SUMMARY")
    print(f"{'='*80}")
    print(f"Samples shown: {shown}")
    if results_summary:
        successes = sum(1 for r in results_summary if r['success'])
        avg_iou = np.mean([r['iou'] for r in results_summary])
        avg_angle = np.mean([r['angle'] for r in results_summary])
        print(f"Success rate: {successes}/{shown} ({successes/shown*100:.1f}%)")
        print(f"Average IoU: {avg_iou:.4f}")
        print(f"Average angle diff: {avg_angle:.2f}°")
    print(f"{'='*80}")


# =============================================================================
# RUN VISUALIZATION
# =============================================================================

print("\n" + "=" * 80)
print("GRASP DETECTION PIPELINE VISUALIZATION WITH MASK")
print("=" * 80)
print("\nThis visualization shows ALL stages of your pipeline:")
print("  1. RGB Input")
print("  2. Depth Map")
print("  3. Ground Truth")
print("  4. GENERATED MASK (key evaluation component)")
print("  5. Edge Detection")
print("  6. Depth Gradients")
print("  7. Mask + Center of Gravity")
print("  8. Predicted Grasps")
print("  9. Final Overlay")
print("=" * 80)

# Show 5 random samples with full pipeline
show_multiple_results(num_samples=5, mode='random')

# Uncomment to show specific modes:
# show_multiple_results(num_samples=5, mode='success')  # Only successful detections
# show_multiple_results(num_samples=5, mode='failure')  # Only failed detections

# Or show a specific sample with full pipeline:
# show_result_with_mask(sample_idx=0, cached_data=cached_data, detector=detector, evaluator=evaluator)

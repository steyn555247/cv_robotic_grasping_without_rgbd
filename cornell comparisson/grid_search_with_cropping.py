# =============================================================================
# EXTENSIVE GRID SEARCH WITH MANUAL CROPPING
# =============================================================================
# This script performs grid search on ~50% of the Cornell dataset
# with manual cropping applied first, then evaluates parameter variations

import cv2
import numpy as np
from tqdm import tqdm
import time
import itertools
from collections import defaultdict
import json

# =============================================================================
# CONFIGURATION
# =============================================================================

# MANUAL CROP REGION (same as your working method)
MANUAL_CROP_BBOX = (100, 150, 400, 300)
ENABLE_MANUAL_CROP = True

# Grid search parameters - based on your working configuration
BEST_PARAMS = {
    # Weights for scoring
    "w_edge": 0.001,
    "w_depth": 0.001,
    "w_cog": 0.999,

    # Detection parameters
    "depth_percentile": 30,
    "num_grasps": 1,
    "candidate_multiplier": 100,
    "min_grasp_length": 1,
    "max_grasp_length": 1000,

    # Algorithm parameters
    "ray_algorithm": "Direct Line with CoG Boost",
    "cog_boost": 3.75,
    "gradient_source": "Contour Direction (80px avg)"
}

# Define parameter search space (variations around best parameters)
# Target: ~425 combinations
PARAM_GRID = {
    # Weight variations - 5 different weight distributions
    # Testing slight variations around CoG-dominant strategy
    "w_edge": [0.001, 0.01, 0.05, 0.1, 0.2],           # 5 values
    "w_depth": [0.001, 0.01, 0.05, 0.1, 0.2],          # 5 values
    "w_cog": [0.6, 0.75, 0.9, 0.95, 0.999],            # 5 values (will normalize to sum=1)

    # Depth percentile variations - test lower values
    "depth_percentile": [20, 30, 40, 50],              # 4 values

    # Number of grasps - keep low since current best is 1
    "num_grasps": [1],                                 # 1 value

    # Candidate multiplier - test variations around 100
    "candidate_multiplier": [50, 75, 100, 125],        # 4 values

    # Grasp length constraints - current is very permissive (1-1000)
    # Test if tighter constraints help
    "min_grasp_length": [1, 10, 20],                   # 3 values
    "max_grasp_length": [500, 750, 1000],              # 3 values

    # Algorithm choice - keep best
    "ray_algorithm": [BEST_PARAMS["ray_algorithm"]],   # 1 value

    # CoG boost - test variations around 3.75
    "cog_boost": [2.5, 3.0, 3.5, 3.75, 4.0, 4.5, 5.0], # 7 values

    # Gradient source - keep best
    "gradient_source": [BEST_PARAMS["gradient_source"]]  # 1 value
}

# BUT: 5×5×5 = 125 weight combinations is too many!
# We'll use a smarter approach: pre-define good weight combinations
# This reduces from 125 to 17 weight combinations

# Smart weight combinations (17 combinations)
WEIGHT_COMBINATIONS = [
    # Current best and close variations
    (0.001, 0.001, 0.998),   # Current best (normalized)
    (0.01, 0.01, 0.98),
    (0.05, 0.05, 0.90),
    (0.1, 0.1, 0.8),

    # Edge-focused variations
    (0.2, 0.1, 0.7),
    (0.3, 0.1, 0.6),
    (0.4, 0.1, 0.5),

    # Depth-focused variations
    (0.1, 0.2, 0.7),
    (0.1, 0.3, 0.6),
    (0.1, 0.4, 0.5),

    # More balanced approaches
    (0.2, 0.2, 0.6),
    (0.25, 0.25, 0.5),
    (0.3, 0.3, 0.4),
    (0.33, 0.33, 0.34),  # Equal weights

    # Edge+Depth focused (less CoG)
    (0.3, 0.4, 0.3),
    (0.4, 0.3, 0.3),
    (0.45, 0.45, 0.1),
]

# Recalculate grid without individual weight parameters
PARAM_GRID_COMPACT = {
    "weight_combo": list(range(len(WEIGHT_COMBINATIONS))),  # 17 values
    "depth_percentile": [20, 30, 40, 50],                   # 4 values
    "num_grasps": [1],                                       # 1 value
    "candidate_multiplier": [50, 75, 100, 125],             # 4 values
    "min_grasp_length": [1, 10, 20],                        # 3 values
    "max_grasp_length": [500, 750, 1000],                   # 3 values
    "ray_algorithm": [BEST_PARAMS["ray_algorithm"]],        # 1 value
    "cog_boost": [2.5, 3.0, 3.5, 3.75, 4.0, 4.5, 5.0],     # 7 values
    "gradient_source": [BEST_PARAMS["gradient_source"]]     # 1 value
}

# Total combinations: 17 × 4 × 1 × 4 × 3 × 3 × 1 × 7 × 1 = 8,568
# Still too many! Let's be even more selective

# FINAL GRID - targeting ~425 combinations
PARAM_GRID_FINAL = {
    "weight_combo": list(range(len(WEIGHT_COMBINATIONS))),  # 17 values
    "depth_percentile": [20, 30, 40],                       # 3 values
    "num_grasps": [1],                                       # 1 value
    "candidate_multiplier": [75, 100, 125],                 # 3 values
    "min_grasp_length": [1, 20],                            # 2 values
    "max_grasp_length": [750, 1000],                        # 2 values
    "ray_algorithm": [BEST_PARAMS["ray_algorithm"]],        # 1 value
    "cog_boost": [3.0, 3.75, 4.5],                         # 3 values
    "gradient_source": [BEST_PARAMS["gradient_source"]]     # 1 value
}

# Total: 17 × 3 × 1 × 3 × 2 × 2 × 1 × 3 × 1 = 1,836
# Still more than 425. Let's reduce weight combinations

# ULTRA FOCUSED - exactly targeting ~400-500 combinations
WEIGHT_COMBINATIONS_FOCUSED = [
    (0.001, 0.001, 0.998),   # Current best
    (0.01, 0.01, 0.98),      # Slight variation
    (0.05, 0.05, 0.90),      # More edge/depth
    (0.1, 0.1, 0.8),         # Even more
    (0.2, 0.1, 0.7),         # Edge-focused
    (0.1, 0.2, 0.7),         # Depth-focused
    (0.2, 0.2, 0.6),         # Balanced CoG-dominant
    (0.3, 0.3, 0.4),         # More balanced
    (0.33, 0.33, 0.34),      # Equal weights
]

PARAM_GRID = {
    "weight_combo": list(range(len(WEIGHT_COMBINATIONS_FOCUSED))),  # 9 values
    "depth_percentile": [20, 30, 40, 50],                           # 4 values
    "num_grasps": [1],                                               # 1 value
    "candidate_multiplier": [75, 100, 125],                         # 3 values
    "min_grasp_length": [1, 20],                                    # 2 values
    "max_grasp_length": [750, 1000],                                # 2 values
    "ray_algorithm": [BEST_PARAMS["ray_algorithm"]],                # 1 value
    "cog_boost": [3.0, 3.75, 4.5],                                 # 3 values
    "gradient_source": [BEST_PARAMS["gradient_source"]]             # 1 value
}

# Total: 9 × 4 × 1 × 3 × 2 × 2 × 1 × 3 × 1 = 432 combinations ✓

print("=" * 80)
print("GRID SEARCH CONFIGURATION")
print("=" * 80)
print(f"Manual Crop: {ENABLE_MANUAL_CROP}")
print(f"Crop Region: x={MANUAL_CROP_BBOX[0]}, y={MANUAL_CROP_BBOX[1]}, "
      f"w={MANUAL_CROP_BBOX[2]}, h={MANUAL_CROP_BBOX[3]}")
print("\nParameter Grid:")
for param, values in PARAM_GRID.items():
    print(f"  {param}: {values}")

# Calculate total combinations
total_combinations = 1
for values in PARAM_GRID.values():
    total_combinations *= len(values)
print(f"\nTotal parameter combinations: {total_combinations}")
print("=" * 80)


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def manual_crop(image, crop_bbox):
    """Manually crop image to specified region."""
    x, y, w, h = crop_bbox
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


def split_dataset(samples, train_ratio=0.5, random_seed=42):
    """Split dataset into train and test sets."""
    np.random.seed(random_seed)
    indices = np.random.permutation(len(samples))
    split_idx = int(len(samples) * train_ratio)
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]

    train_samples = [samples[i] for i in train_indices]
    test_samples = [samples[i] for i in test_indices]

    return train_samples, test_samples


# =============================================================================
# STEP 1: SPLIT DATASET AND PREPARE DATA
# =============================================================================

print("\n" + "=" * 80)
print("STEP 1: DATASET PREPARATION")
print("=" * 80)

# Split dataset (assumes 'samples' variable exists from your notebook)
# If running standalone, you'll need to load samples first
print(f"Total samples available: {len(samples)}")
train_samples, test_samples = split_dataset(samples, train_ratio=0.5)
print(f"Training set: {len(train_samples)} samples (~50%)")
print(f"Test set: {len(test_samples)} samples (~50%)")

# Process training data with manual cropping
print("\nProcessing training data with manual cropping...")
cached_train_data = []
crop_stats = {
    'total': 0,
    'cropped': 0,
    'skipped_no_grasps': 0,
    'grasps_removed': 0,
    'grasps_kept': 0
}

start_time = time.time()

for sample in tqdm(train_samples, desc="Cropping + Depth"):
    try:
        # Load RGB
        rgb = cv2.imread(sample["rgb_path"])
        if rgb is None:
            continue
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)

        crop_stats['total'] += 1
        grasps_to_use = sample["grasps"]
        crop_info = {'cropped': False}

        # MANUAL CROP
        if ENABLE_MANUAL_CROP:
            cropped_rgb, crop_bbox, success = manual_crop(rgb, MANUAL_CROP_BBOX)

            if success:
                original_grasp_count = len(sample["grasps"])
                adjusted_grasps = adjust_grasp_coordinates(sample["grasps"], crop_bbox)

                grasps_removed = original_grasp_count - len(adjusted_grasps)
                crop_stats['grasps_removed'] += grasps_removed
                crop_stats['grasps_kept'] += len(adjusted_grasps)

                if len(adjusted_grasps) > 0:
                    rgb = cropped_rgb
                    grasps_to_use = adjusted_grasps
                    crop_stats['cropped'] += 1
                    crop_info = {
                        'cropped': True,
                        'crop_bbox': crop_bbox,
                        'grasps_removed': grasps_removed
                    }
                else:
                    crop_stats['skipped_no_grasps'] += 1
                    continue

        # Compute depth (assumes depth_model exists from your notebook)
        depth, _ = depth_model.predict(rgb)

        cached_train_data.append({
            "rgb": rgb,
            "depth": depth,
            "grasps": grasps_to_use,
            "crop_info": crop_info,
            "original_path": sample["rgb_path"]
        })

    except Exception as e:
        print(f"\nError processing {sample.get('rgb_path', 'unknown')}: {e}")
        continue

prep_time = time.time() - start_time

print(f"\n✓ Processed {len(cached_train_data)}/{len(train_samples)} training samples")
print(f"Preparation time: {prep_time:.1f}s")
print(f"\nCrop Statistics:")
print(f"  Cropped: {crop_stats['cropped']}")
print(f"  Grasps kept: {crop_stats['grasps_kept']}")
print(f"  Grasps removed: {crop_stats['grasps_removed']}")


# =============================================================================
# STEP 2: GRID SEARCH
# =============================================================================

print("\n" + "=" * 80)
print("STEP 2: GRID SEARCH")
print("=" * 80)

# Generate all parameter combinations
param_names = list(PARAM_GRID.keys())
param_values = list(PARAM_GRID.values())
all_combinations = list(itertools.product(*param_values))

print(f"Testing {len(all_combinations)} parameter combinations...")
print("=" * 80)

# Store results
grid_results = []
best_result = {
    'top1_acc': 0,
    'params': None
}

start_time = time.time()

# Iterate through all parameter combinations
for combo_idx, combo in enumerate(all_combinations):
    # Create parameter dictionary for this combination
    params = dict(zip(param_names, combo))

    # Extract weights from the weight combination index
    weight_idx = params["weight_combo"]
    w_edge, w_depth, w_cog = WEIGHT_COMBINATIONS_FOCUSED[weight_idx]

    # Create full parameter dictionary with actual weight values
    params_full = {
        "w_edge": w_edge,
        "w_depth": w_depth,
        "w_cog": w_cog,
        "depth_percentile": params["depth_percentile"],
        "num_grasps": params["num_grasps"],
        "candidate_multiplier": params["candidate_multiplier"],
        "min_grasp_length": params["min_grasp_length"],
        "max_grasp_length": params["max_grasp_length"],
        "ray_algorithm": params["ray_algorithm"],
        "cog_boost": params["cog_boost"],
        "gradient_source": params["gradient_source"]
    }

    print(f"\n[{combo_idx + 1}/{len(all_combinations)}] Testing parameters:")
    print(f"  Weights: edge={w_edge:.3f}, depth={w_depth:.3f}, cog={w_cog:.3f}")
    print(f"  Depth%={params_full['depth_percentile']}, #grasps={params_full['num_grasps']}, mult={params_full['candidate_multiplier']}")
    print(f"  Length: {params_full['min_grasp_length']}-{params_full['max_grasp_length']}, boost={params_full['cog_boost']:.2f}")

    # Initialize detector with current parameters
    detector = GraspDetector(
        w_edge,
        w_depth,
        w_cog,
    )

    # Evaluate on training data
    top1_successes = 0
    top5_successes = 0
    any_successes = 0
    ious = []
    angle_diffs = []
    total = 0

    for data in tqdm(cached_train_data, desc="  Evaluating", leave=False):
        try:
            # Run grasp detection
            grasps, info = detector.process(
                rgb=data["rgb"],
                depth=data["depth"],
                n_grasps=params_full["num_grasps"],
                pct=params_full["depth_percentile"],
                mult=params_full["candidate_multiplier"],
                min_l=params_full["min_grasp_length"],
                max_l=params_full["max_grasp_length"],
                algo=params_full["ray_algorithm"],
                boost=params_full["cog_boost"],
                grad_src=params_full["gradient_source"]
            )

            # Convert to GraspCandidate (assumes GraspCandidate class exists)
            predictions = [
                GraspCandidate(x=g.x, y=g.y, angle=g.angle, width=g.width, height=g.height)
                for g in grasps
            ]

            # Evaluate (assumes evaluator exists from your notebook)
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
            continue

    # Calculate metrics
    if total > 0:
        top1_acc = top1_successes / total * 100.0
        top5_acc = top5_successes / total * 100.0
        any_acc = any_successes / total * 100.0
        mean_iou = float(np.mean(ious)) if ious else 0.0
        mean_angle = float(np.mean(angle_diffs)) if angle_diffs else 0.0
    else:
        top1_acc = top5_acc = any_acc = mean_iou = mean_angle = 0.0

    # Store results
    result = {
        'params': params_full.copy(),
        'top1_acc': top1_acc,
        'top5_acc': top5_acc,
        'any_acc': any_acc,
        'mean_iou': mean_iou,
        'mean_angle': mean_angle,
        'total_evaluated': total
    }
    grid_results.append(result)

    # Update best result
    if top1_acc > best_result['top1_acc']:
        best_result = {
            'top1_acc': top1_acc,
            'params': params_full.copy(),
            'full_result': result
        }

    print(f"  Results: Top-1={top1_acc:.2f}%, Top-5={top5_acc:.2f}%, IoU={mean_iou:.4f}")

grid_search_time = time.time() - start_time


# =============================================================================
# STEP 3: RESULTS SUMMARY
# =============================================================================

print("\n" + "=" * 80)
print("GRID SEARCH RESULTS")
print("=" * 80)

# Sort results by Top-1 accuracy
grid_results.sort(key=lambda x: x['top1_acc'], reverse=True)

print(f"\nTotal combinations tested: {len(grid_results)}")
print(f"Grid search time: {grid_search_time:.1f}s ({grid_search_time/60:.1f} minutes)")
print(f"Average time per combination: {grid_search_time/len(grid_results):.1f}s")

print("\n" + "-" * 80)
print("TOP 10 PARAMETER COMBINATIONS")
print("-" * 80)

for i, result in enumerate(grid_results[:10]):
    params = result['params']
    print(f"\n#{i+1} - Top-1: {result['top1_acc']:.2f}%, Top-5: {result['top5_acc']:.2f}%, IoU: {result['mean_iou']:.4f}")
    print(f"  Weights: edge={params['w_edge']:.3f}, depth={params['w_depth']:.3f}, cog={params['w_cog']:.3f}")
    print(f"  Depth%={params['depth_percentile']}, #grasps={params['num_grasps']}, mult={params['candidate_multiplier']}")
    print(f"  Length: {params['min_grasp_length']}-{params['max_grasp_length']}, boost={params['cog_boost']:.2f}")

print("\n" + "=" * 80)
print("BEST PARAMETERS FOUND")
print("=" * 80)

best_params = grid_results[0]['params']
print("\nBEST_PARAMS = {")
for key, value in best_params.items():
    if isinstance(value, float):
        print(f'    "{key}": {value:.4f},')
    else:
        print(f'    "{key}": {value},')
print("}")

print(f"\nBest Top-1 Accuracy: {grid_results[0]['top1_acc']:.2f}%")
print(f"Best Top-5 Accuracy: {grid_results[0]['top5_acc']:.2f}%")
print(f"Best Mean IoU: {grid_results[0]['mean_iou']:.4f}")
print(f"Best Mean Angle Diff: {grid_results[0]['mean_angle']:.2f}°")

# Save results to JSON
results_file = 'grid_search_results.json'
with open(results_file, 'w') as f:
    json.dump({
        'best_params': best_params,
        'all_results': grid_results,
        'config': {
            'manual_crop': ENABLE_MANUAL_CROP,
            'crop_bbox': MANUAL_CROP_BBOX,
            'train_samples': len(cached_train_data),
            'grid_search_time': grid_search_time
        }
    }, f, indent=2)

print(f"\n✓ Results saved to: {results_file}")
print("=" * 80)

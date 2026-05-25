# Grid Search with Manual Cropping - Usage Guide

## Overview
This grid search script tests **432 parameter combinations** on approximately 50% of your Cornell dataset with manual cropping applied first.

## Configuration

### Your Current Best Parameters
```python
W_EDGE = 0.001              # Edge quality weight
W_DEPTH = 0.001             # Depth gradient weight
W_COG = 0.999               # CoG proximity weight
DEPTH_PERCENTILE = 30       # Depth percentile cutoff
RAY_ALGORITHM = "Direct Line with CoG Boost"
COG_BOOST_VALUE = 3.75
GRADIENT_SOURCE = "Contour Direction (80px avg)"
MIN_GRASP_LENGTH = 1
MAX_GRASP_LENGTH = 1000
NUM_OUTPUT_GRASPS = 1
CANDIDATE_MULTIPLIER = 100
```

### Manual Crop Region
- Horizontal: 100 to 500 (x=100, width=400)
- Vertical: 150 to 450 (y=150, height=300)

## Parameter Search Space (432 combinations total)

### Weight Combinations (9 variations)
1. (0.001, 0.001, 0.998) - **Current best**
2. (0.01, 0.01, 0.98) - Slight variation
3. (0.05, 0.05, 0.90) - More edge/depth
4. (0.1, 0.1, 0.8) - Even more balanced
5. (0.2, 0.1, 0.7) - Edge-focused
6. (0.1, 0.2, 0.7) - Depth-focused
7. (0.2, 0.2, 0.6) - Balanced CoG-dominant
8. (0.3, 0.3, 0.4) - More balanced
9. (0.33, 0.33, 0.34) - Equal weights

### Other Parameters
- **Depth Percentile**: [20, 30, 40, 50] - 4 values
- **Num Grasps**: [1] - 1 value (keeping your best)
- **Candidate Multiplier**: [75, 100, 125] - 3 values
- **Min Grasp Length**: [1, 20] - 2 values
- **Max Grasp Length**: [750, 1000] - 2 values
- **Ray Algorithm**: ["Direct Line with CoG Boost"] - 1 value (keeping your best)
- **CoG Boost**: [3.0, 3.75, 4.5] - 3 values
- **Gradient Source**: ["Contour Direction (80px avg)"] - 1 value (keeping your best)

**Total: 9 × 4 × 1 × 3 × 2 × 2 × 1 × 3 × 1 = 432 combinations**

## How to Run

### Option 1: Run in Jupyter Notebook
Add this cell to your notebook after all your setup code (depth_model, evaluator, GraspDetector, samples, etc.):

```python
# Execute the grid search script
exec(open('grid_search_with_cropping.py').read())
```

### Option 2: Run as Python Script
Make sure your notebook has exported the necessary variables, then:

```bash
python grid_search_with_cropping.py
```

## Required Variables
The script assumes these variables exist from your notebook:
- `samples` - List of Cornell dataset samples
- `depth_model` - Depth estimation model
- `evaluator` - Grasp evaluation class
- `GraspDetector` - Your grasp detection class
- `GraspCandidate` - Grasp candidate class

## Output

### Console Output
The script will print:
1. Configuration summary
2. Cropping statistics
3. Progress for each parameter combination
4. Top 10 best parameter sets
5. Best parameters in ready-to-use format

### JSON Output File
Results are saved to `grid_search_results.json` containing:
- Best parameters
- All 432 results sorted by Top-1 accuracy
- Configuration details
- Timing information

## Expected Runtime
- Dataset split & crop: ~2-5 minutes (depends on dataset size)
- Grid search: ~30-90 minutes (depends on number of samples in training set)
- Total: ~35-95 minutes for 432 combinations

**Note**: With ~450 training samples (half of 900 total), each combination takes ~5-10 seconds

## Example Output

```
================================================================================
GRID SEARCH RESULTS
================================================================================

Total combinations tested: 432
Grid search time: 3245.2s (54.1 minutes)
Average time per combination: 7.5s

--------------------------------------------------------------------------------
TOP 10 PARAMETER COMBINATIONS
--------------------------------------------------------------------------------

#1 - Top-1: 87.23%, Top-5: 95.12%, IoU: 0.7234
  Weights: edge=0.001, depth=0.001, cog=0.998
  Depth%=30, #grasps=1, mult=100
  Length: 1-1000, boost=3.75

#2 - Top-1: 86.45%, Top-5: 94.87%, IoU: 0.7189
  Weights: edge=0.010, depth=0.010, cog=0.980
  Depth%=30, #grasps=1, mult=100
  Length: 1-1000, boost=4.00

...
```

## Customizing the Search

### To add more weight combinations:
Edit `WEIGHT_COMBINATIONS_FOCUSED` list around line 143

### To change other parameter ranges:
Edit `PARAM_GRID` dictionary around line 155

### To reduce/increase search space:
Adjust the values in `PARAM_GRID` - remember total combinations = product of all list lengths

## Tips
1. Start with the default 432 combinations
2. Review top 10 results to identify patterns
3. If needed, run a second focused search around the best parameters found
4. Use the JSON file to analyze results in detail

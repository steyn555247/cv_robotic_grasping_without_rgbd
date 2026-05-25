# =============================================================================
# DISPLAY ALL PARAMETER VALUES
# =============================================================================

print("=" * 80)
print("CURRENT PARAMETER VALUES")
print("=" * 80)

# Check if BEST_PARAMS exists
try:
    print("\nBEST_PARAMS dictionary:")
    print("-" * 80)
    for key, value in BEST_PARAMS.items():
        print(f"  {key:25s} = {value}")
    print("=" * 80)
except NameError:
    print("\n❌ BEST_PARAMS is not defined!")
    print("=" * 80)

# Also check for individual parameter variables if they exist
print("\nIndividual parameters (if defined separately):")
print("-" * 80)

param_names = [
    'W_EDGE', 'W_DEPTH', 'W_COG',
    'DEPTH_PERCENTILE',
    'RAY_ALGORITHM', 'COG_BOOST_VALUE', 'GRADIENT_SOURCE',
    'MIN_GRASP_LENGTH', 'MAX_GRASP_LENGTH',
    'NUM_OUTPUT_GRASPS', 'CANDIDATE_MULTIPLIER'
]

for param_name in param_names:
    try:
        param_value = eval(param_name)
        print(f"  {param_name:25s} = {param_value}")
    except NameError:
        print(f"  {param_name:25s} = NOT DEFINED")

print("=" * 80)

# Show detector weights if detector exists
print("\nDetector weights (if detector is initialized):")
print("-" * 80)
try:
    print(f"  Edge weight (w_e):        {detector.we}")
    print(f"  Depth weight (w_d):       {detector.wd}")
    print(f"  CoG weight (w_c):         {detector.wc}")
except NameError:
    print("  ❌ Detector not initialized yet")

print("=" * 80)

# Summary table
print("\n" + "=" * 80)
print("PARAMETER SUMMARY FOR VISUALIZATION")
print("=" * 80)

try:
    print(f"""
Parameters being used in visualization:

  QUALITY WEIGHTS:
    Edge quality weight       : {BEST_PARAMS.get('w_edge', 'N/A')}
    Depth quality weight      : {BEST_PARAMS.get('w_depth', 'N/A')}
    CoG quality weight        : {BEST_PARAMS.get('w_cog', 'N/A')}

  MASKING:
    Depth percentile cutoff   : {BEST_PARAMS.get('depth_percentile', 'N/A')}

  RAY CASTING:
    Algorithm                 : {BEST_PARAMS.get('ray_algorithm', 'N/A')}
    CoG boost value           : {BEST_PARAMS.get('cog_boost', 'N/A')}
    Gradient source           : {BEST_PARAMS.get('gradient_source', 'N/A')}

  FILTERING:
    Minimum grasp length      : {BEST_PARAMS.get('min_grasp_length', 'N/A')} px
    Maximum grasp length      : {BEST_PARAMS.get('max_grasp_length', 'N/A')} px

  OUTPUT:
    Number of grasps          : {BEST_PARAMS.get('num_grasps', 'N/A')}
    Candidate multiplier      : {BEST_PARAMS.get('candidate_multiplier', 'N/A')}
""")
except NameError:
    print("\n❌ BEST_PARAMS not found - cannot display summary")

print("=" * 80)

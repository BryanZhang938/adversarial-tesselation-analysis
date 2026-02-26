#!/usr/bin/env python3
"""
Colab Setup Script for Tessellation Dynamics Project
=====================================================

Run this as the FIRST cell in your Colab notebook.

Handles:
1. Extracts project from Google Drive (if uploaded as tar.gz)
2. Installs core pip dependencies
3. Installs graph-tool via condacolab (requires kernel restart)
4. Clones SplineCam and adds to sys.path (no setup.py needed)
5. Patches SplineCam TorchScript float-type errors in utils.py
6. Patches SplineCam multiprocessing to single-process mode
7. Sets matplotlib to 'Agg' backend for headless rendering

Usage in Colab:
    # Cell 1: Mount drive and extract project
    from google.colab import drive
    drive.mount('/content/drive')
    !cp /content/drive/MyDrive/tessellation_project.tar.gz /content/
    !cd /content && tar xzf tessellation_project.tar.gz

    # Cell 2: Run this script
    %run /content/tessellation_project/colab_setup.py

    # Cell 3 (after kernel restart if graph-tool installed):
    %run /content/tessellation_project/colab_setup.py --post-restart

    # Cell 4: Run experiment
    %cd /content/tessellation_project
    !python run_experiment.py --device cpu
"""

import os
import sys
import subprocess
import argparse

# ============================================================================
# Configuration
# ============================================================================
# Auto-detect project directory: use the directory this script lives in,
# or fall back to common locations.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_CANDIDATE_DIRS = [
    _SCRIPT_DIR,
    "/content/tessellation_project",
    "/content/adversarial-tesselation-analysis",
    "/content/adversarial-tessellation-analysis",
]
PROJECT_DIR = _SCRIPT_DIR  # default
for _d in _CANDIDATE_DIRS:
    if os.path.isfile(os.path.join(_d, "run_experiment.py")):
        PROJECT_DIR = _d
        break

SPLINECAM_DIR = "/content/splinecam"


def run_cmd(cmd, check=True):
    """Run a shell command and print output."""
    print(f">>> {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.stdout.strip():
        print(result.stdout.strip())
    if result.stderr.strip():
        # Filter out common pip noise
        for line in result.stderr.strip().split('\n'):
            if 'WARNING' not in line and 'already satisfied' not in line.lower():
                print(f"  [stderr] {line}")
    if check and result.returncode != 0:
        print(f"  [ERROR] Command failed with code {result.returncode}")
    return result.returncode == 0


# ============================================================================
# Step 1: Core pip dependencies
# ============================================================================
def install_pip_deps():
    print("\n" + "="*60)
    print("Step 1: Installing pip dependencies")
    print("="*60)
    run_cmd("pip install -q torch torchvision matplotlib numpy scipy tqdm")
    run_cmd("pip install -q python-igraph networkx")


# ============================================================================
# Step 2: SplineCam (via sys.path, not pip install)
# ============================================================================
def install_splinecam():
    print("\n" + "="*60)
    print("Step 2: Installing SplineCam")
    print("="*60)

    if not os.path.exists(SPLINECAM_DIR):
        run_cmd(f"git clone https://github.com/AhmedImtiazPrio/splinecam.git {SPLINECAM_DIR}")
    else:
        print(f"  SplineCam already cloned at {SPLINECAM_DIR}")

    # Patch graph.py BEFORE importing — wrap graph_tool import in try/except
    # so SplineCam doesn't crash when graph_tool is missing.
    graph_path = os.path.join(SPLINECAM_DIR, "splinecam", "graph.py")
    if os.path.exists(graph_path):
        with open(graph_path, 'r') as f:
            content = f.read()
        if 'import graph_tool' in content and 'try:' not in content.split('import graph_tool')[0][-30:]:
            content = content.replace(
                'import graph_tool',
                'try:\n    import graph_tool\nexcept ImportError:\n    graph_tool = None'
            )
            # Also fix the SyntaxWarning from invalid escape sequence
            content = content.replace('\\i', '\\\\i')
            with open(graph_path, 'w') as f:
                f.write(content)
            print("  Patched graph.py: wrapped graph_tool import in try-except")

    # Add to sys.path (since there's no setup.py/pyproject.toml)
    if SPLINECAM_DIR not in sys.path:
        sys.path.insert(0, SPLINECAM_DIR)
        print(f"  Added {SPLINECAM_DIR} to sys.path")

    # Verify import
    try:
        import splinecam
        print(f"  SplineCam imported successfully from {splinecam.__file__}")
    except ImportError as e:
        print(f"  [WARNING] SplineCam import failed: {e}")
        print("  Grid-based analysis will still work.")


# ============================================================================
# Step 3: Patch SplineCam TorchScript type errors
# ============================================================================
def patch_splinecam_torchscript():
    print("\n" + "="*60)
    print("Step 3: Patching SplineCam TorchScript type issues")
    print("="*60)

    utils_path = os.path.join(SPLINECAM_DIR, "splinecam", "utils.py")
    if not os.path.exists(utils_path):
        print("  SplineCam utils.py not found, skipping patch.")
        return

    with open(utils_path, 'r') as f:
        content = f.read()

    original = content

    # Fix integer defaults that TorchScript can't infer as float
    # These are the specific functions that fail:
    patches = {
        'pad_dist=1,': 'pad_dist: float=1.0,',
        'pad_dist=1)': 'pad_dist: float=1.0)',
        'pad_dist = 1,': 'pad_dist: float = 1.0,',
        'pad_dist = 1)': 'pad_dist: float = 1.0)',
    }

    for old, new in patches.items():
        content = content.replace(old, new)

    if content != original:
        with open(utils_path, 'w') as f:
            f.write(content)
        print("  Patched utils.py: pad_dist int -> float for TorchScript")
    else:
        print("  utils.py already patched or no changes needed.")


# ============================================================================
# Step 4: Patch SplineCam multiprocessing (Colab CUDA fork issue)
# ============================================================================
def patch_splinecam_multiprocessing():
    print("\n" + "="*60)
    print("Step 4: Patching SplineCam multiprocessing for Colab")
    print("="*60)

    compute_path = os.path.join(SPLINECAM_DIR, "splinecam", "compute.py")
    if not os.path.exists(compute_path):
        print("  SplineCam compute.py not found, skipping patch.")
        return

    with open(compute_path, 'r') as f:
        content = f.read()

    original = content

    # Strategy: replace Pool-based parallel map with sequential execution
    # Common patterns in SplineCam compute.py:
    #   from multiprocessing import Pool
    #   pool = Pool(processes=N)
    #   results = pool.map(func, args)
    #
    # We replace pool.map with plain map() and neutralize Pool creation.

    # Approach: Add a wrapper at the top of the file that overrides Pool
    if '# PATCHED: multiprocessing disabled for Colab' not in content:
        patch_header = '''
# PATCHED: multiprocessing disabled for Colab
# Original multiprocessing.Pool is replaced with a sequential fallback
# to avoid CUDA fork() errors in Colab/notebook environments.
import multiprocessing as _mp_original

class _SequentialPool:
    """Drop-in replacement for multiprocessing.Pool that runs sequentially."""
    def __init__(self, *args, **kwargs):
        pass
    def map(self, func, iterable):
        return list(map(func, iterable))
    def starmap(self, func, iterable):
        return [func(*args) for args in iterable]
    def close(self):
        pass
    def join(self):
        pass
    def __enter__(self):
        return self
    def __exit__(self, *args):
        pass

# Override Pool in this module's namespace
try:
    from multiprocessing import Pool as _OriginalPool
except ImportError:
    _OriginalPool = None

Pool = _SequentialPool
# END PATCH
'''
        # Insert after initial imports
        # Find the first line that's not a comment, docstring, or import
        lines = content.split('\n')
        insert_idx = 0
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith('import ') or stripped.startswith('from '):
                insert_idx = i + 1

        lines.insert(insert_idx, patch_header)
        content = '\n'.join(lines)

        with open(compute_path, 'w') as f:
            f.write(content)
        print("  Patched compute.py: replaced Pool with sequential execution")
    else:
        print("  compute.py already patched.")


# ============================================================================
# Step 5: graph-tool via condacolab (optional, for exact SplineCam)
# ============================================================================
def install_graph_tool():
    """
    Install graph-tool via condacolab. This REQUIRES a kernel restart.
    Only needed if you want exact SplineCam tessellation (not grid-based).
    """
    print("\n" + "="*60)
    print("Step 5: Installing graph-tool via condacolab")
    print("="*60)

    try:
        import graph_tool
        print("  graph-tool already installed.")
        return True
    except ImportError:
        pass

    print("  graph-tool not found. Attempting condacolab install...")
    print("  NOTE: This will require a kernel restart after completion.")

    try:
        # Install condacolab
        run_cmd("pip install -q condacolab")
        import condacolab
        condacolab.install()
        # After condacolab.install(), the kernel should restart automatically.
        # If we reach here, it didn't restart yet.
        print("  condacolab installed. Kernel restart may be needed.")
        print("  After restart, run this script with --post-restart flag.")
        return False
    except Exception as e:
        print(f"  [WARNING] condacolab install failed: {e}")
        print("  Continuing without graph-tool. Grid-based analysis will work fine.")
        return False


def install_graph_tool_post_restart():
    """Run after kernel restart to complete graph-tool installation."""
    print("\n" + "="*60)
    print("Step 5b: Post-restart graph-tool installation")
    print("="*60)

    # Remove version pin that conflicts with Python version
    pinned_path = "/usr/local/conda-meta/pinned"
    if os.path.exists(pinned_path):
        os.remove(pinned_path)
        print(f"  Removed conflicting {pinned_path}")

    run_cmd("mamba install -y -c conda-forge graph-tool libcairo pycairo")

    # Also patch SplineCam's graph.py import to be fault-tolerant
    graph_path = os.path.join(SPLINECAM_DIR, "splinecam", "graph.py")
    if os.path.exists(graph_path):
        with open(graph_path, 'r') as f:
            content = f.read()
        if 'try:' not in content.split('import graph_tool')[0][-20:] if 'import graph_tool' in content else True:
            content = content.replace(
                'import graph_tool',
                'try:\n    import graph_tool\nexcept ImportError:\n    graph_tool = None\n    print("[splinecam.graph] graph_tool not available")'
            )
            with open(graph_path, 'w') as f:
                f.write(content)
            print("  Wrapped graph_tool import in try-except")

    # Verify
    try:
        import graph_tool
        print(f"  graph-tool installed successfully: {graph_tool.__version__}")
    except ImportError:
        print("  [WARNING] graph-tool still not available after install.")


# ============================================================================
# Step 6: Environment verification
# ============================================================================
def verify_environment():
    print("\n" + "="*60)
    print("Environment Verification")
    print("="*60)

    import torch
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")

    try:
        import splinecam
        print(f"  SplineCam: available")
    except ImportError:
        print(f"  SplineCam: NOT available (grid analysis will work)")

    try:
        import graph_tool
        print(f"  graph-tool: {graph_tool.__version__}")
    except ImportError:
        print(f"  graph-tool: NOT available")

    try:
        import matplotlib
        matplotlib.use('Agg')
        print(f"  matplotlib: {matplotlib.__version__} (Agg backend)")
    except ImportError:
        print(f"  matplotlib: NOT available")

    # Check project files
    expected_files = [
        "configs/experiment_config.py",
        "src/datasets.py",
        "src/models.py",
        "src/adversarial.py",
        "src/train.py",
        "src/tessellation_analysis.py",
        "src/visualization.py",
        "run_experiment.py",
    ]
    all_present = True
    for f in expected_files:
        path = os.path.join(PROJECT_DIR, f)
        if os.path.exists(path):
            print(f"  [OK] {f}")
        else:
            print(f"  [MISSING] {f}")
            all_present = False

    if all_present:
        print("\n  All project files present. Ready to run experiments.")
    else:
        print(f"\n  [WARNING] Some files missing. Check {PROJECT_DIR}")

    # Recommendation
    if not torch.cuda.is_available():
        print("\n  NOTE: No GPU detected. Grid-based analysis will work fine.")
        print("  For SplineCam exact tessellation, use a GPU runtime:")
        print("  Runtime > Change runtime type > T4 GPU")


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--post-restart', action='store_true',
                        help='Run post-restart steps (graph-tool completion)')
    parser.add_argument('--skip-graph-tool', action='store_true',
                        help='Skip graph-tool installation (grid analysis only)')
    parser.add_argument('--gpu-only', action='store_true',
                        help='Only proceed if GPU is available')
    args = parser.parse_args()

    import torch
    if args.gpu_only and not torch.cuda.is_available():
        print("ERROR: --gpu-only specified but no GPU available.")
        print("Change runtime: Runtime > Change runtime type > T4 GPU")
        sys.exit(1)

    # Set matplotlib backend early
    import matplotlib
    matplotlib.use('Agg')

    if args.post_restart:
        # Post-restart: complete graph-tool, re-setup splinecam
        install_splinecam()
        patch_splinecam_torchscript()
        patch_splinecam_multiprocessing()
        install_graph_tool_post_restart()
    else:
        # Fresh setup
        install_pip_deps()
        install_splinecam()
        patch_splinecam_torchscript()
        patch_splinecam_multiprocessing()

        if not args.skip_graph_tool:
            print("\n" + "="*60)
            print("graph-tool installation (optional)")
            print("="*60)
            print("graph-tool is only needed for SplineCam exact tessellation.")
            print("Grid-based analysis works without it.")
            print("To skip: rerun with --skip-graph-tool")
            print("")
            # Don't auto-install condacolab as it forces kernel restart
            # Just inform the user
            try:
                import graph_tool
                print("graph-tool already available.")
            except ImportError:
                print("graph-tool not installed. To install:")
                print("  1. Run: !pip install condacolab && import condacolab; condacolab.install()")
                print("  2. After kernel restart: %run colab_setup.py --post-restart")

    verify_environment()
    print("\nSetup complete.")


if __name__ == "__main__":
    main()

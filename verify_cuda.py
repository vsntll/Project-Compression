import os
import sys
import re
import glob

def _print_environment_details():
    """Prints critical details about the Python environment for debugging."""
    print("--- Python Environment Diagnostics ---")
    print(f"[*] Python Executable: {sys.executable}")
    print("\n[*] Python's Module Search Paths (sys.path):")
    # Indent the paths for readability
    for i, path in enumerate(sys.path):
        print(f"    {i}: {path}")
    print("--------------------------------------\n")


def _find_and_set_cuda_path():
    """
    Tries to find the CUDA Toolkit path and set environment variables.
    This is crucial for CuPy to locate the necessary CUDA libraries on Windows.
    Returns True on success, False on failure.
    """
    # This logic is primarily for Windows.
    if sys.platform != 'win32':
        return True

    # 1. Set CUDA_PATH
    if 'CUDA_PATH' not in os.environ:
        # Search in the default installation directory.
        possible_paths = glob.glob(r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v*")
        if possible_paths:
            latest_cuda_path = sorted(possible_paths)[-1]
            print(f"[INFO] Automatically setting CUDA_PATH to: {latest_cuda_path}")
            os.environ['CUDA_PATH'] = latest_cuda_path
        else:
            print("[ERROR] CUDA Toolkit not found in default location. Please set the 'CUDA_PATH' environment variable.")
            return False

    # 2. Add CUDA /bin directory to the system PATH for DLL loading
    cuda_bin_path = os.path.join(os.environ['CUDA_PATH'], 'bin')
    if cuda_bin_path not in os.environ['PATH']:
        print(f"[INFO] Adding CUDA bin to system PATH: {cuda_bin_path}")
        os.environ['PATH'] = cuda_bin_path + os.pathsep + os.environ['PATH']
    
    return True

def verify_installation():
    """
    Runs a series of checks to verify the CuPy and CUDA installation.
    """
    print("--- Starting CUDA and CuPy Verification ---")

    # Step 1: Set environment paths (essential on Windows)
    if not _find_and_set_cuda_path():
        print("\n[FAILURE] Could not configure CUDA paths. Halting verification.")
        return

    # Extract system CUDA version from path for later comparison
    system_cuda_major_version = None
    cuda_path = os.environ.get('CUDA_PATH', '')
    match = re.search(r'v(\d+)\.(\d+)', cuda_path)
    if match:
        system_cuda_major_version = int(match.group(1))

    if system_cuda_major_version:
        print(f"\n[INFO] Found system CUDA Toolkit v{system_cuda_major_version}")
    else:
        print("\n[WARNING] Could not determine system CUDA Toolkit version from path.")


    # Step 2: Try to import CuPy
    try:
        import cupy as cp
        print(f"\n[OK] CuPy imported successfully (Version: {cp.__version__})")
    except ImportError as e:
        print(f"\n[FAILURE] Failed to import CuPy. The 'cupy' module is not installed.")
        print(f"Error details: {e}")
        if system_cuda_major_version:
            print("\n" + "="*60)
            print(f"  [SOLUTION] Based on your system's CUDA v{system_cuda_major_version}, please install the correct CuPy package:")
            print(f"  pip install cupy-cuda{system_cuda_major_version}x")
            print("="*60 + "\n")
        else:
            print("\n[SOLUTION] Please install a CuPy package that matches your CUDA Toolkit version (e.g., 'pip install cupy-cuda12x' for CUDA 12).")
        return
    except Exception as e:
        print(f"\n[FAILURE] An unexpected error occurred during CuPy import. This might be a driver or toolkit mismatch.")
        print(f"Error details: {e}")
        return

    # Step 3: Check for available devices
    try:
        device_count = cp.cuda.runtime.getDeviceCount()
        if device_count == 0:
            print("\n[FAILURE] No CUDA-enabled devices were found by CuPy.")
            return

        # --- NEW: Version Mismatch Check ---
        cupy_linked_cuda_version = cp.cuda.runtime.runtimeGetVersion()
        cupy_linked_major_version = cupy_linked_cuda_version // 1000

        if system_cuda_major_version and system_cuda_major_version != cupy_linked_major_version:
            print("\n" + "="*60)
            print("[CRITICAL FAILURE] CUDA VERSION MISMATCH DETECTED!")
            print(f"  - Your system has CUDA Toolkit: v{system_cuda_major_version}")
            print(f"  - Your installed CuPy package is built for: v{cupy_linked_major_version}")
            print(f"\n  [SOLUTION] Please reinstall CuPy with the correct version:")
            print(f"  pip uninstall cupy-cuda{cupy_linked_major_version}x")
            print(f"  pip install cupy-cuda{system_cuda_major_version}x")
            print("="*60 + "\n")
            return # Halt further tests as they will likely fail

        print(f"\n[OK] Found {device_count} CUDA-enabled device(s).")
        print(f"  - CUDA Toolkit version linked with CuPy: {cp.cuda.runtime.runtimeGetVersion()}")
        print(f"  - NVIDIA Driver version: {cp.cuda.runtime.driverGetVersion()}")

        for i in range(device_count):
            device_props = cp.cuda.runtime.getDeviceProperties(i)
            print(f"\n--- Device {i} ---")
            print(f"  Name: {device_props['name'].decode()}")
            print(f"  Compute Capability: {device_props['major']}.{device_props['minor']}")
            print(f"  Total Memory: {device_props['totalGlobalMem'] / (1024**3):.2f} GB")

        # Step 4: Perform a simple test calculation
        print("\n--- Performing Test Calculation ---")
        x_gpu = cp.arange(10, dtype=cp.float32)
        y_gpu = x_gpu**2 + 5
        y_cpu = cp.asnumpy(y_gpu)
        print(f"  Input (GPU): {x_gpu}")
        print(f"  Result (GPU): {y_gpu}")
        print(f"  Result (CPU): {y_cpu}")
        print("\n[OK] Test calculation completed successfully.")

    except cp.cuda.runtime.CUDARuntimeError as e:
        print(f"\n[FAILURE] A CUDA runtime error occurred. This often indicates a driver/toolkit mismatch.")
        print(f"Error details: {e}")

    print("\n--- Verification Complete ---")

if __name__ == "__main__":
    _print_environment_details()
    verify_installation()

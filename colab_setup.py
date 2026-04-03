"""
Colab Setup Helper — Environment configuration for Google Colab.

Run this at the top of the Colab notebook to set up everything:
    exec(open('colab_setup.py').read())
"""
import subprocess
import sys
import os


def install_dependencies():
    """Install all required packages."""
    packages = [
        "xgboost", "lightgbm", "catboost", "optuna",
        "shap", "hdbscan", "imbalanced-learn",
        "scikit-learn", "pandas", "numpy",
        "matplotlib", "seaborn", "pyyaml", "scipy",
    ]
    print("[SETUP] Installing dependencies...")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--quiet"] + packages
    )
    print("[SETUP] ✓ All packages installed")


def check_gpu():
    """Check for GPU availability and return device info."""
    gpu_info = {"available": False, "name": "None", "memory": "N/A"}

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(",")
            gpu_info["available"] = True
            gpu_info["name"] = parts[0].strip()
            gpu_info["memory"] = parts[1].strip() if len(parts) > 1 else "Unknown"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    if not gpu_info["available"]:
        try:
            import torch
            if torch.cuda.is_available():
                gpu_info["available"] = True
                gpu_info["name"] = torch.cuda.get_device_name(0)
                mem = torch.cuda.get_device_properties(0).total_memory
                gpu_info["memory"] = f"{mem / (1024**3):.1f} GB"
        except ImportError:
            pass

    return gpu_info


def setup_paths(repo_name: str = "ha"):
    """
    Set up Python path for the project.

    Works for both:
    - Cloned repo: /content/ha/models/
    - Uploaded files: /content/models/
    """
    possible_roots = [
        f"/content/{repo_name}",
        "/content",
        os.getcwd(),
    ]

    for root in possible_roots:
        models_dir = os.path.join(root, "models")
        if os.path.isdir(models_dir):
            if models_dir not in sys.path:
                sys.path.insert(0, models_dir)
            print(f"[SETUP] ✓ Python path set: {models_dir}")
            return root

    print("[SETUP] ⚠ Could not find models/ directory")
    return os.getcwd()


def upload_dataset():
    """Upload a dataset via Colab's file picker."""
    try:
        from google.colab import files
        print("[UPLOAD] Select your CSV file to upload...")
        uploaded = files.upload()
        if uploaded:
            filename = list(uploaded.keys())[0]
            print(f"[UPLOAD] ✓ Uploaded: {filename} "
                  f"({len(uploaded[filename]):,} bytes)")
            return filename
        else:
            print("[UPLOAD] No file uploaded")
            return None
    except ImportError:
        print("[UPLOAD] Not running in Colab. Use file path directly.")
        return None


def download_results(output_dir: str = "results/hybrid_output"):
    """Download result files from Colab."""
    try:
        from google.colab import files
        for f in os.listdir(output_dir):
            filepath = os.path.join(output_dir, f)
            if os.path.isfile(filepath):
                print(f"[DOWNLOAD] {f}")
                files.download(filepath)
    except ImportError:
        print("[DOWNLOAD] Not running in Colab. Results are in:", output_dir)


def is_colab() -> bool:
    """Check if running in Google Colab."""
    try:
        import google.colab
        return True
    except ImportError:
        return False


def full_setup(repo_name: str = "ha"):
    """Run complete Colab setup."""
    print("=" * 50)
    print("  HYBRID FRAUD DETECTION — COLAB SETUP")
    print("=" * 50)

    # 1. Install deps
    install_dependencies()

    # 2. Check GPU
    gpu = check_gpu()
    print(f"\n[GPU] Available: {'✓ YES' if gpu['available'] else '✗ NO'}")
    if gpu["available"]:
        print(f"[GPU] Device: {gpu['name']}")
        print(f"[GPU] Memory: {gpu['memory']}")
    else:
        print("[GPU] Running in CPU mode (still works, just slower)")

    # 3. Setup paths
    print()
    root = setup_paths(repo_name)

    print(f"\n[SETUP] ✓ Setup complete!")
    print(f"[SETUP] Project root: {root}")
    print(f"[SETUP] Colab: {is_colab()}")

    return root, gpu


# Auto-run if executed directly
if __name__ == "__main__":
    full_setup()

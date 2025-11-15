#!/usr/bin/env python3
"""
Test script to simulate Hugging Face download process as it would happen in judge system.
This simulates a fresh environment with no local models.
"""
import sys
import os
from pathlib import Path

print("=" * 80)
print("TESTING HUGGING FACE DOWNLOAD (SIMULATING JUDGE SYSTEM)")
print("=" * 80)
print()

# Check repository setting from source code
ding_bot_path = Path(__file__).parent.resolve()
main_py_path = ding_bot_path / "src" / "main.py"

print("1. Reading repository configuration from src/main.py:")
with open(main_py_path, 'r') as f:
    content = f.read()
    # Extract HUGGINGFACE_MODEL_REPO value
    for line in content.split('\n'):
        if 'HUGGINGFACE_MODEL_REPO =' in line and not line.strip().startswith('#'):
            # Extract the value
            if "HUGGINGFACE_MODEL_REPO = " in line:
                repo_value = line.split("HUGGINGFACE_MODEL_REPO = ")[1].split("#")[0].strip().strip("'\"")
                print(f"   Found: HUGGINGFACE_MODEL_REPO = '{repo_value}'")
                HUGGINGFACE_MODEL_REPO = repo_value
                break
    else:
        print("   ❌ Could not find HUGGINGFACE_MODEL_REPO in src/main.py")
        sys.exit(1)

print()

if not HUGGINGFACE_MODEL_REPO or HUGGINGFACE_MODEL_REPO == 'YOUR_USERNAME/chess-bot-model':
    print("❌ ERROR: Repository not properly configured!")
    if HUGGINGFACE_MODEL_REPO == 'YOUR_USERNAME/chess-bot-model':
        print("   Fix: Replace 'YOUR_USERNAME/chess-bot-model' with your actual repo")
    else:
        print("   Fix: Set HUGGINGFACE_MODEL_REPO in src/main.py")
    sys.exit(1)

# Check if huggingface_hub is installed
print("2. Checking dependencies:")
try:
    import huggingface_hub
    print(f"   ✅ huggingface_hub installed (version: {huggingface_hub.__version__})")
except ImportError:
    print("   ❌ huggingface_hub not installed!")
    print("   Install with: pip install huggingface_hub")
    sys.exit(1)
print()

# Test download function directly
print("3. Testing download function (simulating judge system - no local models):")
test_output_dir = "test_download_models"
print(f"   Repository: {HUGGINGFACE_MODEL_REPO}")
print(f"   Output directory: {test_output_dir}")
print()

try:
    from huggingface_hub import hf_hub_download, list_repo_files
    
    print("   Step 1: Listing files in repository...")
    files = list_repo_files(repo_id=HUGGINGFACE_MODEL_REPO, repo_type="model")
    print(f"   Found {len(files)} files in repository")
    
    model_files = [f for f in files if f.endswith('.pth')]
    print(f"   Found {len(model_files)} .pth files")
    
    if not model_files:
        print(f"   ❌ ERROR: No .pth files found in {HUGGINGFACE_MODEL_REPO}")
        print(f"   Available files:")
        for f in files[:10]:  # Show first 10
            print(f"     - {f}")
        if len(files) > 10:
            print(f"     ... and {len(files) - 10} more")
        sys.exit(1)
    
    print(f"   Model files found:")
    for f in model_files:
        print(f"     - {f}")
    
    # Download latest model file
    target_file = sorted(model_files)[-1]
    print(f"\n   Step 2: Downloading {target_file}...")
    
    os.makedirs(test_output_dir, exist_ok=True)
    local_path = hf_hub_download(
        repo_id=HUGGINGFACE_MODEL_REPO,
        filename=target_file,
        local_dir=test_output_dir,
        local_dir_use_symlinks=False
    )
    
    if local_path and os.path.exists(local_path):
        file_size = os.path.getsize(local_path) / (1024 * 1024)  # MB
        print(f"   ✅ Download successful!")
        print(f"   File: {local_path}")
        print(f"   Size: {file_size:.1f} MB")
        print()
        print("=" * 80)
        print("✅ ALL TESTS PASSED - Hugging Face download works correctly!")
        print("=" * 80)
        print()
        print("The model will download automatically in the judge system.")
    else:
        print(f"   ❌ Download returned None or file doesn't exist")
        sys.exit(1)
        
except Exception as e:
    print(f"   ❌ Download failed with error: {e}")
    import traceback
    traceback.print_exc()
    print()
    print("Common issues:")
    print("1. Repository doesn't exist or is private")
    print("2. Repository name format incorrect (should be 'username/repo-name')")
    print("3. No .pth files in repository")
    print("4. Network/authentication issues")
    print()
    print(f"Repository tested: {HUGGINGFACE_MODEL_REPO}")
    print(f"Check: https://huggingface.co/{HUGGINGFACE_MODEL_REPO}")
    sys.exit(1)

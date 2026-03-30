#!/usr/bin/env python3
"""
Quick GPU diagnostic script to check PyTorch CUDA setup
"""
import sys

print("=" * 60)
print("GPU/CUDA Diagnostic Check")
print("=" * 60)

# 1. Check PyTorch
try:
    import torch
    print(f"\n✓ PyTorch installed: {torch.__version__}")
except ImportError:
    print("\n✗ PyTorch not installed!")
    sys.exit(1)

# 2. Check CUDA availability
print(f"\nCUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"✓ CUDA is working!")
    print(f"CUDA version (PyTorch): {torch.version.cuda}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")

    for i in range(torch.cuda.device_count()):
        print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
        print(f"  Compute Capability: {torch.cuda.get_device_properties(i).major}.{torch.cuda.get_device_properties(i).minor}")

    # Quick test
    print("\n" + "-" * 60)
    print("Running quick GPU test...")
    try:
        x = torch.rand(100, 100).cuda()
        y = torch.rand(100, 100).cuda()
        z = x @ y
        print("✓ GPU computation successful!")
    except Exception as e:
        print(f"✗ GPU test failed: {e}")
else:
    print("✗ CUDA is NOT available!")
    print("\nPossible reasons:")
    print("1. PyTorch installed without CUDA support (CPU-only version)")
    print("2. CUDA drivers not installed or incompatible")
    print("3. CUDA toolkit version mismatch")

    print("\n" + "-" * 60)
    print("Checking system CUDA installation...")

    # Try to check NVIDIA driver
    import subprocess
    try:
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("\n✓ NVIDIA driver is installed:")
            print(result.stdout)
            print("\nYour GPU is detected by the system, but PyTorch can't use it.")
            print("This means PyTorch was installed WITHOUT CUDA support.")
        else:
            print("\n✗ nvidia-smi command failed")
            print("NVIDIA drivers may not be installed")
    except FileNotFoundError:
        print("\n✗ nvidia-smi not found - NVIDIA drivers not installed")
    except Exception as e:
        print(f"\n✗ Error checking nvidia-smi: {e}")

# 3. Check cuDNN
if torch.cuda.is_available():
    print("\n" + "-" * 60)
    print(f"cuDNN available: {torch.backends.cudnn.is_available()}")
    if torch.backends.cudnn.is_available():
        print(f"cuDNN version: {torch.backends.cudnn.version()}")

print("\n" + "=" * 60)
if not torch.cuda.is_available():
    print("\n🔧 FIX: Uninstall current PyTorch and reinstall with CUDA support:")
    print("\nStep 1: Uninstall current PyTorch")
    print("  pip uninstall torch torchvision torchaudio")
    print("\nStep 2: Install PyTorch with CUDA 12.1")
    print("  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
    print("\nStep 3: Run this script again to verify")
else:
    print("\n✓ Everything looks good! Your GPU is ready for training.")
print("=" * 60)

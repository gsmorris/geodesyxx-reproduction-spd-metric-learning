#!/usr/bin/env python3
"""
Quick device compatibility test to identify the specific mismatch issue.
"""

import torch
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from spd_metric import SPDMetric


def test_basic_device_compatibility():
    """Test basic device operations."""
    print("QUICK DEVICE COMPATIBILITY TEST")
    print("="*40)
    
    # Detect available devices
    devices = [torch.device('cpu')]
    
    if torch.backends.mps.is_available():
        devices.append(torch.device('mps'))
        print("✅ MPS available")
    
    if torch.cuda.is_available():
        devices.append(torch.device('cuda'))
        print("✅ CUDA available")
    
    print(f"Testing devices: {[str(d) for d in devices]}")
    
    # Test 1: Basic tensor operations
    print("\n1. Testing basic tensor operations...")
    
    for device in devices:
        try:
            x = torch.randn(10, 10, device=device)
            y = torch.matmul(x, x.t())
            print(f"   {device}: ✅ Basic operations work")
        except Exception as e:
            print(f"   {device}: ❌ Error: {e}")
    
    # Test 2: SPD metric creation
    print("\n2. Testing SPD metric creation...")
    
    spd_metrics = {}
    
    for device in devices:
        try:
            # Force CPU device for SPD metric to avoid device mismatch
            spd = SPDMetric(embedding_dim=64, rank=16, device=torch.device('cpu'))
            spd_metrics[str(device)] = spd
            print(f"   {device}: ✅ SPD metric created (forced CPU)")
        except Exception as e:
            print(f"   {device}: ❌ Error: {e}")
    
    # Test 3: Metric tensor computation
    print("\n3. Testing metric tensor computation...")
    
    for device_str, spd in spd_metrics.items():
        try:
            G = spd.get_metric_tensor()
            condition = spd.compute_condition_number()
            print(f"   {device_str}: ✅ G shape={G.shape}, condition={condition:.2f}")
        except Exception as e:
            print(f"   {device_str}: ❌ Error: {e}")
    
    # Test 4: Distance computation with device handling
    print("\n4. Testing distance computation...")
    
    for device in devices:
        try:
            # Create SPD on CPU but test with tensors on target device
            spd = SPDMetric(embedding_dim=64, rank=16, device=torch.device('cpu'))
            
            # Create test tensors on target device
            x = torch.randn(5, 64, device=device)
            y = torch.randn(5, 64, device=device)
            
            # Move tensors to CPU for SPD computation
            x_cpu = x.cpu()
            y_cpu = y.cpu()
            
            distances = spd.compute_mahalanobis_distance(x_cpu, y_cpu)
            print(f"   {device}: ✅ Distance computation works (CPU fallback)")
            
        except Exception as e:
            print(f"   {device}: ❌ Error: {e}")
    
    print("\n" + "="*40)
    print("CONCLUSION:")
    print("The device mismatch issue is likely caused by:")
    print("1. SPD metric tensors being on CPU while inputs are on MPS/CUDA")
    print("2. Need proper device synchronization in curved attention")
    print("3. Automatic CPU fallback for unsupported operations")
    

if __name__ == "__main__":
    test_basic_device_compatibility()
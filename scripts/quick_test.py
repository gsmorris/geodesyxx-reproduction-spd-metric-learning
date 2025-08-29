#!/usr/bin/env python3
"""
Quick Test Script for Geodesyxx Package
Validates basic functionality and device compatibility.
Runs in under 30 seconds for rapid validation.
"""

import sys
import os
import time
from pathlib import Path
import warnings

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

def test_imports():
    """Test that all core modules can be imported."""
    print("🔍 Testing imports...")
    
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__}")
        
        # Test our core modules
        from spd_metric import SPDMetric
        from curved_attention import CurvedMultiHeadAttention
        from synthetic_validation import set_seed, create_synthetic_data
        print("✅ Core modules imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure you've run: pip install -r requirements.txt")
        return False

def test_device_compatibility():
    """Test device compatibility (MPS/CUDA/CPU)."""
    print("\n🔧 Testing device compatibility...")
    
    try:
        import torch
        
        # Test CPU
        x = torch.randn(10, 10, device='cpu')
        y = torch.matmul(x, x.t())
        print("✅ CPU working")
        
        # Test MPS (Apple Silicon)
        if torch.backends.mps.is_available():
            try:
                x = torch.randn(10, 10, device='mps')
                y = torch.matmul(x, x.t())
                print("✅ MPS working")
            except Exception as e:
                print(f"⚠️  MPS available but not working: {e}")
        else:
            print("ℹ️  MPS not available")
        
        # Test CUDA
        if torch.cuda.is_available():
            try:
                x = torch.randn(10, 10, device='cuda')
                y = torch.matmul(x, x.t())
                print(f"✅ CUDA working ({torch.cuda.get_device_name()})")
            except Exception as e:
                print(f"⚠️  CUDA available but not working: {e}")
        else:
            print("ℹ️  CUDA not available")
        
        return True
        
    except Exception as e:
        print(f"❌ Device compatibility test failed: {e}")
        return False

def test_spd_metric():
    """Test SPD metric implementation."""
    print("\n🧮 Testing SPD metric...")
    
    try:
        from spd_metric import SPDMetric
        import torch
        
        # Create metric
        metric = SPDMetric(embedding_dim=8, rank=4, device=torch.device('cpu'))
        
        # Test basic functionality
        batch_size = 4
        x = torch.randn(batch_size, 8)
        y = torch.randn(batch_size, 8)
        
        # Compute distances
        distances = metric(x, y)
        assert distances.shape == (batch_size,), f"Expected shape {(batch_size,)}, got {distances.shape}"
        assert torch.all(distances >= 0), "Distances should be non-negative"
        
        # Test condition number
        condition = metric.compute_condition_number()
        assert condition > 0, "Condition number should be positive"
        
        print(f"✅ SPD metric working (condition number: {condition:.2f})")
        return True
        
    except Exception as e:
        print(f"❌ SPD metric test failed: {e}")
        return False

def test_curved_attention():
    """Test curved attention implementation."""
    print("\n🎯 Testing curved attention...")
    
    try:
        from curved_attention import CurvedMultiHeadAttention
        import torch
        
        # Create attention layer
        attention = CurvedMultiHeadAttention(
            embed_dim=64,
            num_heads=4,
            geometry_mode='shared',
            rank=8,
            device=torch.device('cpu')
        )
        
        # Test forward pass
        batch_size, seq_len = 2, 16
        hidden_states = torch.randn(batch_size, seq_len, 64)
        
        outputs = attention(hidden_states)
        output = outputs[0]
        
        assert output.shape == (batch_size, seq_len, 64), f"Expected shape {(batch_size, seq_len, 64)}, got {output.shape}"
        
        # Test parameter counting
        param_counts = attention.get_parameter_count()
        assert 'geometric' in param_counts, "Should have geometric parameters"
        assert param_counts['geometric'] > 0, "Should have non-zero geometric parameters"
        
        print(f"✅ Curved attention working (geometric params: {param_counts['geometric']})")
        return True
        
    except Exception as e:
        print(f"❌ Curved attention test failed: {e}")
        return False

def test_synthetic_validation():
    """Test synthetic validation functionality."""
    print("\n🔬 Testing synthetic validation...")
    
    try:
        from synthetic_validation import set_seed, create_synthetic_data
        import numpy as np
        
        # Set seed for reproducibility
        set_seed(42)
        
        # Create small synthetic dataset
        data = create_synthetic_data(
            dimension=5,
            n_samples=50,
            true_eigenvalues=[5.0, 3.0, 2.0, 1.0, 1.0],
            n_triplets=100,
            seed=42
        )
        
        assert 'triplets' in data, "Should have triplets"
        assert 'embeddings' in data, "Should have embeddings"
        assert len(data['triplets']) == 100, f"Expected 100 triplets, got {len(data['triplets'])}"
        
        print("✅ Synthetic validation working")
        return True
        
    except Exception as e:
        print(f"❌ Synthetic validation test failed: {e}")
        return False

def test_configuration_loading():
    """Test configuration file loading."""
    print("\n📝 Testing configuration loading...")
    
    try:
        import yaml
        
        # Test loading configs
        config_dir = Path(__file__).parent.parent / 'configs'
        
        phase1_3_config = config_dir / 'phase1_3_config.yaml'
        phase4_config = config_dir / 'phase4_config.yaml'
        
        if phase1_3_config.exists():
            with open(phase1_3_config, 'r') as f:
                config = yaml.safe_load(f)
            assert 'synthetic_validation' in config, "Phase 1-3 config should have synthetic_validation"
            print("✅ Phase 1-3 config loaded")
        else:
            print("⚠️  Phase 1-3 config not found")
        
        if phase4_config.exists():
            with open(phase4_config, 'r') as f:
                config = yaml.safe_load(f)
            assert 'model' in config, "Phase 4 config should have model config"
            print("✅ Phase 4 config loaded")
        else:
            print("⚠️  Phase 4 config not found")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration loading test failed: {e}")
        return False

def test_package_structure():
    """Test package structure and file organization."""
    print("\n📦 Testing package structure...")
    
    try:
        base_dir = Path(__file__).parent.parent
        
        # Check essential directories
        essential_dirs = ['src', 'configs', 'scripts']
        for dir_name in essential_dirs:
            dir_path = base_dir / dir_name
            if dir_path.exists():
                print(f"✅ {dir_name}/ directory exists")
            else:
                print(f"⚠️  {dir_name}/ directory missing")
        
        # Check essential files
        essential_files = ['README.md', 'requirements.txt', 'LICENSE']
        for file_name in essential_files:
            file_path = base_dir / file_name
            if file_path.exists():
                print(f"✅ {file_name} exists")
            else:
                print(f"⚠️  {file_name} missing")
        
        # Check src module files
        src_files = ['spd_metric.py', 'curved_attention.py', '__init__.py']
        src_dir = base_dir / 'src'
        for file_name in src_files:
            file_path = src_dir / file_name
            if file_path.exists():
                print(f"✅ src/{file_name} exists")
            else:
                print(f"⚠️  src/{file_name} missing")
        
        return True
        
    except Exception as e:
        print(f"❌ Package structure test failed: {e}")
        return False

def main():
    """Run all quick tests."""
    start_time = time.time()
    
    print("🚀 Running Geodesyxx Quick Test Suite")
    print("=" * 50)
    
    tests = [
        ("Package Structure", test_package_structure),
        ("Imports", test_imports),
        ("Device Compatibility", test_device_compatibility),
        ("SPD Metric", test_spd_metric),
        ("Curved Attention", test_curved_attention),
        ("Synthetic Validation", test_synthetic_validation),
        ("Configuration Loading", test_configuration_loading),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # Suppress warnings during testing
                success = test_func()
                results.append((test_name, success))
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    elapsed = time.time() - start_time
    print("\n" + "=" * 50)
    print("QUICK TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\nResults: {passed}/{total} tests passed")
    print(f"Runtime: {elapsed:.1f}s")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ Geodesyxx package is ready for use")
        
        print("\n📋 Next steps:")
        print("1. Run synthetic validation: python scripts/run_synthetic_validation.py")
        print("2. Run Phase 1-3: python scripts/run_phase1_3.py")
        print("3. Run Phase 4: python scripts/run_phase4.py")
        
    else:
        print("\n⚠️  Some tests failed")
        print("💡 Check the errors above and ensure all requirements are installed")
        
        failed_tests = [name for name, success in results if not success]
        print(f"Failed tests: {', '.join(failed_tests)}")
    
    return passed == total

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
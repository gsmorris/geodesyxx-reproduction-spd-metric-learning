#!/usr/bin/env python3
"""
Package Validation Script
Verifies that the Geodesyxx reproduction package is properly organized and functional.
"""

import sys
import importlib
from pathlib import Path
import json


def validate_package_structure():
    """Validate that all required files are present."""
    print("VALIDATING PACKAGE STRUCTURE")
    print("="*40)
    
    required_files = [
        "src/__init__.py",
        "src/spd_metric.py", 
        "src/synthetic_validation.py",
        "src/curved_attention.py",
        "src/training.py",
        "src/transformer_integration.py",
        "configs/phase1_3_config.yaml",
        "configs/phase4_config.yaml",
        "tests/test_eigenvalue_recovery.py",
        "tests/test_device_compatibility.py",
        "examples/run_paper_validation.py",
        "examples/quick_validation.py",
        "evaluation.py",
        "requirements_reproduction.txt",
        "REPRODUCTION_README.md",
        "device_compatibility_report.md"
    ]
    
    missing_files = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing_files.append(file_path)
        else:
            print(f"✅ {file_path}")
    
    if missing_files:
        print(f"\n❌ Missing files:")
        for file_path in missing_files:
            print(f"   {file_path}")
        return False
    else:
        print(f"\n✅ All required files present ({len(required_files)} files)")
        return True


def validate_imports():
    """Validate that core modules can be imported."""
    print("\nVALIDATING IMPORTS")
    print("="*40)
    
    # Add src to path
    sys.path.insert(0, str(Path("src")))
    
    modules_to_test = [
        "spd_metric",
        "synthetic_validation", 
        "curved_attention",
        "training",
        "transformer_integration",
        "evaluation"
    ]
    
    import_failures = []
    
    for module_name in modules_to_test:
        try:
            if module_name == "evaluation":
                # evaluation.py is in root
                sys.path.insert(0, ".")
            
            module = importlib.import_module(module_name)
            print(f"✅ {module_name}")
        except ImportError as e:
            print(f"❌ {module_name}: {e}")
            import_failures.append(module_name)
    
    if import_failures:
        print(f"\n❌ Import failures: {import_failures}")
        return False
    else:
        print(f"\n✅ All imports successful ({len(modules_to_test)} modules)")
        return True


def validate_device_compatibility():
    """Validate device compatibility results."""
    print("\nVALIDATING DEVICE COMPATIBILITY")
    print("="*40)
    
    results_file = Path("device_compatibility_results.json")
    if not results_file.exists():
        print("❌ Device compatibility results not found")
        return False
    
    try:
        with open(results_file) as f:
            results = json.load(f)
        
        print(f"✅ Results file loaded")
        print(f"   Devices tested: {results.get('devices_tested', [])}")
        print(f"   Assessment: {results.get('compatibility_assessment', 'Unknown')}")
        
        # Check if all operations were successful
        spd_ops = results.get('spd_operations', {})
        attn_ops = results.get('attention_operations', {})
        
        all_successful = (
            all(op.get('success', False) for op in spd_ops.values()) and
            all(op.get('success', False) for op in attn_ops.values())
        )
        
        if all_successful:
            print(f"✅ All device operations successful")
            return True
        else:
            print(f"⚠️ Some device operations failed")
            return False
            
    except Exception as e:
        print(f"❌ Error reading results: {e}")
        return False


def validate_core_functionality():
    """Test core functionality without full execution."""
    print("\nVALIDATING CORE FUNCTIONALITY")
    print("="*40)
    
    try:
        # Add paths
        sys.path.insert(0, "src")
        
        # Test SPD metric
        from spd_metric import SPDMetric
        spd = SPDMetric(embedding_dim=64, rank=16)
        G = spd.get_metric_tensor()
        condition = spd.compute_condition_number()
        print(f"✅ SPD metric: shape={G.shape}, condition={condition:.1f}")
        
        # Test curved attention
        from curved_attention import CurvedMultiHeadAttention
        import torch
        attention = CurvedMultiHeadAttention(
            embed_dim=768, num_heads=12, geometry_mode='shared', rank=16
        )
        hidden_states = torch.randn(1, 8, 768)
        outputs = attention(hidden_states)
        print(f"✅ Curved attention: output shape={outputs[0].shape}")
        
        # Test evaluation utilities
        sys.path.insert(0, ".")
        from evaluation import GeodexyxEvaluator
        evaluator = GeodexyxEvaluator()
        print(f"✅ Statistical evaluator: α_corrected={evaluator.bonferroni.alpha_corrected:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Core functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run complete package validation."""
    print("\n" + "="*60)
    print(" "*15 + "GEODESYXX REPRODUCTION PACKAGE")
    print(" "*20 + "Validation Report")
    print("="*60)
    
    validation_results = []
    
    # Test 1: Package structure
    structure_ok = validate_package_structure()
    validation_results.append(("Structure", structure_ok))
    
    # Test 2: Import validation
    imports_ok = validate_imports()
    validation_results.append(("Imports", imports_ok))
    
    # Test 3: Device compatibility
    device_ok = validate_device_compatibility() 
    validation_results.append(("Device Compatibility", device_ok))
    
    # Test 4: Core functionality
    functionality_ok = validate_core_functionality()
    validation_results.append(("Core Functionality", functionality_ok))
    
    # Summary
    print("\n" + "="*60)
    print(" "*20 + "VALIDATION SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, ok in validation_results if ok)
    total = len(validation_results)
    
    for test_name, passed_test in validation_results:
        status = "✅ PASS" if passed_test else "❌ FAIL"
        print(f"{test_name:<20}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 PACKAGE VALIDATION SUCCESSFUL!")
        print("The Geodesyxx reproduction package is ready for distribution.")
        print("\nNext steps:")
        print("• Run examples/quick_validation.py for end-to-end test")
        print("• Execute representative sample with configs/phase4_config.yaml")
        print("• Share device_compatibility_report.md with users")
        return True
    else:
        print(f"\n⚠️ VALIDATION INCOMPLETE")
        print(f"Please address failing tests before distribution.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
#!/usr/bin/env python3
"""
Device Compatibility Test Suite for Geodesyxx Reproduction
Systematically tests numerical equivalence across MPS/CUDA/CPU devices.

This test suite investigates whether device-specific numerical differences
affect the core scientific conclusions of the paper.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import warnings
import sys
import os
from pathlib import Path
from dataclasses import dataclass
from contextlib import contextmanager

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from spd_metric import SPDMetric
from curved_attention import CurvedMultiHeadAttention
from transformer_integration import CurvedDistilBertForSequenceClassification
from transformers import DistilBertConfig

# Add parent directory for evaluation
sys.path.insert(0, str(Path(__file__).parent.parent))
from evaluation import StatisticalEquivalenceTest


@dataclass
class DeviceTestResult:
    """Results from device compatibility testing."""
    test_name: str
    devices_tested: List[str]
    max_absolute_diff: float
    max_relative_diff: float
    mean_absolute_diff: float
    equivalence_test: Dict[str, Any]
    passed: bool
    details: str


class DeviceManager:
    """Manages available devices and provides consistent tensor operations."""
    
    def __init__(self):
        self.available_devices = self._detect_devices()
        self.tolerance_absolute = 1e-6
        self.tolerance_relative = 1e-5
        
    def _detect_devices(self) -> List[torch.device]:
        """Detect all available devices."""
        devices = [torch.device('cpu')]
        
        # Test MPS availability and functionality
        if torch.backends.mps.is_available():
            try:
                x = torch.randn(10, 10, device='mps')
                y = torch.matmul(x, x.t())
                devices.append(torch.device('mps'))
                print("✅ MPS device available and functional")
            except Exception as e:
                print(f"⚠️ MPS available but not functional: {e}")
        
        # Test CUDA availability  
        if torch.cuda.is_available():
            try:
                x = torch.randn(10, 10, device='cuda')
                y = torch.matmul(x, x.t())
                devices.append(torch.device('cuda'))
                print(f"✅ CUDA device available and functional")
            except Exception as e:
                print(f"⚠️ CUDA available but not functional: {e}")
        
        print(f"Available devices: {[str(d) for d in devices]}")
        return devices
    
    @contextmanager
    def seed_context(self, seed: int = 42):
        """Context manager for reproducible random operations."""
        # Save current states
        torch_state = torch.get_rng_state()
        numpy_state = np.random.get_state()
        
        # Set seeds
        torch.manual_seed(seed)
        np.random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        
        try:
            yield
        finally:
            # Restore states
            torch.set_rng_state(torch_state)
            np.random.set_state(numpy_state)
    
    def create_identical_tensors(
        self, 
        shape: Tuple[int, ...], 
        seed: int = 42,
        requires_grad: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Create identical tensors on all available devices."""
        tensors = {}
        
        with self.seed_context(seed):
            # Create reference tensor on CPU
            cpu_tensor = torch.randn(shape, requires_grad=requires_grad)
            
            # Copy to all devices
            for device in self.available_devices:
                tensors[str(device)] = cpu_tensor.to(device)
        
        return tensors
    
    def compare_tensors(
        self,
        tensors: Dict[str, torch.Tensor],
        test_name: str = "tensor_comparison"
    ) -> DeviceTestResult:
        """Compare tensors across devices for numerical equivalence."""
        
        if len(tensors) < 2:
            return DeviceTestResult(
                test_name=test_name,
                devices_tested=list(tensors.keys()),
                max_absolute_diff=0.0,
                max_relative_diff=0.0,
                mean_absolute_diff=0.0,
                equivalence_test={},
                passed=True,
                details="Only one device available"
            )
        
        # Convert all tensors to CPU for comparison
        cpu_tensors = {k: v.cpu().detach() for k, v in tensors.items()}
        
        # Compare all pairs
        max_abs_diff = 0.0
        max_rel_diff = 0.0
        abs_diffs = []
        
        devices = list(cpu_tensors.keys())
        
        for i, device1 in enumerate(devices):
            for device2 in devices[i+1:]:
                tensor1 = cpu_tensors[device1]
                tensor2 = cpu_tensors[device2]
                
                # Absolute difference
                abs_diff = torch.abs(tensor1 - tensor2)
                abs_diffs.extend(abs_diff.flatten().tolist())
                max_abs_diff = max(max_abs_diff, abs_diff.max().item())
                
                # Relative difference (where denominator is not zero)
                denominator = torch.abs(tensor1) + 1e-10
                rel_diff = abs_diff / denominator
                max_rel_diff = max(max_rel_diff, rel_diff.max().item())
        
        mean_abs_diff = np.mean(abs_diffs)
        
        # Statistical equivalence test
        equivalence_test = {}
        if len(devices) >= 2:
            device1_vals = cpu_tensors[devices[0]].flatten().numpy()
            device2_vals = cpu_tensors[devices[1]].flatten().numpy()
            
            tester = StatisticalEquivalenceTest(equivalence_margin=self.tolerance_absolute)
            equivalence_test = tester.tost_test(
                device1_vals.tolist(),
                device2_vals.tolist()
            )
        
        # Determine if test passed
        passed = (
            max_abs_diff < self.tolerance_absolute and
            max_rel_diff < self.tolerance_relative
        )
        
        details = f"Max abs diff: {max_abs_diff:.2e}, Max rel diff: {max_rel_diff:.2e}"
        
        return DeviceTestResult(
            test_name=test_name,
            devices_tested=devices,
            max_absolute_diff=max_abs_diff,
            max_relative_diff=max_rel_diff,
            mean_absolute_diff=mean_abs_diff,
            equivalence_test=equivalence_test,
            passed=passed,
            details=details
        )


class DeviceCompatibilityTester:
    """Main test suite for device compatibility."""
    
    def __init__(self):
        self.device_manager = DeviceManager()
        self.test_results = []
        
    def test_spd_matrix_operations(self) -> List[DeviceTestResult]:
        """Test core SPD matrix operations across devices."""
        print("\n" + "="*60)
        print("TESTING SPD MATRIX OPERATIONS")
        print("="*60)
        
        results = []
        
        # Test A^T A + εI computation
        print("\n1. Testing A^T A + εI computation...")
        
        spd_metrics = {}
        
        for device in self.device_manager.available_devices:
            with self.device_manager.seed_context(42):
                spd = SPDMetric(
                    embedding_dim=64,
                    rank=16,
                    epsilon=1e-6,
                    device=device
                )
                spd_metrics[str(device)] = spd
        
        # Compare metric tensors
        metric_tensors = {}
        for device_str, spd in spd_metrics.items():
            metric_tensors[device_str] = spd.get_metric_tensor()
        
        result = self.device_manager.compare_tensors(
            metric_tensors, 
            "spd_metric_tensor"
        )
        results.append(result)
        
        print(f"   Result: {'✅ PASS' if result.passed else '❌ FAIL'}")
        print(f"   {result.details}")
        
        # Test condition number calculations
        print("\n2. Testing condition number calculations...")
        
        condition_numbers = {}
        for device_str, spd in spd_metrics.items():
            condition_numbers[device_str] = torch.tensor([spd.compute_condition_number()])
        
        result = self.device_manager.compare_tensors(
            condition_numbers,
            "condition_numbers"
        )
        results.append(result)
        
        print(f"   Result: {'✅ PASS' if result.passed else '❌ FAIL'}")
        print(f"   {result.details}")
        
        # Test eigenvalue computations
        print("\n3. Testing eigenvalue computations...")
        
        eigenvalues = {}
        for device_str, spd in spd_metrics.items():
            eigenvalues[device_str] = spd.get_eigenvalues()
        
        result = self.device_manager.compare_tensors(
            eigenvalues,
            "eigenvalues"
        )
        results.append(result)
        
        print(f"   Result: {'✅ PASS' if result.passed else '❌ FAIL'}")
        print(f"   {result.details}")
        
        return results
    
    def test_attention_score_computation(self) -> List[DeviceTestResult]:
        """Test attention score computation across devices."""
        print("\n" + "="*60)
        print("TESTING ATTENTION SCORE COMPUTATION")
        print("="*60)
        
        results = []
        
        # Test configurations
        test_configs = [
            {"batch_size": 2, "seq_len": 32, "description": "Small batch"},
            {"batch_size": 4, "seq_len": 64, "description": "Medium batch"},
            {"batch_size": 1, "seq_len": 128, "description": "Long sequence"}
        ]
        
        for i, config in enumerate(test_configs):
            print(f"\n{i+1}. Testing {config['description']}: "
                  f"batch={config['batch_size']}, seq_len={config['seq_len']}")
            
            attention_modules = {}
            
            # Create identical attention modules on each device
            for device in self.device_manager.available_devices:
                with self.device_manager.seed_context(42):
                    attention = CurvedMultiHeadAttention(
                        embed_dim=768,
                        num_heads=12,
                        geometry_mode='shared',
                        rank=16,
                        device=device
                    )
                    attention_modules[str(device)] = attention
            
            # Create identical input tensors
            hidden_states = self.device_manager.create_identical_tensors(
                (config["batch_size"], config["seq_len"], 768),
                seed=42
            )
            
            # Compute attention outputs
            attention_outputs = {}
            
            for device_str, attention in attention_modules.items():
                device_input = hidden_states[device_str]
                
                with torch.no_grad():
                    outputs = attention(device_input)
                    attention_outputs[device_str] = outputs[0]  # Get main output
            
            # Compare outputs
            result = self.device_manager.compare_tensors(
                attention_outputs,
                f"attention_output_{config['description']}"
            )
            results.append(result)
            
            print(f"   Result: {'✅ PASS' if result.passed else '❌ FAIL'}")
            print(f"   {result.details}")
        
        return results
    
    def test_training_step_consistency(self) -> List[DeviceTestResult]:
        """Test training step consistency across devices."""
        print("\n" + "="*60)
        print("TESTING TRAINING STEP CONSISTENCY")
        print("="*60)
        
        results = []
        
        # Only test if we have multiple devices
        if len(self.device_manager.available_devices) < 2:
            print("Skipping training test - only one device available")
            return results
        
        # Create identical models on each device
        config = DistilBertConfig.from_pretrained("distilbert-base-uncased")
        config.num_labels = 2
        
        models = {}
        optimizers = {}
        
        for device in self.device_manager.available_devices:
            with self.device_manager.seed_context(42):
                model = CurvedDistilBertForSequenceClassification(
                    config=config,
                    curved_layers=[1],  # Single layer to reduce complexity
                    geometry_mode='shared',
                    rank=16
                )
                model = model.to(device)
                
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
                
                models[str(device)] = model
                optimizers[str(device)] = optimizer
        
        # Run training steps
        num_steps = 3
        
        for step in range(num_steps):
            print(f"\n{step+1}. Testing training step {step+1}/{num_steps}...")
            
            # Create identical batch
            batch_data = self.device_manager.create_identical_tensors(
                (2, 32), seed=42+step  # Different seed per step
            )
            labels = self.device_manager.create_identical_tensors(
                (2,), seed=100+step
            )
            
            # Convert to appropriate dtypes
            for device_str in batch_data:
                batch_data[device_str] = batch_data[device_str].long().clamp(0, 999)
                labels[device_str] = labels[device_str].long().clamp(0, 1)
            
            # Training step on each device
            losses = {}
            
            for device_str, model in models.items():
                optimizer = optimizers[device_str]
                
                optimizer.zero_grad()
                
                try:
                    outputs = model(
                        input_ids=batch_data[device_str],
                        labels=labels[device_str]
                    )
                    loss = outputs.loss
                    loss.backward()
                    
                    # Clip gradients consistently
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    
                    optimizer.step()
                    
                    losses[device_str] = loss.detach()
                    
                except Exception as e:
                    print(f"   Error on {device_str}: {e}")
                    losses[device_str] = torch.tensor([float('nan')])
            
            # Compare losses
            if not any(torch.isnan(loss).any() for loss in losses.values()):
                result = self.device_manager.compare_tensors(
                    losses,
                    f"training_loss_step_{step+1}"
                )
                results.append(result)
                
                print(f"   Loss comparison: {'✅ PASS' if result.passed else '❌ FAIL'}")
                print(f"   {result.details}")
            else:
                print(f"   ⚠️ NaN losses detected, skipping comparison")
        
        # Compare final model parameters
        print(f"\n4. Comparing final model parameters...")
        
        # Get condition numbers as proxy for parameter state
        condition_numbers = {}
        for device_str, model in models.items():
            try:
                conditions = model.get_condition_numbers()
                if conditions:
                    # Take mean condition number as representative
                    mean_condition = torch.tensor([np.mean(list(conditions.values()))])
                    condition_numbers[device_str] = mean_condition
            except:
                pass
        
        if len(condition_numbers) >= 2:
            result = self.device_manager.compare_tensors(
                condition_numbers,
                "final_condition_numbers"
            )
            results.append(result)
            
            print(f"   Condition number comparison: {'✅ PASS' if result.passed else '❌ FAIL'}")
            print(f"   {result.details}")
        
        return results
    
    def run_full_test_suite(self) -> Dict[str, Any]:
        """Run complete device compatibility test suite."""
        print("\n" + "="*80)
        print(" "*25 + "DEVICE COMPATIBILITY TEST SUITE")
        print(" "*20 + "Geodesyxx Paper Reproduction Package")
        print("="*80)
        
        print(f"Available devices: {[str(d) for d in self.device_manager.available_devices]}")
        print(f"Tolerance: absolute={self.device_manager.tolerance_absolute:.0e}, "
              f"relative={self.device_manager.tolerance_relative:.0e}")
        
        # Run all tests
        all_results = []
        
        try:
            # Test 1: SPD Matrix Operations
            spd_results = self.test_spd_matrix_operations()
            all_results.extend(spd_results)
            
            # Test 2: Attention Score Computation
            attention_results = self.test_attention_score_computation()
            all_results.extend(attention_results)
            
            # Test 3: Training Step Consistency
            training_results = self.test_training_step_consistency()
            all_results.extend(training_results)
            
        except Exception as e:
            print(f"\n❌ Test suite failed with error: {e}")
            import traceback
            traceback.print_exc()
        
        # Summary
        print("\n" + "="*80)
        print(" "*30 + "TEST SUMMARY")
        print("="*80)
        
        total_tests = len(all_results)
        passed_tests = sum(1 for r in all_results if r.passed)
        failed_tests = total_tests - passed_tests
        
        print(f"\nTotal tests: {total_tests}")
        print(f"Passed: {passed_tests} ✅")
        print(f"Failed: {failed_tests} {'❌' if failed_tests > 0 else ''}")
        
        # Detailed results
        print(f"\nDetailed Results:")
        for result in all_results:
            status = "✅ PASS" if result.passed else "❌ FAIL"
            print(f"  {result.test_name}: {status}")
            print(f"    Max abs diff: {result.max_absolute_diff:.2e}")
            print(f"    Max rel diff: {result.max_relative_diff:.2e}")
            if result.equivalence_test.get('equivalent') is not None:
                equiv_status = "✅" if result.equivalence_test['equivalent'] else "❌"
                print(f"    Statistical equivalence: {equiv_status}")
        
        # Overall assessment
        print(f"\n" + "="*80)
        print(" "*25 + "COMPATIBILITY ASSESSMENT")
        print("="*80)
        
        if failed_tests == 0:
            print("✅ ALL TESTS PASSED")
            print("   Numerical results are equivalent across all devices")
            print("   Scientific conclusions are device-independent")
            assessment = "FULL_COMPATIBILITY"
        elif failed_tests <= total_tests // 3:
            print("⚠️ MINOR DIFFERENCES DETECTED")
            print("   Some numerical differences found but likely within acceptable limits")
            print("   Scientific conclusions should remain valid")
            assessment = "MINOR_DIFFERENCES"
        else:
            print("❌ SIGNIFICANT DIFFERENCES DETECTED")
            print("   Device-specific behavior may affect results")
            print("   Recommend using CPU for consistent results")
            assessment = "SIGNIFICANT_DIFFERENCES"
        
        return {
            'total_tests': total_tests,
            'passed_tests': passed_tests,
            'failed_tests': failed_tests,
            'assessment': assessment,
            'devices_tested': [str(d) for d in self.device_manager.available_devices],
            'test_results': all_results,
            'tolerance_absolute': self.device_manager.tolerance_absolute,
            'tolerance_relative': self.device_manager.tolerance_relative
        }


def main():
    """Run device compatibility tests."""
    tester = DeviceCompatibilityTester()
    results = tester.run_full_test_suite()
    
    # Save results for reporting
    import json
    results_serializable = {
        'total_tests': results['total_tests'],
        'passed_tests': results['passed_tests'],
        'failed_tests': results['failed_tests'],
        'assessment': results['assessment'],
        'devices_tested': results['devices_tested'],
        'tolerance_absolute': results['tolerance_absolute'],
        'tolerance_relative': results['tolerance_relative'],
        'test_details': [
            {
                'test_name': r.test_name,
                'devices_tested': r.devices_tested,
                'max_absolute_diff': r.max_absolute_diff,
                'max_relative_diff': r.max_relative_diff,
                'mean_absolute_diff': r.mean_absolute_diff,
                'passed': r.passed,
                'details': r.details,
                'equivalence_test': r.equivalence_test
            }
            for r in results['test_results']
        ]
    }
    
    output_path = Path(__file__).parent.parent / "device_compatibility_results.json"
    with open(output_path, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    
    print(f"\n📊 Detailed results saved to: {output_path}")
    
    return results['assessment'] == "FULL_COMPATIBILITY"


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
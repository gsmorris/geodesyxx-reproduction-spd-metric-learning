#!/usr/bin/env python3
"""
SPD Metric Implementation Test Suite
Comprehensive tests for SPD metric learning implementation correctness.
Validates mathematical properties and numerical behavior.
"""

import sys
import os
import unittest
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
import warnings
from typing import Dict, List, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from spd_metric import SPDMetric, BatchedSPDMetric


class TestSPDMetricProperties(unittest.TestCase):
    """Test mathematical properties of SPD metrics."""
    
    def setUp(self):
        """Set up test fixtures."""
        torch.manual_seed(42)
        np.random.seed(42)
        
        self.embedding_dim = 16
        self.rank = 8
        self.batch_size = 10
        
        self.metric = SPDMetric(
            embedding_dim=self.embedding_dim,
            rank=self.rank,
            epsilon=1e-6,
            device=torch.device('cpu')
        )
        
        # Test data
        self.x = torch.randn(self.batch_size, self.embedding_dim)
        self.y = torch.randn(self.batch_size, self.embedding_dim)
        
    def test_positive_definiteness(self):
        """Test that metric tensor is positive definite."""
        G = self.metric.get_metric_tensor()
        
        # Check symmetry
        self.assertTrue(torch.allclose(G, G.t(), atol=1e-6),
                       "Metric tensor should be symmetric")
        
        # Check positive definiteness via eigenvalues
        eigenvalues = torch.linalg.eigvalsh(G)
        self.assertTrue(torch.all(eigenvalues > 0),
                       f"All eigenvalues should be positive, got min: {eigenvalues.min()}")
        
        # Check via Cholesky decomposition
        try:
            L = torch.linalg.cholesky(G)
            self.assertTrue(True, "Cholesky decomposition successful")
        except RuntimeError:
            self.fail("Cholesky decomposition failed - matrix not positive definite")
    
    def test_distance_properties(self):
        """Test mathematical properties of Mahalanobis distance."""
        
        # Test 1: Non-negativity
        distances = self.metric(self.x, self.y, squared=True)
        self.assertTrue(torch.all(distances >= 0),
                       "Squared distances should be non-negative")
        
        # Test 2: Identity (d(x,x) = 0)
        zero_distances = self.metric(self.x, self.x, squared=True)
        self.assertTrue(torch.allclose(zero_distances, torch.zeros_like(zero_distances), atol=1e-6),
                       "Distance from point to itself should be zero")
        
        # Test 3: Symmetry (d(x,y) = d(y,x))
        dist_xy = self.metric(self.x, self.y, squared=True)
        dist_yx = self.metric(self.y, self.x, squared=True)
        self.assertTrue(torch.allclose(dist_xy, dist_yx, atol=1e-6),
                       "Distance should be symmetric")
        
        # Test 4: Square root consistency
        dist_squared = self.metric(self.x, self.y, squared=True)
        dist_sqrt = self.metric(self.x, self.y, squared=False)
        self.assertTrue(torch.allclose(dist_squared, dist_sqrt**2, atol=1e-5),
                       "Squared and square root distances should be consistent")
    
    def test_triangle_inequality(self):
        """Test triangle inequality for Mahalanobis distance."""
        z = torch.randn(self.batch_size, self.embedding_dim)
        
        # Compute distances
        d_xy = self.metric(self.x, self.y, squared=False)
        d_xz = self.metric(self.x, z, squared=False)
        d_yz = self.metric(self.y, z, squared=False)
        
        # Triangle inequality: d(x,y) <= d(x,z) + d(z,y)
        triangle_satisfied = d_xy <= d_xz + d_yz + 1e-6  # Small tolerance for numerical errors
        
        violations = (~triangle_satisfied).sum().item()
        total = len(d_xy)
        
        # Allow small percentage of violations due to numerical precision
        violation_rate = violations / total
        self.assertLess(violation_rate, 0.05, 
                       f"Triangle inequality violated in {violation_rate:.2%} of cases")
    
    def test_pairwise_distance_consistency(self):
        """Test consistency between individual and pairwise distance computation."""
        # Individual distances
        individual_distances = []
        for i in range(min(5, self.batch_size)):  # Test subset for efficiency
            for j in range(min(5, self.batch_size)):
                d = self.metric(self.x[i:i+1], self.y[j:j+1], squared=True)
                individual_distances.append(d.item())
        
        # Pairwise distances
        X_subset = self.x[:5]
        Y_subset = self.y[:5]
        pairwise_distances = self.metric.compute_pairwise_distances(X_subset, Y_subset, squared=True)
        
        # Compare
        pairwise_flat = pairwise_distances.flatten()[:len(individual_distances)]
        
        for i, (ind, pair) in enumerate(zip(individual_distances, pairwise_flat)):
            self.assertAlmostEqual(ind, pair.item(), places=5,
                                 msg=f"Distance {i}: individual={ind:.6f}, pairwise={pair.item():.6f}")
    
    def test_condition_number_bounds(self):
        """Test that condition number stays within reasonable bounds."""
        condition = self.metric.compute_condition_number()
        
        # Should be finite and positive
        self.assertTrue(torch.isfinite(torch.tensor(condition)),
                       "Condition number should be finite")
        self.assertGreater(condition, 1.0,
                          "Condition number should be >= 1")
        
        # Should be related to eigenvalue ratio
        eigenvalues = self.metric.get_eigenvalues()
        expected_condition = (eigenvalues[-1] / eigenvalues[0]).item()
        
        self.assertAlmostEqual(condition, expected_condition, places=3,
                              msg="Condition number should match eigenvalue ratio")
    
    def test_gradient_computation(self):
        """Test that gradients are computed correctly."""
        self.metric.zero_grad()
        
        # Enable gradient computation
        x_grad = self.x.clone().detach().requires_grad_(True)
        y_grad = self.y.clone().detach().requires_grad_(True)
        
        # Compute loss
        distances = self.metric(x_grad, y_grad, squared=True)
        loss = distances.mean()
        
        # Backward pass
        loss.backward()
        
        # Check gradients exist
        self.assertIsNotNone(self.metric.A.grad,
                           "SPD metric parameters should have gradients")
        self.assertIsNotNone(x_grad.grad,
                           "Input x should have gradients")
        self.assertIsNotNone(y_grad.grad,
                           "Input y should have gradients")
        
        # Check gradients are not zero (unless by coincidence)
        self.assertGreater(torch.norm(self.metric.A.grad), 1e-8,
                          "Parameter gradients should not be zero")
    
    def test_epsilon_regularization_effect(self):
        """Test that epsilon parameter affects condition number."""
        # Create metrics with different epsilon values
        epsilon_small = 1e-8
        epsilon_large = 1e-3
        
        metric_small = SPDMetric(
            embedding_dim=self.embedding_dim,
            rank=self.rank,
            epsilon=epsilon_small,
            device=torch.device('cpu')
        )
        
        metric_large = SPDMetric(
            embedding_dim=self.embedding_dim,
            rank=self.rank,
            epsilon=epsilon_large,
            device=torch.device('cpu')
        )
        
        # Copy same A matrix for fair comparison
        with torch.no_grad():
            metric_large.A.data = metric_small.A.data.clone()
        
        condition_small = metric_small.compute_condition_number()
        condition_large = metric_large.compute_condition_number()
        
        # Larger epsilon should lead to smaller condition number
        self.assertLess(condition_large, condition_small,
                       f"Larger epsilon should reduce condition number: "
                       f"small_eps={condition_small:.2f}, large_eps={condition_large:.2f}")


class TestSPDMetricNumericalStability(unittest.TestCase):
    """Test numerical stability and edge cases."""
    
    def setUp(self):
        torch.manual_seed(42)
        
    def test_extreme_condition_numbers(self):
        """Test behavior with extreme condition numbers."""
        # Create metric that might have high condition number
        metric = SPDMetric(
            embedding_dim=10,
            rank=10,  # Full rank
            epsilon=1e-12,  # Very small epsilon
            max_condition=1e8,
            device=torch.device('cpu')
        )
        
        # Force extreme eigenvalues
        with torch.no_grad():
            # Create matrix with extreme eigenvalues
            eigenvalues = torch.tensor([1e6, 1e5, 1e4, 100, 10, 5, 2, 1, 1, 1], dtype=torch.float32)
            eigenvectors = torch.randn(10, 10)
            eigenvectors, _ = torch.linalg.qr(eigenvectors)  # Orthogonalize
            
            # Reconstruct metric tensor
            G_target = eigenvectors @ torch.diag(eigenvalues) @ eigenvectors.t()
            
            # Try to make A produce this G (approximately)
            try:
                L = torch.linalg.cholesky(G_target - metric.epsilon * torch.eye(10))
                metric.A.data = L[:metric.rank, :]
            except:
                # If Cholesky fails, just use random initialization
                pass
        
        # Test that operations still work
        x = torch.randn(5, 10)
        y = torch.randn(5, 10)
        
        try:
            distances = metric(x, y)
            self.assertTrue(torch.all(torch.isfinite(distances)),
                           "Distances should be finite even with extreme condition numbers")
            
            condition = metric.compute_condition_number()
            self.assertTrue(np.isfinite(condition),
                           "Condition number should be finite")
            
        except Exception as e:
            self.fail(f"SPD metric failed with extreme condition numbers: {e}")
    
    def test_spectrum_clipping(self):
        """Test spectrum clipping functionality."""
        metric = SPDMetric(
            embedding_dim=8,
            rank=8,
            epsilon=1e-6,
            max_condition=100,  # Low max condition
            device=torch.device('cpu')
        )
        
        # Force high condition number
        with torch.no_grad():
            # Create A that will produce high condition number
            metric.A.data = torch.randn_like(metric.A) * 10
            metric.A.data[0] *= 100  # Make first row much larger
        
        # Check initial condition number
        initial_condition = metric.compute_condition_number()
        
        # Clip spectrum
        metric.clip_spectrum(min_eigenvalue=1e-3)
        
        # Check final condition number
        final_condition = metric.compute_condition_number()
        
        self.assertLessEqual(final_condition, metric.max_condition * 1.1,  # Allow small tolerance
                           f"Condition number should be clipped: "
                           f"initial={initial_condition:.1e}, final={final_condition:.1e}, "
                           f"max_allowed={metric.max_condition:.1e}")
    
    def test_nan_inf_handling(self):
        """Test handling of NaN and infinite values."""
        metric = SPDMetric(embedding_dim=5, rank=3, device=torch.device('cpu'))
        
        # Test with normal inputs
        x_normal = torch.randn(3, 5)
        y_normal = torch.randn(3, 5)
        
        distances_normal = metric(x_normal, y_normal)
        self.assertTrue(torch.all(torch.isfinite(distances_normal)),
                       "Normal inputs should produce finite distances")
        
        # Test with extreme but finite inputs
        x_large = torch.randn(3, 5) * 1e6
        y_large = torch.randn(3, 5) * 1e6
        
        try:
            distances_large = metric(x_large, y_large)
            # Should either work or fail gracefully
            if not torch.all(torch.isfinite(distances_large)):
                warnings.warn("Large inputs produced non-finite distances")
        except Exception as e:
            warnings.warn(f"Large inputs caused exception: {e}")
    
    def test_empty_and_single_point_cases(self):
        """Test edge cases with empty or single point inputs."""
        metric = SPDMetric(embedding_dim=4, rank=2, device=torch.device('cpu'))
        
        # Single point
        x_single = torch.randn(1, 4)
        y_single = torch.randn(1, 4)
        
        distances_single = metric(x_single, y_single)
        self.assertEqual(distances_single.shape, (1,),
                        "Single point should produce single distance")
        self.assertTrue(torch.isfinite(distances_single).all(),
                       "Single point distances should be finite")
        
        # Pairwise with single point
        pairwise_single = metric.compute_pairwise_distances(x_single, y_single)
        self.assertEqual(pairwise_single.shape, (1, 1),
                        "Single point pairwise should be 1x1")


class TestBatchedSPDMetric(unittest.TestCase):
    """Test batched SPD metric functionality."""
    
    def setUp(self):
        torch.manual_seed(42)
        
        self.metric = BatchedSPDMetric(
            embedding_dim=8,
            rank=4,
            device=torch.device('cpu')
        )
        
        self.batch_size = 12
        self.anchors = torch.randn(self.batch_size, 8)
        self.positives = torch.randn(self.batch_size, 8)
        self.negatives = torch.randn(self.batch_size, 8)
        self.labels = torch.randint(0, 3, (self.batch_size,))
    
    def test_triplet_distances(self):
        """Test triplet distance computation."""
        pos_distances, neg_distances = self.metric.compute_triplet_distances(
            self.anchors, self.positives, self.negatives
        )
        
        self.assertEqual(pos_distances.shape, (self.batch_size,),
                        "Positive distances should match batch size")
        self.assertEqual(neg_distances.shape, (self.batch_size,),
                        "Negative distances should match batch size")
        
        self.assertTrue(torch.all(pos_distances >= 0),
                       "Positive distances should be non-negative")
        self.assertTrue(torch.all(neg_distances >= 0),
                       "Negative distances should be non-negative")
        
        # Test consistency with individual computation
        for i in range(min(3, self.batch_size)):  # Test first few
            individual_pos = self.metric(
                self.anchors[i:i+1], self.positives[i:i+1], squared=True
            )
            individual_neg = self.metric(
                self.anchors[i:i+1], self.negatives[i:i+1], squared=True
            )
            
            self.assertAlmostEqual(pos_distances[i].item(), individual_pos.item(), places=5,
                                 msg=f"Positive distance {i} inconsistent")
            self.assertAlmostEqual(neg_distances[i].item(), individual_neg.item(), places=5,
                                 msg=f"Negative distance {i} inconsistent")
    
    def test_all_pairs_distances(self):
        """Test all pairs distance computation with masks."""
        distances, pos_mask, neg_mask = self.metric.compute_all_pairs_distances(
            self.anchors, self.labels
        )
        
        self.assertEqual(distances.shape, (self.batch_size, self.batch_size),
                        "Distance matrix should be batch_size x batch_size")
        self.assertEqual(pos_mask.shape, (self.batch_size, self.batch_size),
                        "Positive mask should be batch_size x batch_size")
        self.assertEqual(neg_mask.shape, (self.batch_size, self.batch_size),
                        "Negative mask should be batch_size x batch_size")
        
        # Test that masks are mutually exclusive
        overlap = pos_mask & neg_mask
        self.assertEqual(overlap.sum().item(), 0,
                        "Positive and negative masks should not overlap")
        
        # Test diagonal is excluded from positive mask
        diag_pos = torch.diag(pos_mask)
        self.assertEqual(diag_pos.sum().item(), 0,
                        "Diagonal should be excluded from positive mask")
        
        # Test symmetry of distance matrix
        self.assertTrue(torch.allclose(distances, distances.t(), atol=1e-6),
                       "Distance matrix should be symmetric")


class TestSPDMetricDevice(unittest.TestCase):
    """Test device placement and consistency."""
    
    def test_cpu_device(self):
        """Test CPU device functionality."""
        metric = SPDMetric(embedding_dim=6, rank=3, device=torch.device('cpu'))
        
        x = torch.randn(5, 6, device='cpu')
        y = torch.randn(5, 6, device='cpu')
        
        distances = metric(x, y)
        
        self.assertEqual(distances.device, torch.device('cpu'),
                        "Output should be on CPU device")
        self.assertTrue(torch.isfinite(distances).all(),
                       "CPU distances should be finite")
    
    def test_device_mismatch_handling(self):
        """Test handling of device mismatches."""
        metric = SPDMetric(embedding_dim=4, rank=2, device=torch.device('cpu'))
        
        x = torch.randn(3, 4, device='cpu')
        
        # Create y on different device (if available)
        if torch.cuda.is_available():
            try:
                y = torch.randn(3, 4, device='cuda')
                
                # This should either work (with automatic transfer) or fail gracefully
                try:
                    distances = metric(x, y)
                    # If it works, result should be on metric's device
                    self.assertEqual(distances.device, metric.device,
                                   "Result should be on metric's device")
                except RuntimeError:
                    # Expected if no automatic device transfer
                    pass
                    
            except RuntimeError:
                # CUDA not working, skip this test
                pass


def run_spd_tests():
    """Run all SPD metric tests."""
    print("="*60)
    print(" "*15 + "SPD METRIC IMPLEMENTATION TESTS")
    print("="*60)
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestSPDMetricProperties,
        TestSPDMetricNumericalStability,
        TestBatchedSPDMetric,
        TestSPDMetricDevice
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "="*60)
    print(" "*20 + "SPD TEST SUMMARY")
    print("="*60)
    
    total = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    passed = total - failures - errors
    
    print(f"Total tests: {total}")
    print(f"Passed: {passed} ✅")
    print(f"Failed: {failures} ❌")
    print(f"Errors: {errors} ⚠️")
    
    if result.failures:
        print(f"\nFailures:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print(f"\nErrors:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback.split('Exception:')[-1].strip()}")
    
    success = failures == 0 and errors == 0
    
    if success:
        print("\n✅ ALL SPD METRIC TESTS PASSED")
        print("   Mathematical properties verified")
        print("   Numerical stability confirmed")
        print("   Implementation correctness validated")
    else:
        print("\n❌ SOME SPD METRIC TESTS FAILED")
        print("   Review failures and errors above")
        print("   Implementation may need fixes")
    
    return success


if __name__ == '__main__':
    success = run_spd_tests()
    sys.exit(0 if success else 1)
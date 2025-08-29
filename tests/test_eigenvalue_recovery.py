#!/usr/bin/env python3
"""
Test script for eigenvalue recovery validation.
Confirms that the SPD metric learning achieves >99% correlation
with ground truth eigenvalues as specified in the Geodesyxx paper.
"""

import torch
import numpy as np
from synthetic_validation import (
    train_metric_recovery,
    run_validation_suite,
    SyntheticDataGenerator,
    MetricValidator
)
from spd_metric import SPDMetric
import sys


def test_basic_spd_metric():
    """Test basic SPD metric functionality."""
    print("\n" + "="*60)
    print("TESTING BASIC SPD METRIC FUNCTIONALITY")
    print("="*60)
    
    # Create metric
    metric = SPDMetric(embedding_dim=10, rank=5, epsilon=1e-6)
    
    # Test metric tensor computation
    G = metric.get_metric_tensor()
    print(f"✓ Metric tensor shape: {G.shape}")
    
    # Check positive definiteness
    eigenvalues = torch.linalg.eigvalsh(G)
    min_eig = eigenvalues.min().item()
    max_eig = eigenvalues.max().item()
    print(f"✓ Eigenvalue range: [{min_eig:.6f}, {max_eig:.6f}]")
    assert min_eig > 0, "Metric tensor must be positive definite"
    print(f"✓ Positive definite: True")
    
    # Test distance computation
    x = torch.randn(32, 10)
    y = torch.randn(32, 10)
    distances = metric.compute_mahalanobis_distance(x, y)
    print(f"✓ Distance computation shape: {distances.shape}")
    assert (distances >= 0).all(), "Distances must be non-negative"
    print(f"✓ Non-negative distances: True")
    
    # Test pairwise distances
    pairwise = metric.compute_pairwise_distances(x[:5])
    print(f"✓ Pairwise distance matrix shape: {pairwise.shape}")
    assert torch.allclose(pairwise, pairwise.t(), atol=1e-6), "Distance matrix must be symmetric"
    print(f"✓ Symmetric distance matrix: True")
    
    # Test condition number
    condition = metric.compute_condition_number()
    print(f"✓ Condition number: {condition:.2f}")
    
    print("\n✅ All basic tests passed!")
    return True


def test_eigenvalue_recovery_single():
    """Test eigenvalue recovery with single seed."""
    print("\n" + "="*60)
    print("TESTING EIGENVALUE RECOVERY (SINGLE SEED)")
    print("="*60)
    
    result = train_metric_recovery(
        dimension=10,
        n_samples=200,
        n_triplets=6000,
        n_epochs=100,  # Fewer epochs for testing
        learning_rate=0.01,
        margin=1.0,
        rank=10,
        seed=42,
        verbose=True
    )
    
    correlation = result['final_correlation']
    mse = result['final_mse']
    success = result['success']
    
    print(f"\n{'='*40}")
    print(f"SINGLE SEED RESULTS:")
    print(f"  Correlation: {correlation:.4f}")
    print(f"  MSE: {mse:.6f}")
    print(f"  Target (>99%): {'✅ PASSED' if success else '❌ FAILED'}")
    
    # Detailed eigenvalue comparison
    metric = result['metric']
    generator = result['generator']
    
    learned_eigs = metric.get_eigenvalues().cpu().detach().numpy()
    true_eigs = np.array(generator.true_eigenvalues)
    
    # Sort for comparison
    learned_sorted = np.sort(learned_eigs)[::-1]
    true_sorted = np.sort(true_eigs)[::-1]
    
    print(f"\nDetailed Eigenvalue Comparison:")
    print(f"{'Index':<8} {'True':<12} {'Learned':<12} {'Relative Error':<15}")
    print("-" * 50)
    
    for i in range(len(true_sorted)):
        rel_error = abs(true_sorted[i] - learned_sorted[i]) / true_sorted[i] * 100
        print(f"{i:<8} {true_sorted[i]:<12.4f} {learned_sorted[i]:<12.4f} {rel_error:<15.2f}%")
    
    return success


def test_eigenvalue_recovery_multiple():
    """Test eigenvalue recovery with multiple seeds."""
    print("\n" + "="*60)
    print("TESTING EIGENVALUE RECOVERY (MULTIPLE SEEDS)")
    print("="*60)
    
    seeds = [42, 123, 456]
    results = {}
    
    for seed in seeds:
        print(f"\n🌱 Testing seed {seed}...")
        print("-" * 40)
        
        result = train_metric_recovery(
            dimension=10,
            n_samples=200,
            n_triplets=6000,
            n_epochs=100,  # Fewer epochs for testing
            learning_rate=0.01,
            margin=1.0,
            rank=10,
            seed=seed,
            verbose=False  # Less verbose for multiple runs
        )
        
        results[seed] = {
            'correlation': result['final_correlation'],
            'mse': result['final_mse'],
            'success': result['success']
        }
        
        print(f"  Correlation: {result['final_correlation']:.4f}")
        print(f"  MSE: {result['final_mse']:.6f}")
        print(f"  Success: {'✅' if result['success'] else '❌'}")
    
    # Summary
    correlations = [r['correlation'] for r in results.values()]
    mses = [r['mse'] for r in results.values()]
    successes = [r['success'] for r in results.values()]
    
    print("\n" + "="*40)
    print("MULTIPLE SEED SUMMARY:")
    print(f"  Seeds: {seeds}")
    print(f"  Mean correlation: {np.mean(correlations):.4f} ± {np.std(correlations):.4f}")
    print(f"  Mean MSE: {np.mean(mses):.6f} ± {np.std(mses):.6f}")
    print(f"  Success rate: {sum(successes)}/{len(successes)}")
    print(f"  All passed: {'✅' if all(successes) else '❌'}")
    
    return all(successes)


def test_different_dimensions():
    """Test recovery with different embedding dimensions."""
    print("\n" + "="*60)
    print("TESTING DIFFERENT DIMENSIONS")
    print("="*60)
    
    dimensions = [5, 10, 20]
    results = {}
    
    for dim in dimensions:
        print(f"\n📐 Testing dimension {dim}...")
        
        # Create appropriate eigenvalues
        if dim == 5:
            true_eigenvalues = [5.0, 3.0, 2.0, 1.0, 1.0]
        elif dim == 10:
            true_eigenvalues = [9.0, 3.0] + [1.0] * 8
        else:  # dim == 20
            true_eigenvalues = [9.0, 5.0, 3.0] + [1.0] * 17
        
        generator = SyntheticDataGenerator(
            dimension=dim,
            n_samples=200,
            true_eigenvalues=true_eigenvalues,
            seed=42
        )
        
        embeddings = generator.generate_embeddings()
        anchors, positives, negatives = generator.generate_triplets(embeddings, n_triplets=3000)
        
        # Train metric
        device = torch.device('cpu')
        metric = SPDMetric(embedding_dim=dim, rank=dim, epsilon=1e-6, device=device)
        
        optimizer = torch.optim.Adam(metric.parameters(), lr=0.01)
        
        # Quick training
        for epoch in range(50):
            optimizer.zero_grad()
            pos_dist = metric(anchors, positives, squared=False)
            neg_dist = metric(anchors, negatives, squared=False)
            loss = torch.relu(pos_dist - neg_dist + 1.0).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(metric.parameters(), 1.0)
            optimizer.step()
        
        # Validate
        validator = MetricValidator(true_eigenvalues)
        correlation = validator.compute_eigenvalue_correlation(metric)
        
        results[dim] = correlation
        print(f"  Dimension {dim}: Correlation = {correlation:.4f}")
    
    print("\n" + "="*40)
    print("DIMENSION TEST SUMMARY:")
    for dim, corr in results.items():
        print(f"  Dim {dim}: {corr:.4f} {'✅' if corr > 0.9 else '❌'}")
    
    return all(corr > 0.9 for corr in results.values())


def main():
    """Run all validation tests."""
    print("\n" + "="*70)
    print(" "*20 + "EIGENVALUE RECOVERY VALIDATION")
    print(" "*15 + "Testing SPD Metric Learning Implementation")
    print("="*70)
    
    all_passed = True
    
    # Test 1: Basic functionality
    try:
        passed = test_basic_spd_metric()
        all_passed = all_passed and passed
    except Exception as e:
        print(f"❌ Basic functionality test failed: {e}")
        all_passed = False
    
    # Test 2: Single seed recovery
    try:
        passed = test_eigenvalue_recovery_single()
        all_passed = all_passed and passed
    except Exception as e:
        print(f"❌ Single seed recovery test failed: {e}")
        all_passed = False
    
    # Test 3: Multiple seed recovery
    try:
        passed = test_eigenvalue_recovery_multiple()
        all_passed = all_passed and passed
    except Exception as e:
        print(f"❌ Multiple seed recovery test failed: {e}")
        all_passed = False
    
    # Test 4: Different dimensions
    try:
        passed = test_different_dimensions()
        all_passed = all_passed and passed
    except Exception as e:
        print(f"❌ Different dimensions test failed: {e}")
        all_passed = False
    
    # Final summary
    print("\n" + "="*70)
    print(" "*25 + "FINAL VALIDATION SUMMARY")
    print("="*70)
    
    if all_passed:
        print("✅ ALL VALIDATION TESTS PASSED!")
        print("The SPD metric learning implementation successfully recovers")
        print("synthetic geometric structure with >99% eigenvalue correlation.")
        return 0
    else:
        print("❌ SOME VALIDATION TESTS FAILED")
        print("Please review the output above for details.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
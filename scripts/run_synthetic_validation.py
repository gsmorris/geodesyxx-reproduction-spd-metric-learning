#!/usr/bin/env python3
"""
Synthetic Validation Script for SPD Metric Learning
Validates eigenvalue recovery performance as per Geodesyxx paper Phase 1.
Target: >99% correlation between learned and true eigenvalues.
"""

import sys
import os
import argparse
from pathlib import Path
import yaml
import torch
import numpy as np
from typing import Dict, Tuple

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from synthetic_validation import (
    set_seed, create_synthetic_data, train_spd_metric,
    evaluate_eigenvalue_recovery, plot_eigenvalue_recovery
)

def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def run_synthetic_validation(config_path: str = None, seed: int = 42) -> Dict:
    """
    Run synthetic validation experiment.
    
    Args:
        config_path: Path to configuration file
        seed: Random seed for reproducibility
        
    Returns:
        Dictionary containing validation results
    """
    # Load configuration
    if config_path is None:
        config_path = Path(__file__).parent.parent / 'configs' / 'phase1_3_config.yaml'
    
    config = load_config(config_path)
    synthetic_config = config['synthetic_validation']
    
    # Set seed for reproducibility
    set_seed(seed)
    
    print(f"🔬 Running Synthetic Validation (seed={seed})")
    print(f"📐 Dimension: {synthetic_config['dimension']}")
    print(f"📊 Samples: {synthetic_config['n_samples']}")
    print(f"🎯 Target correlation: {synthetic_config['target_correlation']}")
    
    # Create synthetic data
    print("\n📦 Generating synthetic data...")
    data = create_synthetic_data(
        dimension=synthetic_config['dimension'],
        n_samples=synthetic_config['n_samples'],
        true_eigenvalues=synthetic_config['true_eigenvalues'],
        n_triplets=synthetic_config['n_triplets'],
        seed=seed
    )
    
    print(f"✅ Generated {len(data['triplets'])} triplets")
    
    # Train SPD metric
    print("\n🏋️ Training SPD metric...")
    results = train_spd_metric(
        triplets=data['triplets'],
        dimension=synthetic_config['dimension'],
        rank=synthetic_config['rank'],
        learning_rate=synthetic_config['learning_rate'],
        n_epochs=synthetic_config['n_epochs'],
        margin=synthetic_config['margin']
    )
    
    print(f"✅ Training completed in {results['training_time']:.2f}s")
    
    # Evaluate eigenvalue recovery
    print("\n📊 Evaluating eigenvalue recovery...")
    recovery_results = evaluate_eigenvalue_recovery(
        learned_metric=results['metric'],
        true_eigenvalues=np.array(synthetic_config['true_eigenvalues'])
    )
    
    correlation = recovery_results['correlation']
    target = synthetic_config['target_correlation']
    
    print(f"🎯 Eigenvalue correlation: {correlation:.4f}")
    print(f"🎯 Target correlation: {target:.4f}")
    
    if correlation >= target:
        print("✅ VALIDATION PASSED: Target correlation achieved!")
    else:
        print("❌ VALIDATION FAILED: Target correlation not achieved")
    
    # Create output directory
    output_dir = Path('results/synthetic_validation')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot results
    print("\n📈 Creating visualizations...")
    plot_eigenvalue_recovery(
        recovery_results,
        save_path=output_dir / f'eigenvalue_recovery_seed_{seed}.png'
    )
    
    # Compile final results
    final_results = {
        'seed': seed,
        'config': synthetic_config,
        'eigenvalue_correlation': correlation,
        'target_achieved': correlation >= target,
        'true_eigenvalues': synthetic_config['true_eigenvalues'],
        'learned_eigenvalues': recovery_results['learned_eigenvalues'].tolist(),
        'training_time': results['training_time'],
        'final_loss': results['final_loss'],
        'condition_number': recovery_results['condition_number']
    }
    
    # Save results
    import json
    results_path = output_dir / f'synthetic_validation_seed_{seed}.json'
    with open(results_path, 'w') as f:
        json.dump(final_results, f, indent=2)
    
    print(f"💾 Results saved to: {results_path}")
    print(f"📈 Plot saved to: {output_dir / f'eigenvalue_recovery_seed_{seed}.png'}")
    
    return final_results

def main():
    """Main entry point for synthetic validation."""
    parser = argparse.ArgumentParser(description='Run synthetic validation for SPD metric learning')
    parser.add_argument('--config', '-c', type=str, default=None,
                       help='Path to configuration file')
    parser.add_argument('--seed', '-s', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--seeds', nargs='+', type=int,
                       help='Multiple seeds to run (overrides --seed)')
    parser.add_argument('--output-dir', '-o', type=str, default='results/synthetic_validation',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Determine seeds to run
    if args.seeds:
        seeds = args.seeds
    else:
        seeds = [args.seed]
    
    print(f"🚀 Starting synthetic validation with seeds: {seeds}")
    
    all_results = []
    for seed in seeds:
        print(f"\n{'='*60}")
        print(f"Running validation with seed: {seed}")
        print('='*60)
        
        try:
            results = run_synthetic_validation(args.config, seed)
            all_results.append(results)
            
            if results['target_achieved']:
                print(f"✅ Seed {seed}: PASSED")
            else:
                print(f"❌ Seed {seed}: FAILED")
        
        except Exception as e:
            print(f"💥 Error with seed {seed}: {str(e)}")
            all_results.append({
                'seed': seed,
                'error': str(e),
                'target_achieved': False
            })
    
    # Summary
    print(f"\n{'='*60}")
    print("SYNTHETIC VALIDATION SUMMARY")
    print('='*60)
    
    passed = sum(1 for r in all_results if r.get('target_achieved', False))
    total = len(all_results)
    
    print(f"Results: {passed}/{total} seeds passed")
    
    if passed == total:
        print("🎉 ALL VALIDATIONS PASSED!")
        print("✅ SPD metric implementation is working correctly")
    else:
        print("⚠️  Some validations failed - check individual results")
    
    # Save summary
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    summary_path = output_dir / 'validation_summary.json'
    summary = {
        'total_seeds': total,
        'passed_seeds': passed,
        'success_rate': passed / total if total > 0 else 0,
        'individual_results': all_results
    }
    
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"📄 Summary saved to: {summary_path}")
    
    return passed == total

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
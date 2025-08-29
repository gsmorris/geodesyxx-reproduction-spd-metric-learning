#!/usr/bin/env python3
"""
Phase 1-3 Experiments: Global SPD Metric Learning
Reproduces WordSim353 and WordNet evaluation experiments from the Geodesyxx paper.
Expected Results: Performance degradation demonstrating negative results.
"""

import sys
import os
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any
import yaml
import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# Import required modules
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from spd_metric import SPDMetric
    from evaluation import BootstrapAnalyzer, StatisticalResult
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("💡 Make sure you've installed the requirements: pip install -r requirements.txt")
    sys.exit(1)

def load_config(config_path: str) -> Dict:
    """Load configuration from YAML file."""
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print(f"❌ Configuration file not found: {config_path}")
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"❌ Error parsing configuration file: {e}")
        sys.exit(1)

def setup_device(device_preference: List[str]) -> torch.device:
    """Setup optimal device based on preferences."""
    for device_name in device_preference:
        if device_name == 'cuda' and torch.cuda.is_available():
            device = torch.device('cuda')
            print(f"🔧 Using CUDA device: {torch.cuda.get_device_name()}")
            return device
        elif device_name == 'mps' and torch.backends.mps.is_available():
            try:
                device = torch.device('mps')
                # Test MPS functionality
                x = torch.randn(10, 10, device=device)
                _ = torch.matmul(x, x.t())
                print("🔧 Using MPS device (Apple Silicon)")
                return device
            except:
                print("⚠️  MPS available but not working, trying next device...")
        elif device_name == 'cpu':
            device = torch.device('cpu')
            print("🔧 Using CPU device")
            return device
    
    # Fallback to CPU
    print("⚠️  Falling back to CPU")
    return torch.device('cpu')

def load_embeddings(config: Dict) -> Tuple[Dict[str, np.ndarray], int]:
    """
    Load pre-trained embeddings (placeholder for actual implementation).
    
    Returns:
        Tuple of (word_to_vector dict, embedding_dimension)
    """
    # This is a placeholder - in real implementation would load actual GloVe vectors
    print("📥 Loading embeddings...")
    embedding_dim = config['embeddings']['dimension']
    
    # For demonstration, create dummy embeddings
    # In real implementation: load from glove.6B.300d.txt or similar
    dummy_vocab = ['good', 'bad', 'excellent', 'terrible', 'cat', 'dog', 'house', 'car']
    embeddings = {}
    
    np.random.seed(42)  # For reproducible dummy data
    for word in dummy_vocab:
        embeddings[word] = np.random.randn(embedding_dim).astype(np.float32)
        if config['embeddings']['preprocessing']['l2_normalize']:
            embeddings[word] = embeddings[word] / np.linalg.norm(embeddings[word])
    
    print(f"✅ Loaded {len(embeddings)} embeddings (dimension: {embedding_dim})")
    return embeddings, embedding_dim

def load_wordsim353() -> List[Tuple[str, str, float]]:
    """
    Load WordSim353 dataset (placeholder for actual implementation).
    
    Returns:
        List of (word1, word2, human_rating) tuples
    """
    print("📥 Loading WordSim353 dataset...")
    
    # Placeholder data - in real implementation would load from CSV
    dummy_pairs = [
        ('good', 'excellent', 8.5),
        ('bad', 'terrible', 8.0),
        ('cat', 'dog', 6.5),
        ('house', 'car', 2.0),
    ]
    
    print(f"✅ Loaded {len(dummy_pairs)} word pairs")
    return dummy_pairs

def load_wordnet_pairs() -> List[Tuple[str, str, int]]:
    """
    Load WordNet hypernym pairs (placeholder for actual implementation).
    
    Returns:
        List of (word1, word2, path_length) tuples
    """
    print("📥 Loading WordNet pairs...")
    
    # Placeholder data
    dummy_pairs = [
        ('cat', 'animal', 2),
        ('dog', 'animal', 2),
        ('house', 'building', 1),
        ('car', 'vehicle', 1),
    ]
    
    print(f"✅ Loaded {len(dummy_pairs)} hypernym pairs")
    return dummy_pairs

def train_global_spd_metric(
    embeddings: Dict[str, np.ndarray],
    config: Dict,
    device: torch.device,
    seed: int
) -> Tuple[SPDMetric, Dict[str, Any]]:
    """
    Train global SPD metric on embedding space.
    
    Returns:
        Tuple of (trained_metric, training_stats)
    """
    print("🏋️  Training global SPD metric...")
    
    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Initialize metric
    metric_config = config['global_metric']
    metric = SPDMetric(
        embedding_dim=metric_config['embedding_dim'],
        rank=metric_config['rank'],
        epsilon=metric_config['epsilon'],
        max_condition=metric_config['max_condition_number'],
        device=device
    )
    
    # Setup optimizer
    training_config = config['training']
    optimizer = optim.Adam(
        metric.parameters(),
        lr=training_config['learning_rate'],
        weight_decay=training_config['weight_decay']
    )
    
    # Create triplets for training (simplified approach)
    word_list = list(embeddings.keys())
    embeddings_tensor = torch.stack([
        torch.tensor(embeddings[word], device=device) for word in word_list
    ])
    
    # Training statistics
    stats = {
        'losses': [],
        'condition_numbers': [],
        'training_time': 0
    }
    
    start_time = time.time()
    
    for epoch in range(training_config['max_epochs']):
        epoch_loss = 0
        n_batches = 0
        
        # Simple triplet mining (in real implementation would be more sophisticated)
        for i in range(0, len(word_list), training_config['batch_size']):
            batch_end = min(i + training_config['batch_size'], len(word_list))
            if batch_end - i < 3:  # Need at least 3 items for triplets
                continue
            
            # Create simple triplets within batch
            batch_embeddings = embeddings_tensor[i:batch_end]
            
            # Compute pairwise distances
            distances = metric.compute_pairwise_distances(batch_embeddings)
            
            # Simple triplet loss (placeholder - would use proper hard negative mining)
            margin = training_config['triplet_loss']['margin']
            loss = torch.clamp(distances.mean() - margin, min=0)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            n_batches += 1
        
        avg_loss = epoch_loss / max(n_batches, 1)
        condition_number = metric.compute_condition_number()
        
        stats['losses'].append(avg_loss)
        stats['condition_numbers'].append(condition_number)
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1}/{training_config['max_epochs']}: "
                  f"Loss={avg_loss:.4f}, Condition={condition_number:.1f}")
    
    stats['training_time'] = time.time() - start_time
    
    print(f"✅ Training completed in {stats['training_time']:.2f}s")
    print(f"📊 Final condition number: {stats['condition_numbers'][-1]:.1f}")
    
    return metric, stats

def evaluate_wordsim353(
    embeddings: Dict[str, np.ndarray],
    pairs: List[Tuple[str, str, float]],
    metric: SPDMetric = None,
    device: torch.device = None
) -> Dict[str, float]:
    """
    Evaluate on WordSim353 dataset.
    
    Returns:
        Dictionary with evaluation metrics
    """
    print("📊 Evaluating on WordSim353...")
    
    predicted_similarities = []
    human_ratings = []
    
    for word1, word2, human_rating in pairs:
        if word1 in embeddings and word2 in embeddings:
            emb1 = torch.tensor(embeddings[word1], device=device).unsqueeze(0)
            emb2 = torch.tensor(embeddings[word2], device=device).unsqueeze(0)
            
            if metric is not None:
                # Use learned metric
                distance = metric.compute_mahalanobis_distance(emb1, emb2, squared=False)
                similarity = 1.0 / (1.0 + distance.item())  # Convert distance to similarity
            else:
                # Use Euclidean distance (baseline)
                distance = torch.norm(emb1 - emb2)
                similarity = 1.0 / (1.0 + distance.item())
            
            predicted_similarities.append(similarity)
            human_ratings.append(human_rating)
    
    # Compute Spearman correlation
    from scipy.stats import spearmanr
    correlation, p_value = spearmanr(predicted_similarities, human_ratings)
    
    results = {
        'spearman_correlation': correlation,
        'p_value': p_value,
        'n_pairs': len(predicted_similarities)
    }
    
    print(f"✅ WordSim353: ρ = {correlation:.3f} (p = {p_value:.3f})")
    return results

def evaluate_wordnet(
    embeddings: Dict[str, np.ndarray],
    pairs: List[Tuple[str, str, int]],
    metric: SPDMetric = None,
    device: torch.device = None
) -> Dict[str, float]:
    """
    Evaluate on WordNet hypernym pairs.
    
    Returns:
        Dictionary with evaluation metrics
    """
    print("📊 Evaluating on WordNet pairs...")
    
    predicted_distances = []
    path_lengths = []
    
    for word1, word2, path_length in pairs:
        if word1 in embeddings and word2 in embeddings:
            emb1 = torch.tensor(embeddings[word1], device=device).unsqueeze(0)
            emb2 = torch.tensor(embeddings[word2], device=device).unsqueeze(0)
            
            if metric is not None:
                distance = metric.compute_mahalanobis_distance(emb1, emb2, squared=False)
            else:
                distance = torch.norm(emb1 - emb2)
            
            predicted_distances.append(distance.item())
            path_lengths.append(path_length)
    
    # Compute Spearman correlation
    from scipy.stats import spearmanr
    correlation, p_value = spearmanr(predicted_distances, path_lengths)
    
    results = {
        'spearman_correlation': correlation,
        'p_value': p_value,
        'n_pairs': len(predicted_distances)
    }
    
    print(f"✅ WordNet: ρ = {correlation:.3f} (p = {p_value:.3f})")
    return results

def run_phase1_3_experiment(config_path: str, seed: int) -> Dict[str, Any]:
    """
    Run complete Phase 1-3 experiment for one seed.
    
    Returns:
        Dictionary containing all experimental results
    """
    print(f"🚀 Running Phase 1-3 experiment (seed={seed})")
    
    # Load configuration
    config = load_config(config_path)
    
    # Setup device
    device = setup_device(config['hardware']['device_preference'])
    
    # Load data
    embeddings, embed_dim = load_embeddings(config)
    wordsim_pairs = load_wordsim353()
    wordnet_pairs = load_wordnet_pairs()
    
    # Baseline evaluation (no learned metric)
    print("\n📊 Baseline Evaluation (Euclidean distance)")
    baseline_wordsim = evaluate_wordsim353(embeddings, wordsim_pairs, None, device)
    baseline_wordnet = evaluate_wordnet(embeddings, wordnet_pairs, None, device)
    
    # Train SPD metric
    print("\n🏋️  Training Phase")
    trained_metric, training_stats = train_global_spd_metric(embeddings, config, device, seed)
    
    # Evaluation with learned metric
    print("\n📊 Learned Metric Evaluation")
    learned_wordsim = evaluate_wordsim353(embeddings, wordsim_pairs, trained_metric, device)
    learned_wordnet = evaluate_wordnet(embeddings, wordnet_pairs, trained_metric, device)
    
    # Compile results
    results = {
        'seed': seed,
        'config': config,
        'device': str(device),
        'baseline': {
            'wordsim353': baseline_wordsim,
            'wordnet': baseline_wordnet
        },
        'learned_metric': {
            'wordsim353': learned_wordsim,
            'wordnet': learned_wordnet
        },
        'training_stats': training_stats,
        'performance_changes': {
            'wordsim353_change': learned_wordsim['spearman_correlation'] - baseline_wordsim['spearman_correlation'],
            'wordnet_change': learned_wordnet['spearman_correlation'] - baseline_wordnet['spearman_correlation']
        }
    }
    
    # Print summary
    print(f"\n📋 RESULTS SUMMARY (seed={seed})")
    print(f"WordSim353: {baseline_wordsim['spearman_correlation']:.3f} → {learned_wordsim['spearman_correlation']:.3f} "
          f"(Δ = {results['performance_changes']['wordsim353_change']:.3f})")
    print(f"WordNet: {baseline_wordnet['spearman_correlation']:.3f} → {learned_wordnet['spearman_correlation']:.3f} "
          f"(Δ = {results['performance_changes']['wordnet_change']:.3f})")
    print(f"Final condition number: {training_stats['condition_numbers'][-1]:.1f}")
    
    return results

def main():
    """Main entry point for Phase 1-3 experiments."""
    parser = argparse.ArgumentParser(description='Run Phase 1-3 experiments')
    parser.add_argument('--config', '-c', type=str, 
                       default='configs/phase1_3_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--seed', '-s', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--seeds', nargs='+', type=int,
                       help='Multiple seeds to run (overrides --seed)')
    parser.add_argument('--output-dir', '-o', type=str, default='results/phase1_3',
                       help='Output directory for results')
    
    args = parser.parse_args()
    
    # Determine seeds to run
    if args.seeds:
        seeds = args.seeds
    else:
        # Default seeds from config
        config = load_config(args.config)
        seeds = config.get('seeds', [args.seed])
    
    print(f"🚀 Starting Phase 1-3 experiments with seeds: {seeds}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    all_results = []
    
    for seed in seeds:
        print(f"\n{'='*80}")
        print(f"EXPERIMENT: Phase 1-3 with seed {seed}")
        print('='*80)
        
        try:
            results = run_phase1_3_experiment(args.config, seed)
            all_results.append(results)
            
            # Save individual result
            result_path = output_dir / f'phase1_3_results_seed_{seed}.json'
            with open(result_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            print(f"💾 Results saved: {result_path}")
            
        except Exception as e:
            print(f"💥 Error with seed {seed}: {str(e)}")
            import traceback
            traceback.print_exc()
            
            all_results.append({
                'seed': seed,
                'error': str(e),
                'success': False
            })
    
    # Aggregate results across seeds
    print(f"\n{'='*80}")
    print("PHASE 1-3 SUMMARY")
    print('='*80)
    
    successful_results = [r for r in all_results if 'error' not in r]
    
    if successful_results:
        # Compute aggregate statistics
        wordsim_changes = [r['performance_changes']['wordsim353_change'] for r in successful_results]
        wordnet_changes = [r['performance_changes']['wordnet_change'] for r in successful_results]
        condition_numbers = [r['training_stats']['condition_numbers'][-1] for r in successful_results]
        
        summary = {
            'n_seeds': len(successful_results),
            'wordsim353_change': {
                'mean': np.mean(wordsim_changes),
                'std': np.std(wordsim_changes),
                'all_values': wordsim_changes
            },
            'wordnet_change': {
                'mean': np.mean(wordnet_changes),
                'std': np.std(wordnet_changes),
                'all_values': wordnet_changes
            },
            'condition_numbers': {
                'mean': np.mean(condition_numbers),
                'std': np.std(condition_numbers),
                'range': [min(condition_numbers), max(condition_numbers)],
                'all_values': condition_numbers
            },
            'individual_results': all_results
        }
        
        print(f"📊 Successful experiments: {len(successful_results)}/{len(seeds)}")
        print(f"📉 WordSim353 change: {summary['wordsim353_change']['mean']:.3f} ± {summary['wordsim353_change']['std']:.3f}")
        print(f"📉 WordNet change: {summary['wordnet_change']['mean']:.3f} ± {summary['wordnet_change']['std']:.3f}")
        print(f"🔢 Condition numbers: {summary['condition_numbers']['mean']:.1f} ± {summary['condition_numbers']['std']:.1f}")
        print(f"   Range: [{summary['condition_numbers']['range'][0]:.1f}, {summary['condition_numbers']['range'][1]:.1f}]")
        
        # Check if results align with paper expectations
        expected_degradation = summary['wordsim353_change']['mean'] < 0 and summary['wordnet_change']['mean'] < 0
        condition_in_range = 18 <= summary['condition_numbers']['mean'] <= 40
        
        if expected_degradation:
            print("✅ Performance degradation observed (consistent with negative results)")
        else:
            print("⚠️  Unexpected performance improvement (inconsistent with paper)")
        
        if condition_in_range:
            print("✅ Condition numbers in expected range [18, 40]")
        else:
            print(f"⚠️  Condition numbers outside expected range: {summary['condition_numbers']['mean']:.1f}")
    
    else:
        summary = {
            'n_seeds': 0,
            'error': 'All experiments failed',
            'individual_results': all_results
        }
        print("❌ All experiments failed")
    
    # Save aggregate summary
    summary_path = output_dir / 'phase1_3_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"📄 Summary saved: {summary_path}")
    
    return len(successful_results) > 0

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
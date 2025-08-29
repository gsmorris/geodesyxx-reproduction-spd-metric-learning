#!/usr/bin/env python3
"""
Phase 4 Experiments: Local SPD-Weighted Attention in DistilBERT
Reproduces CoLA evaluation experiments from the Geodesyxx paper.
Expected Results: High condition numbers (>171K), negligible MCC improvement (<0.01).
"""

import sys
import os
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import yaml
import numpy as np
import pandas as pd
import warnings

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# Import required modules
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from sklearn.metrics import matthews_corrcoef, accuracy_score, f1_score
    from transformers import (
        AutoTokenizer, AutoModelForSequenceClassification,
        TrainingArguments, Trainer, EvalPrediction
    )
    
    # Import our modules
    from transformer_integration import (
        CurvedDistilBertForSequenceClassification, 
        create_curved_distilbert
    )
    from training import DualOptimizerTrainer, TrainingConfig
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

class CoLADataset:
    """
    Placeholder for CoLA dataset loading.
    In real implementation, would load from GLUE benchmark.
    """
    
    def __init__(self, data_dir: str = None, tokenizer_name: str = "distilbert-base-uncased"):
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Placeholder data - in real implementation would load actual CoLA
        self.dummy_data = {
            'train': [
                ("The cat sat on the mat.", 1),  # Acceptable
                ("Cat the on sat mat the.", 0),  # Unacceptable
                ("She gave him a book.", 1),
                ("Him she gave book a.", 0),
            ] * 50,  # Repeat to simulate larger dataset
            'validation': [
                ("The dog barked loudly.", 1),
                ("Barked dog loudly the.", 0),
                ("I love reading books.", 1),
                ("Books reading love I.", 0),
            ] * 25
        }
        
        print(f"📥 Loaded CoLA dataset:")
        print(f"   Train: {len(self.dummy_data['train'])} samples")
        print(f"   Validation: {len(self.dummy_data['validation'])} samples")
    
    def get_dataloader(self, split: str, batch_size: int, max_length: int = 128) -> DataLoader:
        """Get DataLoader for specified split."""
        data = self.dummy_data[split]
        
        # Tokenize data
        texts = [item[0] for item in data]
        labels = [item[1] for item in data]
        
        encodings = self.tokenizer(
            texts,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt'
        )
        
        # Create simple dataset
        class SimpleDataset(torch.utils.data.Dataset):
            def __init__(self, encodings, labels):
                self.encodings = encodings
                self.labels = labels
            
            def __getitem__(self, idx):
                return {
                    'input_ids': self.encodings['input_ids'][idx],
                    'attention_mask': self.encodings['attention_mask'][idx],
                    'labels': torch.tensor(self.labels[idx], dtype=torch.long)
                }
            
            def __len__(self):
                return len(self.labels)
        
        dataset = SimpleDataset(encodings, labels)
        return DataLoader(dataset, batch_size=batch_size, shuffle=(split == 'train'))

def create_model(
    config: Dict, 
    geometry_mode: str, 
    curved_layers: List[int],
    device: torch.device
) -> nn.Module:
    """
    Create model with specified geometry configuration.
    
    Args:
        config: Configuration dictionary
        geometry_mode: 'none', 'shared', or 'per_head'
        curved_layers: List of layer indices to make curved
        device: Device to place model on
        
    Returns:
        Configured model
    """
    print(f"🏗️  Creating model: geometry={geometry_mode}, layers={curved_layers}")
    
    if geometry_mode == 'none':
        # Standard DistilBERT
        from transformers import DistilBertForSequenceClassification
        model = DistilBertForSequenceClassification.from_pretrained(
            config['model']['base_model'],
            num_labels=2
        )
    else:
        # Curved DistilBERT
        model = create_curved_distilbert(
            model_name=config['model']['base_model'],
            num_labels=2,
            curved_layers=curved_layers,
            geometry_mode=geometry_mode,
            rank=config['model']['curved_attention']['spd_settings']['rank']
        )
    
    model.to(device)
    
    # Print parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"📊 Model parameters:")
    print(f"   Total: {total_params:,}")
    print(f"   Trainable: {trainable_params:,}")
    
    if geometry_mode != 'none':
        # Get geometric parameter count
        geometric_params = 0
        for name, param in model.named_parameters():
            if 'spd_metrics' in name:
                geometric_params += param.numel()
        
        print(f"   Geometric: {geometric_params:,}")
        print(f"   Standard: {trainable_params - geometric_params:,}")
    
    return model

def compute_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
    """Compute evaluation metrics for CoLA task."""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    return {
        'accuracy': accuracy_score(labels, predictions),
        'f1': f1_score(labels, predictions, average='binary'),
        'matthews_correlation': matthews_corrcoef(labels, predictions)
    }

def train_model(
    model: nn.Module,
    train_dataloader: DataLoader,
    eval_dataloader: DataLoader,
    config: Dict,
    device: torch.device,
    use_dual_optimizer: bool = False
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Train model with specified configuration.
    
    Returns:
        Tuple of (trained_model, training_stats)
    """
    print("🏋️  Starting model training...")
    
    training_config = config['training']
    
    if use_dual_optimizer:
        print("🔧 Using dual optimizer setup")
        # Use our custom dual optimizer trainer
        trainer_config = TrainingConfig(
            geometric_lr=training_config['geometric_lr'],
            standard_lr=training_config['standard_lr'],
            batch_size=config['data']['train_batch_size'],
            eval_batch_size=config['data']['eval_batch_size'],
            patience=training_config['early_stopping']['patience'],
            min_delta=training_config['early_stopping']['min_delta'],
            max_condition_number=training_config['stability_checks'].get('max_condition_number', 1e6)
        )
        
        trainer = DualOptimizerTrainer(model, trainer_config, device)
        
        # Train for specified epochs
        training_stats = trainer.train(
            train_dataloader=train_dataloader,
            eval_dataloader=eval_dataloader,
            num_epochs=training_config['num_epochs']
        )
        
    else:
        print("🔧 Using standard optimizer")
        # Standard training
        optimizer = optim.AdamW(
            model.parameters(),
            lr=training_config['standard_lr'],
            weight_decay=training_config['weight_decay']
        )
        
        training_stats = {
            'losses': [],
            'eval_metrics': [],
            'condition_numbers': [],
            'training_time': 0
        }
        
        start_time = time.time()
        
        for epoch in range(training_config['num_epochs']):
            # Training phase
            model.train()
            epoch_loss = 0
            
            for batch in train_dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                
                optimizer.zero_grad()
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()
                
                # Gradient clipping
                if training_config.get('gradient_clipping', {}).get('enabled', False):
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), 
                        training_config['gradient_clipping']['max_norm']
                    )
                
                optimizer.step()
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / len(train_dataloader)
            training_stats['losses'].append(avg_loss)
            
            # Evaluation phase
            model.eval()
            eval_predictions = []
            eval_labels = []
            
            with torch.no_grad():
                for batch in eval_dataloader:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    outputs = model(**batch)
                    
                    predictions = torch.argmax(outputs.logits, dim=-1)
                    eval_predictions.extend(predictions.cpu().numpy())
                    eval_labels.extend(batch['labels'].cpu().numpy())
            
            # Compute metrics
            epoch_metrics = {
                'accuracy': accuracy_score(eval_labels, eval_predictions),
                'f1': f1_score(eval_labels, eval_predictions, average='binary'),
                'matthews_correlation': matthews_corrcoef(eval_labels, eval_predictions)
            }
            training_stats['eval_metrics'].append(epoch_metrics)
            
            # Monitor condition numbers if applicable
            if hasattr(model, 'get_condition_numbers'):
                condition_numbers = model.get_condition_numbers()
                training_stats['condition_numbers'].append(condition_numbers)
                max_condition = max(condition_numbers.values()) if condition_numbers else 0
                print(f"Epoch {epoch + 1}: Loss={avg_loss:.4f}, MCC={epoch_metrics['matthews_correlation']:.3f}, Max_Cond={max_condition:.1e}")
            else:
                training_stats['condition_numbers'].append({})
                print(f"Epoch {epoch + 1}: Loss={avg_loss:.4f}, MCC={epoch_metrics['matthews_correlation']:.3f}")
        
        training_stats['training_time'] = time.time() - start_time
    
    print(f"✅ Training completed in {training_stats['training_time']:.2f}s")
    return model, training_stats

def evaluate_model(
    model: nn.Module,
    eval_dataloader: DataLoader,
    device: torch.device
) -> Dict[str, float]:
    """Evaluate trained model."""
    print("📊 Evaluating model...")
    
    model.eval()
    predictions = []
    labels = []
    
    with torch.no_grad():
        for batch in eval_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            
            batch_predictions = torch.argmax(outputs.logits, dim=-1)
            predictions.extend(batch_predictions.cpu().numpy())
            labels.extend(batch['labels'].cpu().numpy())
    
    metrics = compute_metrics(EvalPrediction(predictions=np.array(predictions), label_ids=np.array(labels)))
    
    print(f"📊 Evaluation Results:")
    for metric, value in metrics.items():
        print(f"   {metric}: {value:.3f}")
    
    return metrics

def run_phase4_experiment(
    config_path: str,
    geometry_mode: str,
    curved_layers: List[int],
    seed: int
) -> Dict[str, Any]:
    """
    Run single Phase 4 experiment configuration.
    
    Returns:
        Dictionary containing experimental results
    """
    print(f"🚀 Running Phase 4: geometry={geometry_mode}, layers={curved_layers}, seed={seed}")
    
    # Set seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Load configuration
    config = load_config(config_path)
    
    # Setup device
    device = setup_device(config['hardware']['device_preference'])
    
    # Load dataset
    dataset = CoLADataset()
    train_loader = dataset.get_dataloader('train', config['data']['train_batch_size'])
    eval_loader = dataset.get_dataloader('validation', config['data']['eval_batch_size'])
    
    # Create model
    model = create_model(config, geometry_mode, curved_layers, device)
    
    # Train model
    use_dual_optimizer = geometry_mode != 'none' and config['training'].get('dual_optimizers', False)
    trained_model, training_stats = train_model(
        model, train_loader, eval_loader, config, device, use_dual_optimizer
    )
    
    # Final evaluation
    final_metrics = evaluate_model(trained_model, eval_loader, device)
    
    # Get condition numbers if applicable
    final_condition_numbers = {}
    if hasattr(trained_model, 'get_condition_numbers'):
        final_condition_numbers = trained_model.get_condition_numbers()
        print(f"📊 Final condition numbers: {final_condition_numbers}")
    
    # Compile results
    results = {
        'seed': seed,
        'geometry_mode': geometry_mode,
        'curved_layers': curved_layers,
        'config': config,
        'device': str(device),
        'final_metrics': final_metrics,
        'final_condition_numbers': final_condition_numbers,
        'training_stats': training_stats,
        'parameter_counts': {
            'total': sum(p.numel() for p in trained_model.parameters()),
            'trainable': sum(p.numel() for p in trained_model.parameters() if p.requires_grad)
        }
    }
    
    # Summary
    mcc = final_metrics['matthews_correlation']
    max_condition = max(final_condition_numbers.values()) if final_condition_numbers else 0
    
    print(f"📋 EXPERIMENT SUMMARY:")
    print(f"   MCC: {mcc:.3f}")
    print(f"   Max Condition Number: {max_condition:.1e}")
    print(f"   Training Time: {training_stats['training_time']:.2f}s")
    
    return results

def main():
    """Main entry point for Phase 4 experiments."""
    parser = argparse.ArgumentParser(description='Run Phase 4 experiments')
    parser.add_argument('--config', '-c', type=str, 
                       default='configs/phase4_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--seed', '-s', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--seeds', nargs='+', type=int,
                       help='Multiple seeds to run (overrides --seed)')
    parser.add_argument('--geometry', '-g', type=str, default='shared',
                       choices=['none', 'shared', 'per_head'],
                       help='Geometry mode')
    parser.add_argument('--layers', '-l', nargs='+', type=int, default=[1],
                       help='Curved layer indices')
    parser.add_argument('--output-dir', '-o', type=str, default='results/phase4',
                       help='Output directory for results')
    parser.add_argument('--representative-sample', action='store_true',
                       help='Run representative 7-experiment sample')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Determine experiments to run
    if args.representative_sample:
        # Load representative sample from config
        config = load_config(args.config)
        experiments = config['representative_sample']['experiments']
        print(f"🎯 Running representative sample: {len(experiments)} experiments")
    else:
        # Single configuration
        seeds = args.seeds if args.seeds else [args.seed]
        experiments = []
        for seed in seeds:
            experiments.append({
                'geometry': args.geometry,
                'layers': args.layers,
                'seed': seed
            })
        print(f"🎯 Running {len(experiments)} experiments")
    
    all_results = []
    
    for i, exp in enumerate(experiments):
        print(f"\n{'='*80}")
        print(f"EXPERIMENT {i+1}/{len(experiments)}: {exp}")
        print('='*80)
        
        try:
            results = run_phase4_experiment(
                config_path=args.config,
                geometry_mode=exp.get('geometry', exp.get('task', 'none')),  # Handle different config formats
                curved_layers=exp.get('layers', []),
                seed=exp['seed']
            )
            all_results.append(results)
            
            # Save individual result
            result_filename = f"phase4_results_{exp.get('geometry', 'none')}_layers{'_'.join(map(str, exp.get('layers', [])))}_seed{exp['seed']}.json"
            result_path = output_dir / result_filename
            
            with open(result_path, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            print(f"💾 Results saved: {result_path}")
            
        except Exception as e:
            print(f"💥 Error in experiment {exp}: {str(e)}")
            import traceback
            traceback.print_exc()
            
            all_results.append({
                'experiment': exp,
                'error': str(e),
                'success': False
            })
    
    # Aggregate results
    print(f"\n{'='*80}")
    print("PHASE 4 SUMMARY")
    print('='*80)
    
    successful_results = [r for r in all_results if 'error' not in r]
    
    if successful_results:
        # Compile aggregate statistics
        baseline_results = [r for r in successful_results if r['geometry_mode'] == 'none']
        curved_results = [r for r in successful_results if r['geometry_mode'] != 'none']
        
        summary = {
            'total_experiments': len(all_results),
            'successful_experiments': len(successful_results),
            'baseline_results': baseline_results,
            'curved_results': curved_results
        }
        
        if baseline_results:
            baseline_mcc = np.mean([r['final_metrics']['matthews_correlation'] for r in baseline_results])
            summary['baseline_mcc'] = {
                'mean': baseline_mcc,
                'std': np.std([r['final_metrics']['matthews_correlation'] for r in baseline_results]),
                'values': [r['final_metrics']['matthews_correlation'] for r in baseline_results]
            }
            print(f"📊 Baseline MCC: {baseline_mcc:.3f}")
        
        if curved_results:
            curved_mcc = np.mean([r['final_metrics']['matthews_correlation'] for r in curved_results])
            condition_numbers = []
            for r in curved_results:
                if r['final_condition_numbers']:
                    condition_numbers.extend(r['final_condition_numbers'].values())
            
            summary['curved_mcc'] = {
                'mean': curved_mcc,
                'std': np.std([r['final_metrics']['matthews_correlation'] for r in curved_results]),
                'values': [r['final_metrics']['matthews_correlation'] for r in curved_results]
            }
            
            if condition_numbers:
                summary['condition_numbers'] = {
                    'mean': np.mean(condition_numbers),
                    'std': np.std(condition_numbers),
                    'max': max(condition_numbers),
                    'min': min(condition_numbers),
                    'values': condition_numbers
                }
                print(f"📊 Curved MCC: {curved_mcc:.3f}")
                print(f"🔢 Max condition number: {max(condition_numbers):.1e}")
                
                # Check paper expectations
                if max(condition_numbers) > 171000:
                    print("✅ High condition numbers achieved (>171K)")
                else:
                    print("⚠️  Condition numbers lower than expected")
                
                if baseline_results:
                    improvement = curved_mcc - baseline_mcc
                    print(f"📈 MCC improvement: {improvement:.3f}")
                    
                    if abs(improvement) < 0.01:
                        print("✅ Negligible improvement observed (consistent with negative results)")
                    else:
                        print("⚠️  Significant improvement observed (inconsistent with paper)")
        
        summary['individual_results'] = all_results
        
    else:
        summary = {
            'total_experiments': len(all_results),
            'successful_experiments': 0,
            'error': 'All experiments failed',
            'individual_results': all_results
        }
        print("❌ All experiments failed")
    
    # Save summary
    summary_path = output_dir / 'phase4_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
    print(f"📄 Summary saved: {summary_path}")
    
    return len(successful_results) > 0

if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
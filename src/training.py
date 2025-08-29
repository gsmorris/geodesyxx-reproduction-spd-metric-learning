"""
Dual Optimizer Training Setup for SPD-Weighted Transformers
Implements the training methodology from Phase 4 of the Geodesyxx paper.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
import warnings
import time
from dataclasses import dataclass
from sklearn.metrics import matthews_corrcoef, accuracy_score, f1_score
import gc


@dataclass
class TrainingConfig:
    """Training configuration matching Phase 4 specifications."""
    
    # Learning rates (from paper)
    geometric_lr: float = 1e-4
    standard_lr: float = 1e-5
    
    # Early stopping
    patience: int = 3
    min_delta: float = 1e-4
    
    # Batch sizes
    batch_size: int = 16
    eval_batch_size: int = 32
    
    # Memory management
    max_memory_gb: float = 8.0
    memory_check_frequency: int = 10
    
    # Stability monitoring
    max_condition_number: float = 1e6
    gradient_clip_norm: float = 1.0
    stability_check_frequency: int = 5
    
    # Statistical tracking
    compute_effect_sizes: bool = True
    confidence_level: float = 0.95
    bonferroni_alpha: float = 0.017
    
    # Device management
    device_preference: List[str] = None
    cpu_fallback: bool = True
    
    def __post_init__(self):
        if self.device_preference is None:
            self.device_preference = ['mps', 'cuda', 'cpu']


class DeviceManager:
    """Handles device selection and memory management across MPS/CUDA/CPU."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = self._select_optimal_device()
        self.memory_stats = {}
        
    def _select_optimal_device(self) -> torch.device:
        """Select optimal device based on availability and preference."""
        
        for device_name in self.config.device_preference:
            if device_name == 'mps' and torch.backends.mps.is_available():
                try:
                    # Test MPS functionality
                    x = torch.randn(100, 100, device='mps')
                    y = torch.matmul(x, x.t())
                    del x, y
                    return torch.device('mps')
                except Exception as e:
                    warnings.warn(f"MPS test failed: {e}")
                    
            elif device_name == 'cuda' and torch.cuda.is_available():
                try:
                    # Test CUDA functionality
                    x = torch.randn(100, 100, device='cuda')
                    y = torch.matmul(x, x.t())
                    del x, y
                    return torch.device('cuda')
                except Exception as e:
                    warnings.warn(f"CUDA test failed: {e}")
                    
            elif device_name == 'cpu':
                return torch.device('cpu')
        
        # Fallback to CPU if all else fails
        return torch.device('cpu')
    
    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage in GB."""
        usage = {}
        
        if self.device.type == 'cuda':
            usage['allocated'] = torch.cuda.memory_allocated() / 1e9
            usage['reserved'] = torch.cuda.memory_reserved() / 1e9
            usage['max_allocated'] = torch.cuda.max_memory_allocated() / 1e9
        elif self.device.type == 'mps':
            try:
                usage['allocated'] = torch.mps.current_allocated_memory() / 1e9
                usage['max_allocated'] = usage['allocated']  # MPS doesn't track reserved
            except:
                usage['allocated'] = 0.0
                usage['max_allocated'] = 0.0
        else:
            # CPU memory is harder to track precisely
            usage['allocated'] = 0.0
            usage['max_allocated'] = 0.0
        
        return usage
    
    def check_memory_usage(self) -> bool:
        """Check if memory usage is within limits."""
        usage = self.get_memory_usage()
        
        if 'allocated' in usage and usage['allocated'] > self.config.max_memory_gb:
            warnings.warn(f"Memory usage {usage['allocated']:.2f}GB exceeds limit {self.config.max_memory_gb}GB")
            return False
        
        return True
    
    def cleanup_memory(self):
        """Clean up GPU memory."""
        gc.collect()
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
        elif self.device.type == 'mps':
            try:
                torch.mps.empty_cache()
            except:
                pass


class StabilityMonitor:
    """Monitors training stability for geometric parameters."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.condition_history = []
        self.gradient_norms = []
        self.nan_detections = 0
        self.inf_detections = 0
        self.warnings_issued = 0
        
    def check_model_stability(self, model: nn.Module) -> Dict[str, Any]:
        """Check model for numerical stability issues."""
        stability_report = {
            'has_nan': False,
            'has_inf': False,
            'condition_numbers': {},
            'max_condition': 0.0,
            'gradient_norms': {},
            'stability_ok': True
        }
        
        # Check for NaN/Inf in parameters
        for name, param in model.named_parameters():
            if param.grad is not None:
                if torch.isnan(param.grad).any():
                    stability_report['has_nan'] = True
                    self.nan_detections += 1
                if torch.isinf(param.grad).any():
                    stability_report['has_inf'] = True
                    self.inf_detections += 1
                
                # Track gradient norms
                grad_norm = param.grad.norm().item()
                stability_report['gradient_norms'][name] = grad_norm
        
        # Check condition numbers for SPD metrics
        if hasattr(model, 'get_condition_numbers'):
            try:
                conditions = model.get_condition_numbers()
                stability_report['condition_numbers'] = conditions
                if conditions:
                    stability_report['max_condition'] = max(conditions.values())
                    
                    # Check for extremely high condition numbers
                    if stability_report['max_condition'] > self.config.max_condition_number:
                        stability_report['stability_ok'] = False
                        self.warnings_issued += 1
            except:
                pass
        
        # Overall stability assessment
        if stability_report['has_nan'] or stability_report['has_inf']:
            stability_report['stability_ok'] = False
        
        return stability_report
    
    def get_statistics(self) -> Dict[str, float]:
        """Get stability statistics."""
        return {
            'nan_detections': self.nan_detections,
            'inf_detections': self.inf_detections,
            'warnings_issued': self.warnings_issued,
            'avg_condition': np.mean(self.condition_history) if self.condition_history else 0.0,
            'max_condition': max(self.condition_history) if self.condition_history else 0.0
        }


class EffectSizeCalculator:
    """Calculate Cohen's d effect sizes for statistical analysis."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        
    def cohens_d(self, group1: List[float], group2: List[float]) -> float:
        """Calculate Cohen's d effect size."""
        if len(group1) < 2 or len(group2) < 2:
            return 0.0
        
        n1, n2 = len(group1), len(group2)
        mean1, mean2 = np.mean(group1), np.mean(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        
        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
        
        if pooled_std == 0:
            return 0.0
        
        return (mean1 - mean2) / pooled_std
    
    def bootstrap_ci(
        self,
        data: List[float],
        confidence: float = 0.95,
        n_bootstrap: int = 1000
    ) -> Tuple[float, float]:
        """Calculate bootstrap confidence interval."""
        if len(data) < 2:
            return (0.0, 0.0)
        
        np.random.seed(42)  # Reproducible results
        bootstrap_means = []
        
        for _ in range(n_bootstrap):
            sample = np.random.choice(data, size=len(data), replace=True)
            bootstrap_means.append(np.mean(sample))
        
        alpha = 1 - confidence
        lower = np.percentile(bootstrap_means, 100 * alpha / 2)
        upper = np.percentile(bootstrap_means, 100 * (1 - alpha / 2))
        
        return (lower, upper)


class DualOptimizerTrainer:
    """
    Main trainer with dual optimizer setup.
    Implements the training methodology from Phase 4 Geodesyxx paper.
    """
    
    def __init__(self, model: nn.Module, config: TrainingConfig):
        self.model = model
        self.config = config
        self.device_manager = DeviceManager(config)
        self.stability_monitor = StabilityMonitor(config)
        self.effect_calculator = EffectSizeCalculator(config)
        
        # Move model to device
        self.model = self.model.to(self.device_manager.device)
        
        # Set up dual optimizers
        self.optimizers = self._setup_dual_optimizers()
        
        # Training state
        self.step_count = 0
        self.epoch_count = 0
        self.best_metric = -float('inf')
        self.patience_counter = 0
        self.early_stopping = False
        
        # Statistics tracking
        self.training_history = {
            'loss': [],
            'accuracy': [],
            'mcc': [],
            'condition_numbers': [],
            'memory_usage': [],
            'effect_sizes': []
        }
        
    def _setup_dual_optimizers(self) -> Dict[str, optim.Optimizer]:
        """Set up separate optimizers for geometric and standard parameters."""
        optimizers = {}
        
        # Separate parameters
        geometric_params = []
        standard_params = []
        
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # Check if this is a geometric parameter (SPD metric A matrix)
                if 'spd_metrics' in name and '.A' in name:
                    geometric_params.append(param)
                else:
                    standard_params.append(param)
        
        # Create optimizers with different learning rates
        if geometric_params:
            optimizers['geometric'] = optim.Adam(
                geometric_params,
                lr=self.config.geometric_lr,
                weight_decay=1e-4
            )
        
        if standard_params:
            optimizers['standard'] = optim.Adam(
                standard_params,
                lr=self.config.standard_lr,
                weight_decay=1e-4
            )
        
        print(f"Dual optimizers set up:")
        print(f"  Geometric parameters: {len(geometric_params)} ({self.config.geometric_lr:.1e} LR)")
        print(f"  Standard parameters: {len(standard_params)} ({self.config.standard_lr:.1e} LR)")
        
        return optimizers
    
    def train_epoch(
        self,
        train_loader: DataLoader,
        loss_fn: Callable,
        epoch: int
    ) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        epoch_losses = []
        epoch_predictions = []
        epoch_labels = []
        
        for batch_idx, batch in enumerate(train_loader):
            # Move batch to device
            if isinstance(batch, dict):
                batch = {k: v.to(self.device_manager.device) if torch.is_tensor(v) else v 
                        for k, v in batch.items()}
            else:
                batch = [item.to(self.device_manager.device) if torch.is_tensor(item) else item 
                        for item in batch]
            
            # Forward pass
            if isinstance(batch, dict):
                outputs = self.model(**batch)
                loss = loss_fn(outputs.logits, batch['labels'])
                predictions = torch.argmax(outputs.logits, dim=-1)
                labels = batch['labels']
            else:
                # Handle tuple/list batch format
                inputs, labels = batch[0], batch[1]
                outputs = self.model(inputs)
                loss = loss_fn(outputs, labels)
                predictions = torch.argmax(outputs, dim=-1)
            
            # Zero gradients for both optimizers
            for optimizer in self.optimizers.values():
                optimizer.zero_grad()
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.config.gradient_clip_norm
            )
            
            # Check stability before optimizer step
            if self.step_count % self.config.stability_check_frequency == 0:
                stability = self.stability_monitor.check_model_stability(self.model)
                
                if not stability['stability_ok']:
                    warnings.warn(f"Stability issues detected at step {self.step_count}")
                    
                    # Clip SPD spectrum if available
                    if hasattr(self.model, 'clip_geometric_spectrum'):
                        self.model.clip_geometric_spectrum()
            
            # Optimizer steps
            for optimizer in self.optimizers.values():
                optimizer.step()
            
            # Track metrics
            epoch_losses.append(loss.item())
            epoch_predictions.extend(predictions.cpu().numpy())
            epoch_labels.extend(labels.cpu().numpy())
            
            # Memory check
            if batch_idx % self.config.memory_check_frequency == 0:
                if not self.device_manager.check_memory_usage():
                    self.device_manager.cleanup_memory()
            
            self.step_count += 1
        
        # Calculate epoch metrics
        epoch_loss = np.mean(epoch_losses)
        epoch_acc = accuracy_score(epoch_labels, epoch_predictions)
        epoch_mcc = matthews_corrcoef(epoch_labels, epoch_predictions)
        
        # Update history
        self.training_history['loss'].append(epoch_loss)
        self.training_history['accuracy'].append(epoch_acc)
        self.training_history['mcc'].append(epoch_mcc)
        
        # Track memory usage
        memory_usage = self.device_manager.get_memory_usage()
        self.training_history['memory_usage'].append(memory_usage.get('allocated', 0.0))
        
        # Track condition numbers
        if hasattr(self.model, 'get_condition_numbers'):
            conditions = self.model.get_condition_numbers()
            self.training_history['condition_numbers'].append(conditions)
        
        return {
            'loss': epoch_loss,
            'accuracy': epoch_acc,
            'mcc': epoch_mcc,
            'memory_gb': memory_usage.get('allocated', 0.0)
        }
    
    def evaluate(
        self,
        eval_loader: DataLoader,
        loss_fn: Callable
    ) -> Dict[str, float]:
        """Evaluate the model."""
        self.model.eval()
        
        eval_losses = []
        eval_predictions = []
        eval_labels = []
        
        with torch.no_grad():
            for batch in eval_loader:
                # Move batch to device
                if isinstance(batch, dict):
                    batch = {k: v.to(self.device_manager.device) if torch.is_tensor(v) else v 
                            for k, v in batch.items()}
                    
                    outputs = self.model(**batch)
                    loss = loss_fn(outputs.logits, batch['labels'])
                    predictions = torch.argmax(outputs.logits, dim=-1)
                    labels = batch['labels']
                else:
                    batch = [item.to(self.device_manager.device) if torch.is_tensor(item) else item 
                            for item in batch]
                    inputs, labels = batch[0], batch[1]
                    
                    outputs = self.model(inputs)
                    loss = loss_fn(outputs, labels)
                    predictions = torch.argmax(outputs, dim=-1)
                
                eval_losses.append(loss.item())
                eval_predictions.extend(predictions.cpu().numpy())
                eval_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        eval_loss = np.mean(eval_losses)
        eval_acc = accuracy_score(eval_labels, eval_predictions)
        eval_mcc = matthews_corrcoef(eval_labels, eval_predictions)
        eval_f1 = f1_score(eval_labels, eval_predictions, average='weighted')
        
        return {
            'loss': eval_loss,
            'accuracy': eval_acc,
            'mcc': eval_mcc,
            'f1': eval_f1
        }
    
    def check_early_stopping(self, metric_value: float) -> bool:
        """Check if early stopping criteria are met."""
        if metric_value > self.best_metric + self.config.min_delta:
            self.best_metric = metric_value
            self.patience_counter = 0
        else:
            self.patience_counter += 1
        
        if self.patience_counter >= self.config.patience:
            self.early_stopping = True
            return True
        
        return False
    
    def compute_effect_sizes(self, baseline_scores: List[float]) -> Dict[str, float]:
        """Compute effect sizes compared to baseline."""
        if not self.config.compute_effect_sizes or not baseline_scores:
            return {}
        
        current_scores = self.training_history['mcc'][-5:]  # Last 5 epochs
        
        if len(current_scores) < 2:
            return {}
        
        effect_size = self.effect_calculator.cohens_d(current_scores, baseline_scores)
        ci_lower, ci_upper = self.effect_calculator.bootstrap_ci(current_scores)
        
        return {
            'cohens_d': effect_size,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'significant': abs(effect_size) > 0.2  # Small effect threshold
        }
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get comprehensive training summary."""
        stability_stats = self.stability_monitor.get_statistics()
        
        summary = {
            'epochs_trained': self.epoch_count,
            'steps_trained': self.step_count,
            'early_stopped': self.early_stopping,
            'best_mcc': self.best_metric,
            'device': str(self.device_manager.device),
            'stability': stability_stats,
            'memory_peak_gb': max(self.training_history['memory_usage']) if self.training_history['memory_usage'] else 0.0
        }
        
        # Add condition number statistics
        if self.training_history['condition_numbers']:
            all_conditions = []
            for epoch_conditions in self.training_history['condition_numbers']:
                if isinstance(epoch_conditions, dict):
                    all_conditions.extend(epoch_conditions.values())
            
            if all_conditions:
                summary['condition_stats'] = {
                    'mean': np.mean(all_conditions),
                    'max': max(all_conditions),
                    'std': np.std(all_conditions)
                }
        
        return summary


# Utility functions for CoLA task setup
def create_cola_loss_function():
    """Create loss function for CoLA task."""
    return nn.CrossEntropyLoss()


def create_cola_metrics():
    """Create metrics for CoLA evaluation."""
    def compute_metrics(predictions, labels):
        accuracy = accuracy_score(labels, predictions)
        mcc = matthews_corrcoef(labels, predictions)
        return {
            'accuracy': accuracy,
            'mcc': mcc
        }
    return compute_metrics


if __name__ == "__main__":
    # Test device compatibility
    config = TrainingConfig()
    device_manager = DeviceManager(config)
    
    print(f"Selected device: {device_manager.device}")
    print(f"Memory usage: {device_manager.get_memory_usage()}")
    
    # Test stability monitor
    stability_monitor = StabilityMonitor(config)
    print(f"Stability monitor initialized")
    
    print("✅ Training components test passed!")
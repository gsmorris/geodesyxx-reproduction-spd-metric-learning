"""
Synthetic Validation for SPD Metric Learning
Validates eigenvalue recovery as per Geodesyxx paper specifications.
Target: >99% correlation between learned and true eigenvalues.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple, Dict, List, Optional
from .spd_metric import SPDMetric
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
import random


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class SyntheticDataGenerator:
    """
    Generate synthetic data with known anisotropic structure.
    Following Geodesyxx paper specifications exactly.
    """
    
    def __init__(
        self,
        dimension: int = 10,
        n_samples: int = 200,
        true_eigenvalues: Optional[List[float]] = None,
        seed: int = 42
    ):
        """
        Initialize synthetic data generator.
        
        Args:
            dimension: Embedding dimension
            n_samples: Number of samples to generate
            true_eigenvalues: True metric eigenvalues (default: [9.0, 3.0, 1.0, ...])
            seed: Random seed
        """
        self.dimension = dimension
        self.n_samples = n_samples
        self.seed = seed
        
        # Set default eigenvalues as per paper
        if true_eigenvalues is None:
            self.true_eigenvalues = [9.0, 3.0] + [1.0] * (dimension - 2)
        else:
            self.true_eigenvalues = true_eigenvalues
            
        assert len(self.true_eigenvalues) == dimension
        
        # Create true metric tensor
        self.true_metric = torch.diag(torch.tensor(self.true_eigenvalues, dtype=torch.float32))
        
        # Set seed
        set_seed(seed)
        
    def generate_embeddings(self) -> torch.Tensor:
        """
        Generate embeddings with structure induced by true metric.
        
        Returns:
            torch.Tensor: Embeddings of shape (n_samples, dimension)
        """
        # Generate embeddings from multivariate normal
        # Covariance is inverse of metric (precision matrix)
        covariance = torch.inverse(self.true_metric)
        
        # Use Cholesky decomposition for sampling
        L = torch.linalg.cholesky(covariance)
        
        # Generate standard normal samples
        z = torch.randn(self.n_samples, self.dimension)
        
        # Transform to have desired covariance
        embeddings = z @ L.t()
        
        return embeddings
    
    def compute_true_distances(self, X: torch.Tensor) -> torch.Tensor:
        """
        Compute true Mahalanobis distances using known metric.
        
        Args:
            X: Embeddings of shape (n_samples, dimension)
            
        Returns:
            torch.Tensor: Distance matrix of shape (n_samples, n_samples)
        """
        n = X.shape[0]
        distances = torch.zeros(n, n)
        
        for i in range(n):
            for j in range(i+1, n):
                diff = X[i] - X[j]
                dist_squared = diff @ self.true_metric @ diff
                distances[i, j] = torch.sqrt(dist_squared)
                distances[j, i] = distances[i, j]
                
        return distances
    
    def generate_triplets(
        self,
        embeddings: torch.Tensor,
        n_triplets: int = 6000,
        strategy: str = 'distance_ranking'
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Generate triplets based on true distance rankings.
        
        Args:
            embeddings: Embeddings of shape (n_samples, dimension)
            n_triplets: Number of triplets to generate
            strategy: Triplet generation strategy
            
        Returns:
            Tuple of (anchors, positives, negatives) each of shape (n_triplets, dimension)
        """
        # Compute true distances
        distances = self.compute_true_distances(embeddings)
        n = embeddings.shape[0]
        
        anchors_list = []
        positives_list = []
        negatives_list = []
        
        for _ in range(n_triplets):
            # Select anchor
            anchor_idx = np.random.randint(n)
            
            # Get distances from anchor to all other points
            anchor_distances = distances[anchor_idx].numpy()
            
            # Sort indices by distance
            sorted_indices = np.argsort(anchor_distances)
            
            # Select positive from nearest neighbors (excluding self)
            positive_candidates = sorted_indices[1:n//3]  # Nearest third
            if len(positive_candidates) > 0:
                positive_idx = np.random.choice(positive_candidates)
            else:
                positive_idx = sorted_indices[1]
            
            # Select negative from farther points
            negative_candidates = sorted_indices[2*n//3:]  # Farthest third
            if len(negative_candidates) > 0:
                negative_idx = np.random.choice(negative_candidates)
            else:
                negative_idx = sorted_indices[-1]
            
            anchors_list.append(embeddings[anchor_idx])
            positives_list.append(embeddings[positive_idx])
            negatives_list.append(embeddings[negative_idx])
        
        anchors = torch.stack(anchors_list)
        positives = torch.stack(positives_list)
        negatives = torch.stack(negatives_list)
        
        return anchors, positives, negatives


class TripletLoss(nn.Module):
    """
    Triplet loss for metric learning.
    L = max(0, d(a,p) - d(a,n) + margin)
    """
    
    def __init__(self, margin: float = 1.0):
        super().__init__()
        self.margin = margin
        
    def forward(
        self,
        pos_distances: torch.Tensor,
        neg_distances: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute triplet loss.
        
        Args:
            pos_distances: Distances between anchors and positives
            neg_distances: Distances between anchors and negatives
            
        Returns:
            torch.Tensor: Triplet loss
        """
        losses = torch.relu(pos_distances - neg_distances + self.margin)
        return losses.mean()


class MetricValidator:
    """
    Validate learned metric against ground truth.
    """
    
    def __init__(self, true_eigenvalues: List[float]):
        """
        Initialize validator.
        
        Args:
            true_eigenvalues: Ground truth eigenvalues
        """
        self.true_eigenvalues = torch.tensor(true_eigenvalues)
        
    def compute_eigenvalue_correlation(
        self,
        learned_metric: SPDMetric
    ) -> float:
        """
        Compute correlation between learned and true eigenvalues.
        
        Args:
            learned_metric: Learned SPD metric
            
        Returns:
            float: Pearson correlation coefficient
        """
        # Get learned eigenvalues
        learned_eigenvalues = learned_metric.get_eigenvalues()
        
        # Sort both sets of eigenvalues for comparison
        true_sorted = torch.sort(self.true_eigenvalues, descending=True)[0]
        learned_sorted = torch.sort(learned_eigenvalues, descending=True)[0]
        
        # Compute correlation
        correlation, _ = pearsonr(
            true_sorted.cpu().numpy(),
            learned_sorted.cpu().detach().numpy()
        )
        
        return correlation
    
    def compute_eigenvalue_mse(
        self,
        learned_metric: SPDMetric
    ) -> float:
        """
        Compute MSE between learned and true eigenvalues.
        
        Args:
            learned_metric: Learned SPD metric
            
        Returns:
            float: Mean squared error
        """
        learned_eigenvalues = learned_metric.get_eigenvalues()
        
        # Sort for comparison
        true_sorted = torch.sort(self.true_eigenvalues, descending=True)[0]
        learned_sorted = torch.sort(learned_eigenvalues, descending=True)[0]
        
        # Compute MSE
        mse = ((true_sorted - learned_sorted.cpu().detach()) ** 2).mean().item()
        
        return mse
    
    def plot_eigenvalue_comparison(
        self,
        learned_metric: SPDMetric,
        save_path: Optional[str] = None
    ):
        """
        Plot comparison of true vs learned eigenvalues.
        
        Args:
            learned_metric: Learned SPD metric
            save_path: Path to save plot
        """
        learned_eigenvalues = learned_metric.get_eigenvalues()
        
        # Sort for comparison
        true_sorted = torch.sort(self.true_eigenvalues, descending=True)[0].numpy()
        learned_sorted = torch.sort(learned_eigenvalues, descending=True)[0].cpu().detach().numpy()
        
        plt.figure(figsize=(10, 6))
        
        # Plot eigenvalues
        indices = np.arange(len(true_sorted))
        plt.subplot(1, 2, 1)
        plt.plot(indices, true_sorted, 'b-', label='True', linewidth=2)
        plt.plot(indices, learned_sorted, 'r--', label='Learned', linewidth=2)
        plt.xlabel('Eigenvalue Index')
        plt.ylabel('Eigenvalue')
        plt.title('Eigenvalue Comparison')
        plt.legend()
        plt.yscale('log')
        plt.grid(True, alpha=0.3)
        
        # Plot correlation
        plt.subplot(1, 2, 2)
        plt.scatter(true_sorted, learned_sorted, alpha=0.6)
        plt.plot([true_sorted.min(), true_sorted.max()],
                 [true_sorted.min(), true_sorted.max()],
                 'r--', alpha=0.5)
        plt.xlabel('True Eigenvalues')
        plt.ylabel('Learned Eigenvalues')
        plt.title(f'Correlation: {self.compute_eigenvalue_correlation(learned_metric):.4f}')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.show()


def train_metric_recovery(
    dimension: int = 10,
    n_samples: int = 200,
    n_triplets: int = 6000,
    n_epochs: int = 200,
    learning_rate: float = 0.01,
    margin: float = 1.0,
    rank: int = 10,
    seed: int = 42,
    device: Optional[torch.device] = None,
    verbose: bool = True
) -> Dict:
    """
    Train SPD metric to recover synthetic geometric structure.
    
    Args:
        dimension: Embedding dimension
        n_samples: Number of synthetic samples
        n_triplets: Number of training triplets
        n_epochs: Number of training epochs
        learning_rate: Learning rate for optimizer
        margin: Triplet loss margin
        rank: Rank of metric factorization
        seed: Random seed
        device: Device to train on
        verbose: Print training progress
        
    Returns:
        Dict containing trained metric, history, and validation results
    """
    # Set device
    if device is None:
        if torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            # Use CPU for now due to MPS eigenvalue compatibility
            device = torch.device('cpu')
    
    if verbose:
        print(f"Training on device: {device}")
    
    # Set seed
    set_seed(seed)
    
    # Generate synthetic data
    generator = SyntheticDataGenerator(
        dimension=dimension,
        n_samples=n_samples,
        seed=seed
    )
    
    embeddings = generator.generate_embeddings().to(device)
    
    # Generate triplets
    anchors, positives, negatives = generator.generate_triplets(
        embeddings.cpu(),
        n_triplets=n_triplets
    )
    anchors = anchors.to(device)
    positives = positives.to(device)
    negatives = negatives.to(device)
    
    # Initialize metric
    metric = SPDMetric(
        embedding_dim=dimension,
        rank=rank,
        epsilon=1e-6,
        device=device
    )
    
    # Initialize optimizer with gradient clipping
    optimizer = optim.Adam(metric.parameters(), lr=learning_rate)
    
    # Initialize loss
    triplet_loss_fn = TripletLoss(margin=margin)
    
    # Initialize validator
    validator = MetricValidator(generator.true_eigenvalues)
    
    # Training history
    history = {
        'loss': [],
        'correlation': [],
        'mse': [],
        'condition_number': []
    }
    
    # Training loop
    batch_size = 256
    n_batches = (n_triplets + batch_size - 1) // batch_size
    
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        
        # Shuffle triplets
        perm = torch.randperm(n_triplets)
        anchors = anchors[perm]
        positives = positives[perm]
        negatives = negatives[perm]
        
        for batch_idx in range(n_batches):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, n_triplets)
            
            # Get batch
            batch_anchors = anchors[start_idx:end_idx]
            batch_positives = positives[start_idx:end_idx]
            batch_negatives = negatives[start_idx:end_idx]
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Compute distances
            pos_distances = metric(batch_anchors, batch_positives, squared=False)
            neg_distances = metric(batch_anchors, batch_negatives, squared=False)
            
            # Compute loss
            loss = triplet_loss_fn(pos_distances, neg_distances)
            
            # Add regularization
            reg_loss = metric.regularization_loss(target_condition=10.0)
            total_loss = loss + 0.01 * reg_loss
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(metric.parameters(), max_norm=1.0)
            
            # Update
            optimizer.step()
            
            # Clip spectrum if needed
            if batch_idx % 10 == 0:
                metric.clip_spectrum(min_eigenvalue=1e-6)
            
            epoch_loss += loss.item()
        
        # Compute validation metrics
        with torch.no_grad():
            correlation = validator.compute_eigenvalue_correlation(metric)
            mse = validator.compute_eigenvalue_mse(metric)
            condition = metric.compute_condition_number()
        
        # Update history
        history['loss'].append(epoch_loss / n_batches)
        history['correlation'].append(correlation)
        history['mse'].append(mse)
        history['condition_number'].append(condition)
        
        # Print progress
        if verbose and (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}:")
            print(f"  Loss: {history['loss'][-1]:.4f}")
            print(f"  Correlation: {correlation:.4f}")
            print(f"  MSE: {mse:.4f}")
            print(f"  Condition: {condition:.1f}")
    
    # Final validation
    final_correlation = validator.compute_eigenvalue_correlation(metric)
    final_mse = validator.compute_eigenvalue_mse(metric)
    
    if verbose:
        print(f"\n{'='*50}")
        print(f"FINAL RESULTS:")
        print(f"  Eigenvalue Correlation: {final_correlation:.4f}")
        print(f"  Eigenvalue MSE: {final_mse:.6f}")
        print(f"  Success: {'✅' if final_correlation > 0.99 else '❌'}")
        
        # Print eigenvalue comparison
        learned_eigs = metric.get_eigenvalues()
        true_eigs = torch.tensor(generator.true_eigenvalues)
        
        print(f"\nEigenvalue Comparison:")
        print(f"  {'True':<10} {'Learned':<10} {'Error':<10}")
        for i in range(min(5, dimension)):
            t = true_eigs[i].item()
            l = learned_eigs[i].item()
            e = abs(t - l)
            print(f"  {t:<10.4f} {l:<10.4f} {e:<10.4f}")
    
    return {
        'metric': metric,
        'history': history,
        'final_correlation': final_correlation,
        'final_mse': final_mse,
        'validator': validator,
        'generator': generator,
        'success': final_correlation > 0.99
    }


def run_validation_suite(seeds: List[int] = [42, 123, 456]) -> Dict:
    """
    Run complete validation suite with multiple seeds.
    
    Args:
        seeds: List of random seeds to test
        
    Returns:
        Dict containing results for all seeds
    """
    results = {}
    
    print("="*60)
    print("RUNNING EIGENVALUE RECOVERY VALIDATION SUITE")
    print("="*60)
    
    for seed in seeds:
        print(f"\n🌱 Testing with seed {seed}...")
        print("-"*40)
        
        result = train_metric_recovery(
            dimension=10,
            n_samples=200,
            n_triplets=6000,
            n_epochs=200,
            learning_rate=0.01,
            margin=1.0,
            rank=10,
            seed=seed,
            verbose=True
        )
        
        results[seed] = result
    
    # Summary statistics
    correlations = [r['final_correlation'] for r in results.values()]
    mses = [r['final_mse'] for r in results.values()]
    successes = [r['success'] for r in results.values()]
    
    print("\n" + "="*60)
    print("VALIDATION SUITE SUMMARY")
    print("="*60)
    print(f"Seeds tested: {seeds}")
    print(f"Correlations: {correlations}")
    print(f"Mean correlation: {np.mean(correlations):.4f} ± {np.std(correlations):.4f}")
    print(f"Mean MSE: {np.mean(mses):.6f} ± {np.std(mses):.6f}")
    print(f"Success rate: {sum(successes)}/{len(successes)} ({100*sum(successes)/len(successes):.0f}%)")
    print(f"All passed (>99%): {'✅' if all(successes) else '❌'}")
    
    return results


if __name__ == "__main__":
    # Run single validation
    print("Running single validation with default parameters...")
    result = train_metric_recovery(seed=42, verbose=True)
    
    # Plot results if successful
    if result['success']:
        print("\n✅ Validation successful! Plotting results...")
        result['validator'].plot_eigenvalue_comparison(
            result['metric'],
            save_path='eigenvalue_recovery.png'
        )
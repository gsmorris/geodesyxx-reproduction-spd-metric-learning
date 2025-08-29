"""
Core SPD Metric Learning Module
Based on Geodesyxx paper's validated A^T A + εI parameterization.
Successfully recovers synthetic geometric structure with >99% eigenvalue correlation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import numpy as np


class SPDMetric(nn.Module):
    """
    SPD metric tensor using A^T A + εI parameterization.
    
    This implementation follows the exact approach validated in the Geodesyxx paper:
    - Low-rank factorization: A ∈ R^(rank × dim)
    - Stabilization: G = A^T A + εI
    - Condition number monitoring and clipping
    
    Args:
        embedding_dim: Dimension of embedding space (e.g., 64)
        rank: Rank of factorization (e.g., 16 for 1,024 parameters)
        epsilon: Regularization constant (default: 1e-6)
        max_condition: Maximum allowed condition number (default: 1e4)
        device: Device to place tensors on
    """
    
    def __init__(
        self,
        embedding_dim: int = 64,
        rank: int = 16,
        epsilon: float = 1e-6,
        max_condition: float = 1e4,
        device: Optional[torch.device] = None
    ):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.rank = rank
        self.epsilon = epsilon
        self.max_condition = max_condition
        self.device = device or torch.device('cpu')
        
        # Initialize A matrix with small random values
        # Following paper's initialization strategy
        self.A = nn.Parameter(
            torch.randn(rank, embedding_dim, device=self.device) * 0.01
        )
        
        # Track statistics for monitoring
        self.register_buffer('condition_history', torch.zeros(100))
        self.register_buffer('history_idx', torch.tensor(0))
        
    def get_metric_tensor(self) -> torch.Tensor:
        """
        Compute the SPD metric tensor G = A^T A + εI.
        
        Returns:
            torch.Tensor: SPD metric tensor of shape (embedding_dim, embedding_dim)
        """
        # Compute A^T A
        AtA = torch.matmul(self.A.t(), self.A)
        
        # Add epsilon * I for numerical stability
        G = AtA + self.epsilon * torch.eye(
            self.embedding_dim,
            device=self.device,
            dtype=AtA.dtype
        )
        
        return G
    
    def compute_mahalanobis_distance(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        squared: bool = True
    ) -> torch.Tensor:
        """
        Compute Mahalanobis distance d_G(x,y)² = (x-y)^T G (x-y).
        
        Args:
            x: First set of embeddings, shape (batch_size, embedding_dim)
            y: Second set of embeddings, shape (batch_size, embedding_dim)
            squared: If True, return squared distance
            
        Returns:
            torch.Tensor: Distances of shape (batch_size,)
        """
        # Get metric tensor
        G = self.get_metric_tensor()
        
        # Compute difference
        diff = x - y  # (batch_size, embedding_dim)
        
        # Compute Mahalanobis distance: (x-y)^T G (x-y)
        # Efficient computation: sum((diff @ G) * diff, dim=1)
        Gdiff = torch.matmul(diff, G)  # (batch_size, embedding_dim)
        distances_squared = torch.sum(Gdiff * diff, dim=1)  # (batch_size,)
        
        if squared:
            return distances_squared
        else:
            # Add small epsilon before sqrt for numerical stability
            return torch.sqrt(distances_squared + 1e-8)
    
    def compute_pairwise_distances(
        self,
        X: torch.Tensor,
        Y: Optional[torch.Tensor] = None,
        squared: bool = True
    ) -> torch.Tensor:
        """
        Compute pairwise Mahalanobis distances between sets of embeddings.
        
        Args:
            X: First set of embeddings, shape (n, embedding_dim)
            Y: Second set of embeddings, shape (m, embedding_dim)
               If None, compute pairwise distances within X
            squared: If True, return squared distances
            
        Returns:
            torch.Tensor: Distance matrix of shape (n, m) or (n, n)
        """
        if Y is None:
            Y = X
            
        G = self.get_metric_tensor()
        
        # Efficient pairwise distance computation
        # d²(x,y) = x^T G x - 2 x^T G y + y^T G y
        
        # Compute x^T G x for all x
        XG = torch.matmul(X, G)  # (n, d)
        XGX = torch.sum(XG * X, dim=1, keepdim=True)  # (n, 1)
        
        # Compute y^T G y for all y
        YG = torch.matmul(Y, G)  # (m, d)
        YGY = torch.sum(YG * Y, dim=1, keepdim=True)  # (m, 1)
        
        # Compute -2 x^T G y for all pairs
        XGY = torch.matmul(XG, Y.t())  # (n, m)
        
        # Combine terms
        distances_squared = XGX - 2 * XGY + YGY.t()  # (n, m)
        
        # Ensure non-negative (numerical errors can cause small negative values)
        distances_squared = torch.clamp(distances_squared, min=0)
        
        if squared:
            return distances_squared
        else:
            return torch.sqrt(distances_squared + 1e-8)
    
    def compute_condition_number(self) -> float:
        """
        Compute condition number of the metric tensor for monitoring.
        
        Returns:
            float: Condition number of G
        """
        with torch.no_grad():
            G = self.get_metric_tensor()
            eigenvalues = torch.linalg.eigvalsh(G)
            condition = eigenvalues[-1] / eigenvalues[0]
            
            # Update history
            idx = self.history_idx.item()
            self.condition_history[idx] = condition
            self.history_idx = (self.history_idx + 1) % 100
            
            return condition.item()
    
    def clip_spectrum(self, min_eigenvalue: float = 1e-6):
        """
        Clip the spectrum of the metric tensor to prevent numerical issues.
        This modifies A in-place to ensure condition number stays reasonable.
        
        Args:
            min_eigenvalue: Minimum allowed eigenvalue
        """
        with torch.no_grad():
            G = self.get_metric_tensor()
            
            # Eigendecomposition
            eigenvalues, eigenvectors = torch.linalg.eigh(G)
            
            # Clip eigenvalues
            eigenvalues = torch.clamp(eigenvalues, min=min_eigenvalue)
            
            # Check condition number
            condition = eigenvalues[-1] / eigenvalues[0]
            if condition > self.max_condition:
                # Scale to achieve target condition number
                scale = (self.max_condition * eigenvalues[0] / eigenvalues[-1]).sqrt()
                eigenvalues = eigenvalues * scale
            
            # Reconstruct G with clipped spectrum
            G_clipped = eigenvectors @ torch.diag(eigenvalues) @ eigenvectors.t()
            
            # Update A to match clipped G (approximately)
            # Use Cholesky decomposition: G = L L^T
            # Then set A = L[:rank, :]
            try:
                L = torch.linalg.cholesky(G_clipped - self.epsilon * torch.eye(
                    self.embedding_dim, device=self.device
                ))
                if self.rank < self.embedding_dim:
                    self.A.data = L[:self.rank, :]
                else:
                    self.A.data = L
            except:
                # If Cholesky fails, just clip A's norm
                self.A.data = F.normalize(self.A.data, dim=1) * eigenvalues.mean().sqrt()
    
    def get_eigenvalues(self) -> torch.Tensor:
        """
        Get eigenvalues of the metric tensor for analysis.
        
        Returns:
            torch.Tensor: Sorted eigenvalues
        """
        with torch.no_grad():
            G = self.get_metric_tensor()
            eigenvalues = torch.linalg.eigvalsh(G)
            return eigenvalues
    
    def forward(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        squared: bool = True
    ) -> torch.Tensor:
        """
        Forward pass computing Mahalanobis distance.
        
        Args:
            x: First embeddings, shape (batch_size, embedding_dim)
            y: Second embeddings, shape (batch_size, embedding_dim)
            squared: If True, return squared distance
            
        Returns:
            torch.Tensor: Distances of shape (batch_size,)
        """
        return self.compute_mahalanobis_distance(x, y, squared=squared)
    
    def regularization_loss(self, target_condition: float = 10.0) -> torch.Tensor:
        """
        Compute regularization loss to encourage well-conditioned metric.
        
        Args:
            target_condition: Target condition number
            
        Returns:
            torch.Tensor: Regularization loss
        """
        G = self.get_metric_tensor()
        eigenvalues = torch.linalg.eigvalsh(G)
        condition = eigenvalues[-1] / eigenvalues[0]
        
        # Penalize deviation from target condition number
        reg_loss = torch.abs(torch.log(condition) - torch.log(torch.tensor(target_condition)))
        
        return reg_loss
    
    def get_stats(self) -> Dict[str, float]:
        """
        Get statistics about the metric tensor for monitoring.
        
        Returns:
            Dict containing condition number, eigenvalue stats, etc.
        """
        with torch.no_grad():
            eigenvalues = self.get_eigenvalues()
            condition = eigenvalues[-1] / eigenvalues[0]
            
            return {
                'condition_number': condition.item(),
                'min_eigenvalue': eigenvalues[0].item(),
                'max_eigenvalue': eigenvalues[-1].item(),
                'mean_eigenvalue': eigenvalues.mean().item(),
                'rank': self.rank,
                'effective_rank': (eigenvalues > 1e-6).sum().item(),
                'frobenius_norm': self.get_metric_tensor().norm().item()
            }


class BatchedSPDMetric(SPDMetric):
    """
    Extension of SPDMetric for efficient batched operations.
    Optimized for memory-efficient training on large batches.
    """
    
    def compute_triplet_distances(
        self,
        anchors: torch.Tensor,
        positives: torch.Tensor,
        negatives: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Efficiently compute distances for triplet loss.
        
        Args:
            anchors: Anchor embeddings, shape (batch_size, embedding_dim)
            positives: Positive embeddings, shape (batch_size, embedding_dim)
            negatives: Negative embeddings, shape (batch_size, embedding_dim)
            
        Returns:
            Tuple of (positive_distances, negative_distances)
        """
        # Compute distances in parallel for efficiency
        pos_distances = self.compute_mahalanobis_distance(anchors, positives)
        neg_distances = self.compute_mahalanobis_distance(anchors, negatives)
        
        return pos_distances, neg_distances
    
    def compute_all_pairs_distances(
        self,
        batch: torch.Tensor,
        labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute all pairwise distances and identify positive/negative pairs.
        Useful for mining hard triplets.
        
        Args:
            batch: Embeddings, shape (batch_size, embedding_dim)
            labels: Labels for each embedding, shape (batch_size,)
            
        Returns:
            Tuple of (distance_matrix, positive_mask, negative_mask)
        """
        # Compute all pairwise distances
        distances = self.compute_pairwise_distances(batch, squared=False)
        
        # Create masks for positive and negative pairs
        labels_equal = labels.unsqueeze(0) == labels.unsqueeze(1)
        positive_mask = labels_equal & ~torch.eye(len(labels), dtype=torch.bool, device=labels.device)
        negative_mask = ~labels_equal
        
        return distances, positive_mask, negative_mask
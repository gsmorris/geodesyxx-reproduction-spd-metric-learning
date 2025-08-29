"""
SPD-Weighted Attention for Transformers
Replaces standard dot-product attention with learned Mahalanobis distances.
Based on Phase 4 implementation from Geodesyxx paper.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import warnings
from typing import Optional, Tuple, Dict, Union, List
from .spd_metric import SPDMetric


class CurvedMultiHeadAttention(nn.Module):
    """
    Multi-head attention with SPD-weighted distance computation.
    
    Replaces scores = Q @ K^T / sqrt(d_k) with scores = -d_G(q_i, k_j)² / sqrt(d_k)
    where d_G is the learned Mahalanobis distance.
    
    Args:
        embed_dim: Total embedding dimension (768 for DistilBERT)
        num_heads: Number of attention heads (12 for DistilBERT)
        dropout: Dropout probability
        geometry_mode: 'none', 'shared', or 'per_head'
        rank: Rank for SPD metric factorization (default: 16)
        device: Device to place tensors on
    """
    
    def __init__(
        self,
        embed_dim: int = 768,
        num_heads: int = 12,
        dropout: float = 0.1,
        geometry_mode: str = 'none',
        rank: int = 16,
        device: Optional[torch.device] = None
    ):
        super().__init__()
        
        assert embed_dim % num_heads == 0, f"embed_dim {embed_dim} must be divisible by num_heads {num_heads}"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.geometry_mode = geometry_mode
        self.rank = rank
        self.scale = 1.0 / math.sqrt(self.head_dim)  # Temperature scaling
        
        # Set device with proper fallback handling
        self.device = self._get_optimal_device(device)
        
        # Standard attention projections
        self.q_proj = nn.Linear(embed_dim, embed_dim, device=self.device)
        self.k_proj = nn.Linear(embed_dim, embed_dim, device=self.device)
        self.v_proj = nn.Linear(embed_dim, embed_dim, device=self.device)
        self.out_proj = nn.Linear(embed_dim, embed_dim, device=self.device)
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize SPD metrics based on geometry mode
        self.spd_metrics = nn.ModuleDict()
        self._initialize_geometry()
        
        # Statistics tracking
        self.register_buffer('attention_stats', torch.zeros(6))  # [calls, avg_condition, max_condition, nan_count, inf_count, warning_count]
        self.register_buffer('step_count', torch.tensor(0))
        
    def _get_optimal_device(self, requested_device: Optional[torch.device]) -> torch.device:
        """Get optimal device with MPS support and CPU fallback."""
        if requested_device is not None:
            return requested_device
            
        # For now, default to CPU to avoid device mismatch issues
        # TODO: Fix MPS compatibility
        return torch.device('cpu')
    
    def _initialize_geometry(self):
        """Initialize SPD metric tensors based on geometry mode."""
        if self.geometry_mode == 'none':
            # No geometric parameters
            return
        elif self.geometry_mode == 'shared':
            # Single metric shared across all heads
            self.spd_metrics['shared'] = SPDMetric(
                embedding_dim=self.head_dim,
                rank=self.rank,
                epsilon=1e-6,
                max_condition=1e4,
                device=self.device
            )
        elif self.geometry_mode == 'per_head':
            # Separate metric for each head
            for i in range(self.num_heads):
                self.spd_metrics[f'head_{i}'] = SPDMetric(
                    embedding_dim=self.head_dim,
                    rank=self.rank,
                    epsilon=1e-6,
                    max_condition=1e4,
                    device=self.device
                )
        else:
            raise ValueError(f"Unknown geometry_mode: {self.geometry_mode}")
    
    def _compute_curved_attention_scores(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        head_idx: int
    ) -> torch.Tensor:
        """
        Compute SPD-weighted attention scores.
        
        Args:
            q: Query tensor, shape (batch_size, seq_len, head_dim)
            k: Key tensor, shape (batch_size, seq_len, head_dim)
            head_idx: Head index for per-head geometry
            
        Returns:
            torch.Tensor: Attention scores, shape (batch_size, seq_len, seq_len)
        """
        batch_size, seq_len, _ = q.shape
        
        # Ensure tensors are on correct device
        q = q.to(self.device)
        k = k.to(self.device)
        
        # Get appropriate SPD metric
        if self.geometry_mode == 'shared':
            metric = self.spd_metrics['shared']
        elif self.geometry_mode == 'per_head':
            metric = self.spd_metrics[f'head_{head_idx}']
        else:
            raise ValueError("Should not call curved scores with geometry_mode='none'")
        
        # Compute pairwise Mahalanobis distances
        # Reshape for efficient computation: (batch_size * seq_len, head_dim)
        q_flat = q.reshape(-1, self.head_dim)
        k_flat = k.reshape(-1, self.head_dim)
        
        # Compute all pairwise distances
        distances_squared = torch.zeros(batch_size * seq_len, batch_size * seq_len, device=self.device)
        
        # Process in chunks to manage memory
        chunk_size = min(256, seq_len)
        
        for i in range(0, batch_size * seq_len, chunk_size):
            end_i = min(i + chunk_size, batch_size * seq_len)
            q_chunk = q_flat[i:end_i]  # (chunk_size, head_dim)
            
            # Compute distances from this chunk to all k vectors
            dist_chunk = metric.compute_pairwise_distances(q_chunk, k_flat, squared=True)
            distances_squared[i:end_i] = dist_chunk
        
        # Reshape back to (batch_size, seq_len, seq_len) for each query-key pair within batch
        scores = torch.zeros(batch_size, seq_len, seq_len, device=self.device)
        
        for b in range(batch_size):
            start_idx = b * seq_len
            end_idx = (b + 1) * seq_len
            
            # Extract the block diagonal for this batch element
            batch_distances = distances_squared[start_idx:end_idx, start_idx:end_idx]
            
            # Convert to attention scores: -d²/sqrt(d_k)
            scores[b] = -batch_distances * self.scale
        
        # Update statistics
        self._update_statistics(metric)
        
        return scores
    
    def _compute_standard_attention_scores(
        self,
        q: torch.Tensor,
        k: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute standard dot-product attention scores.
        
        Args:
            q: Query tensor, shape (batch_size, seq_len, head_dim)
            k: Key tensor, shape (batch_size, seq_len, head_dim)
            
        Returns:
            torch.Tensor: Attention scores, shape (batch_size, seq_len, seq_len)
        """
        # Standard scaled dot-product: Q @ K^T / sqrt(d_k)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        return scores
    
    def _update_statistics(self, metric: SPDMetric):
        """Update attention statistics for monitoring."""
        self.step_count += 1
        
        with torch.no_grad():
            try:
                condition = metric.compute_condition_number()
                
                # Update running statistics
                self.attention_stats[0] += 1  # call count
                
                # Running average of condition numbers
                alpha = 0.01  # Smoothing factor
                self.attention_stats[1] = (1 - alpha) * self.attention_stats[1] + alpha * condition
                
                # Max condition number
                self.attention_stats[2] = max(self.attention_stats[2], condition)
                
                # Check for numerical issues
                G = metric.get_metric_tensor()
                if torch.isnan(G).any():
                    self.attention_stats[3] += 1
                if torch.isinf(G).any():
                    self.attention_stats[4] += 1
                
                # Issue warnings for extreme condition numbers
                if condition > 1e5 and self.step_count % 100 == 0:
                    self.attention_stats[5] += 1
                    warnings.warn(f"High condition number detected: {condition:.1e}")
                    
            except Exception as e:
                warnings.warn(f"Error computing attention statistics: {e}")
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass with SPD-weighted or standard attention.
        
        Args:
            hidden_states: Input tensor, shape (batch_size, seq_len, embed_dim)
            attention_mask: Mask tensor, shape (batch_size, 1, 1, seq_len)
            head_mask: Head mask tensor (not implemented)
            output_attentions: Whether to output attention weights
            
        Returns:
            Tuple of (output, attention_weights)
        """
        batch_size, seq_len, embed_dim = hidden_states.shape
        
        # Ensure tensors are on the correct device
        hidden_states = hidden_states.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)
        
        # Project to Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        # Shape: (batch_size, num_heads, seq_len, head_dim)
        
        # Compute attention scores for each head
        all_attention_scores = []
        
        for head_idx in range(self.num_heads):
            q_head = q[:, head_idx]  # (batch_size, seq_len, head_dim)
            k_head = k[:, head_idx]  # (batch_size, seq_len, head_dim)
            
            if self.geometry_mode == 'none':
                scores = self._compute_standard_attention_scores(q_head, k_head)
            else:
                scores = self._compute_curved_attention_scores(q_head, k_head, head_idx)
            
            all_attention_scores.append(scores)
        
        # Stack attention scores: (batch_size, num_heads, seq_len, seq_len)
        attention_scores = torch.stack(all_attention_scores, dim=1)
        
        # Apply attention mask if provided
        if attention_mask is not None:
            attention_scores += attention_mask
        
        # Apply softmax to get attention weights
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, v)
        
        # Reshape and concatenate heads
        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, embed_dim
        )
        
        # Final output projection
        output = self.out_proj(context)
        
        outputs = (output,)
        if output_attentions:
            outputs += (attention_weights,)
        
        return outputs
    
    def get_geometric_parameters(self) -> Dict[str, torch.Tensor]:
        """Get all geometric parameters for separate optimization."""
        geometric_params = {}
        
        for name, metric in self.spd_metrics.items():
            geometric_params[f"{name}_A"] = metric.A
        
        return geometric_params
    
    def get_standard_parameters(self) -> Dict[str, torch.Tensor]:
        """Get standard attention parameters."""
        standard_params = {}
        
        for name, param in self.named_parameters():
            if not any(spd_name in name for spd_name in self.spd_metrics.keys()):
                standard_params[name] = param
        
        return standard_params
    
    def get_parameter_count(self) -> Dict[str, int]:
        """Get parameter counts matching paper's Table 3."""
        counts = {}
        
        # Standard attention parameters
        standard_count = (
            self.q_proj.weight.numel() + self.q_proj.bias.numel() +
            self.k_proj.weight.numel() + self.k_proj.bias.numel() +
            self.v_proj.weight.numel() + self.v_proj.bias.numel() +
            self.out_proj.weight.numel() + self.out_proj.bias.numel()
        )
        counts['standard'] = standard_count
        
        # Geometric parameters
        geometric_count = 0
        for metric in self.spd_metrics.values():
            geometric_count += metric.A.numel()
        counts['geometric'] = geometric_count
        
        counts['total'] = standard_count + geometric_count
        
        return counts
    
    def get_condition_numbers(self) -> Dict[str, float]:
        """Get current condition numbers for all SPD metrics."""
        conditions = {}
        
        with torch.no_grad():
            for name, metric in self.spd_metrics.items():
                try:
                    conditions[name] = metric.compute_condition_number()
                except:
                    conditions[name] = float('inf')
        
        return conditions
    
    def clip_geometric_spectrum(self, min_eigenvalue: float = 1e-6):
        """Clip spectrum of all SPD metrics for numerical stability."""
        for metric in self.spd_metrics.values():
            metric.clip_spectrum(min_eigenvalue)
    
    def get_attention_stats(self) -> Dict[str, float]:
        """Get attention statistics for monitoring."""
        stats = self.attention_stats.cpu().numpy()
        
        return {
            'total_calls': int(stats[0]),
            'avg_condition_number': float(stats[1]),
            'max_condition_number': float(stats[2]),
            'nan_detections': int(stats[3]),
            'inf_detections': int(stats[4]),
            'warning_count': int(stats[5]),
            'steps': int(self.step_count.item())
        }


class CurvedDistilBertAttention(nn.Module):
    """
    DistilBERT-compatible attention layer with SPD weighting.
    Drop-in replacement for transformers.DistilBertAttention.
    """
    
    def __init__(self, config, geometry_mode: str = 'none', rank: int = 16):
        super().__init__()
        
        self.attention = CurvedMultiHeadAttention(
            embed_dim=config.dim,
            num_heads=config.n_heads,
            dropout=config.attention_dropout,
            geometry_mode=geometry_mode,
            rank=rank
        )
        
        # DistilBERT specific attributes
        self.output_attentions = getattr(config, 'output_attentions', False)
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: torch.Tensor,
        head_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False
    ) -> Tuple[torch.Tensor, ...]:
        """
        DistilBERT-compatible forward pass.
        Note: DistilBERT passes Q, K, V separately, but we compute them internally.
        """
        # DistilBERT expects query as the main input
        # We'll use query as hidden_states and ignore separate K, V
        outputs = self.attention(
            hidden_states=query,
            attention_mask=mask,
            head_mask=head_mask,
            output_attentions=output_attentions or self.output_attentions
        )
        
        return outputs


def create_curved_attention_layer(
    config,
    geometry_mode: str = 'none',
    rank: int = 16,
    layer_idx: int = 0
) -> CurvedDistilBertAttention:
    """
    Factory function to create curved attention layer.
    
    Args:
        config: DistilBERT configuration
        geometry_mode: 'none', 'shared', or 'per_head'
        rank: Rank for SPD factorization
        layer_idx: Layer index (for debugging)
        
    Returns:
        CurvedDistilBertAttention layer
    """
    return CurvedDistilBertAttention(
        config=config,
        geometry_mode=geometry_mode,
        rank=rank
    )


def validate_device_compatibility() -> Dict[str, bool]:
    """
    Validate device compatibility for MPS/CUDA/CPU.
    
    Returns:
        Dict indicating which devices are available and working
    """
    compatibility = {
        'mps_available': False,
        'mps_working': False,
        'cuda_available': False,
        'cuda_working': False,
        'cpu_working': False
    }
    
    # Check MPS
    if torch.backends.mps.is_available():
        compatibility['mps_available'] = True
        try:
            x = torch.randn(10, 10, device='mps')
            y = torch.matmul(x, x.t())
            compatibility['mps_working'] = True
        except:
            pass
    
    # Check CUDA
    if torch.cuda.is_available():
        compatibility['cuda_available'] = True
        try:
            x = torch.randn(10, 10, device='cuda')
            y = torch.matmul(x, x.t())
            compatibility['cuda_working'] = True
        except:
            pass
    
    # Check CPU
    try:
        x = torch.randn(10, 10, device='cpu')
        y = torch.matmul(x, x.t())
        compatibility['cpu_working'] = True
    except:
        pass
    
    return compatibility


if __name__ == "__main__":
    # Quick test
    print("Testing CurvedMultiHeadAttention...")
    
    # Test device compatibility
    compat = validate_device_compatibility()
    print(f"Device compatibility: {compat}")
    
    # Create test layer
    attention = CurvedMultiHeadAttention(
        embed_dim=768,
        num_heads=12,
        geometry_mode='shared',
        rank=16
    )
    
    # Test forward pass
    batch_size, seq_len = 2, 128
    hidden_states = torch.randn(batch_size, seq_len, 768, device=attention.device)
    
    outputs = attention(hidden_states)
    print(f"Output shape: {outputs[0].shape}")
    
    # Test parameter counting
    param_counts = attention.get_parameter_count()
    print(f"Parameter counts: {param_counts}")
    
    # Test condition numbers
    if attention.geometry_mode != 'none':
        conditions = attention.get_condition_numbers()
        print(f"Condition numbers: {conditions}")
    
    print("✅ CurvedMultiHeadAttention test passed!")
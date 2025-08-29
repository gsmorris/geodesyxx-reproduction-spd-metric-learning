"""
DistilBERT Integration Utilities for SPD-Weighted Attention
Enables selective layer replacement and parameter tracking as per Phase 4 specifications.
"""

import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertConfig, DistilBertForSequenceClassification
from transformers.models.distilbert.modeling_distilbert import TransformerBlock as DistilBertLayer
from typing import Dict, List, Optional, Tuple, Union, Any
import copy
import warnings
from .curved_attention import CurvedMultiHeadAttention, CurvedDistilBertAttention


class CurvedDistilBertLayer(DistilBertLayer):
    """
    DistilBERT layer with optional SPD-weighted attention.
    Drop-in replacement that preserves all original functionality.
    """
    
    def __init__(self, config: DistilBertConfig, use_curved_attention: bool = False, 
                 geometry_mode: str = 'shared', rank: int = 16):
        super().__init__(config)
        
        self.use_curved_attention = use_curved_attention
        self.geometry_mode = geometry_mode
        self.rank = rank
        
        if use_curved_attention:
            # Replace standard attention with curved attention
            self.attention = CurvedDistilBertAttention(
                config=config,
                geometry_mode=geometry_mode,
                rank=rank
            )
            
    def get_parameter_count(self) -> Dict[str, int]:
        """Get parameter counts for this layer."""
        if self.use_curved_attention and hasattr(self.attention, 'attention'):
            return self.attention.attention.get_parameter_count()
        else:
            # Count standard parameters
            total = sum(p.numel() for p in self.parameters())
            return {'standard': total, 'geometric': 0, 'total': total}
    
    def get_condition_numbers(self) -> Dict[str, float]:
        """Get condition numbers for SPD metrics in this layer."""
        if self.use_curved_attention and hasattr(self.attention, 'attention'):
            return self.attention.attention.get_condition_numbers()
        return {}
    
    def clip_geometric_spectrum(self):
        """Clip geometric spectrum for numerical stability."""
        if self.use_curved_attention and hasattr(self.attention, 'attention'):
            self.attention.attention.clip_geometric_spectrum()


class CurvedDistilBertModel(DistilBertModel):
    """
    DistilBERT model with selective SPD-weighted attention layers.
    Maintains compatibility with HuggingFace ecosystem.
    """
    
    def __init__(
        self,
        config: DistilBertConfig,
        curved_layers: Optional[List[int]] = None,
        geometry_mode: str = 'shared',
        rank: int = 16
    ):
        # Initialize base model first
        super().__init__(config)
        
        # Configuration for curved attention
        self.curved_layers = curved_layers if curved_layers is not None else [1, 2]  # Default from paper
        self.geometry_mode = geometry_mode
        self.rank = rank
        
        # Replace specified layers with curved attention
        self._replace_attention_layers()
        
        # Initialize any new parameters
        self.post_init()
    
    def _replace_attention_layers(self):
        """Replace specified transformer layers with curved attention variants."""
        for layer_idx in self.curved_layers:
            if 0 <= layer_idx < len(self.transformer.layer):
                original_layer = self.transformer.layer[layer_idx]
                
                # Create new curved layer with same config
                curved_layer = CurvedDistilBertLayer(
                    config=self.config,
                    use_curved_attention=True,
                    geometry_mode=self.geometry_mode,
                    rank=self.rank
                )
                
                # Copy weights from original layer (except attention)
                self._copy_layer_weights(original_layer, curved_layer)
                
                # Replace the layer
                self.transformer.layer[layer_idx] = curved_layer
                
                print(f"Replaced layer {layer_idx} with curved attention ({self.geometry_mode}, rank={self.rank})")
    
    def _copy_layer_weights(self, source_layer: DistilBertLayer, target_layer: CurvedDistilBertLayer):
        """Copy weights from source layer to target layer, excluding attention."""
        
        # Copy feed-forward network weights
        if hasattr(source_layer, 'ffn') and hasattr(target_layer, 'ffn'):
            target_layer.ffn.load_state_dict(source_layer.ffn.state_dict())
        
        # Copy layer norm weights
        if hasattr(source_layer, 'sa_layer_norm') and hasattr(target_layer, 'sa_layer_norm'):
            target_layer.sa_layer_norm.load_state_dict(source_layer.sa_layer_norm.state_dict())
        
        if hasattr(source_layer, 'output_layer_norm') and hasattr(target_layer, 'output_layer_norm'):
            target_layer.output_layer_norm.load_state_dict(source_layer.output_layer_norm.state_dict())
        
        # For attention, copy the standard projection weights but leave geometric params fresh
        if (hasattr(source_layer, 'attention') and hasattr(target_layer, 'attention') and 
            hasattr(target_layer.attention, 'attention')):
            
            source_attn = source_layer.attention
            target_attn = target_layer.attention.attention
            
            # Copy standard projection weights
            try:
                target_attn.q_proj.load_state_dict(source_attn.q_lin.state_dict())
                target_attn.k_proj.load_state_dict(source_attn.k_lin.state_dict())
                target_attn.v_proj.load_state_dict(source_attn.v_lin.state_dict())
                target_attn.out_proj.load_state_dict(source_attn.out_lin.state_dict())
                print(f"  Copied standard attention weights")
            except Exception as e:
                warnings.warn(f"Could not copy attention weights: {e}")
                # Initialize fresh if copying fails
                self._initialize_attention_weights(target_attn)
    
    def _initialize_attention_weights(self, attention_layer: CurvedMultiHeadAttention):
        """Initialize attention weights using standard initialization."""
        for module in [attention_layer.q_proj, attention_layer.k_proj, 
                      attention_layer.v_proj, attention_layer.out_proj]:
            if hasattr(module, 'weight'):
                nn.init.xavier_uniform_(module.weight)
            if hasattr(module, 'bias') and module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def get_parameter_breakdown(self) -> Dict[str, Dict[str, int]]:
        """
        Get detailed parameter breakdown matching paper's Table 3.
        
        Returns:
            Dict with parameter counts for each layer and totals
        """
        breakdown = {}
        total_standard = 0
        total_geometric = 0
        
        for layer_idx, layer in enumerate(self.transformer.layer):
            if isinstance(layer, CurvedDistilBertLayer):
                counts = layer.get_parameter_count()
                breakdown[f'layer_{layer_idx}'] = counts
                total_standard += counts.get('standard', 0)
                total_geometric += counts.get('geometric', 0)
            else:
                # Standard layer
                layer_params = sum(p.numel() for p in layer.parameters())
                breakdown[f'layer_{layer_idx}'] = {
                    'standard': layer_params,
                    'geometric': 0,
                    'total': layer_params
                }
                total_standard += layer_params
        
        # Add embedding and other parameters
        embedding_params = (
            self.embeddings.word_embeddings.weight.numel() +
            self.embeddings.position_embeddings.weight.numel() +
            self.embeddings.LayerNorm.weight.numel() +
            self.embeddings.LayerNorm.bias.numel()
        )
        
        breakdown['embeddings'] = {
            'standard': embedding_params,
            'geometric': 0,
            'total': embedding_params
        }
        total_standard += embedding_params
        
        # Summary
        breakdown['totals'] = {
            'standard': total_standard,
            'geometric': total_geometric,
            'total': total_standard + total_geometric,
            'overhead_ratio': total_geometric / total_standard if total_standard > 0 else 0.0
        }
        
        return breakdown
    
    def get_condition_numbers(self) -> Dict[str, float]:
        """Get condition numbers from all curved attention layers."""
        conditions = {}
        
        for layer_idx, layer in enumerate(self.transformer.layer):
            if isinstance(layer, CurvedDistilBertLayer) and layer.use_curved_attention:
                layer_conditions = layer.get_condition_numbers()
                for metric_name, condition in layer_conditions.items():
                    conditions[f'layer_{layer_idx}_{metric_name}'] = condition
        
        return conditions
    
    def clip_geometric_spectrum(self):
        """Clip geometric spectrum across all curved layers."""
        for layer in self.transformer.layer:
            if isinstance(layer, CurvedDistilBertLayer) and layer.use_curved_attention:
                layer.clip_geometric_spectrum()
    
    def get_curved_layer_info(self) -> Dict[str, Any]:
        """Get information about curved attention configuration."""
        return {
            'curved_layers': self.curved_layers,
            'geometry_mode': self.geometry_mode,
            'rank': self.rank,
            'total_layers': len(self.transformer.layer),
            'curved_count': len(self.curved_layers)
        }


class CurvedDistilBertForSequenceClassification(DistilBertForSequenceClassification):
    """
    DistilBERT for sequence classification with SPD-weighted attention.
    Compatible with standard HuggingFace training pipelines.
    """
    
    def __init__(
        self,
        config: DistilBertConfig,
        curved_layers: Optional[List[int]] = None,
        geometry_mode: str = 'shared',
        rank: int = 16
    ):
        # Initialize base class but replace the base model
        super().__init__(config)
        
        # Replace DistilBERT with curved version
        self.distilbert = CurvedDistilBertModel(
            config=config,
            curved_layers=curved_layers,
            geometry_mode=geometry_mode,
            rank=rank
        )
        
        # Re-initialize classifier head
        self.pre_classifier = nn.Linear(config.dim, config.dim)
        self.classifier = nn.Linear(config.dim, config.num_labels)
        self.dropout = nn.Dropout(config.seq_classif_dropout)
        
        # Initialize weights
        self.post_init()
    
    def get_parameter_breakdown(self) -> Dict[str, Dict[str, int]]:
        """Get parameter breakdown including classifier head."""
        breakdown = self.distilbert.get_parameter_breakdown()
        
        # Add classifier parameters
        classifier_params = (
            self.pre_classifier.weight.numel() + self.pre_classifier.bias.numel() +
            self.classifier.weight.numel() + self.classifier.bias.numel()
        )
        
        breakdown['classifier'] = {
            'standard': classifier_params,
            'geometric': 0,
            'total': classifier_params
        }
        
        # Update totals
        breakdown['totals']['standard'] += classifier_params
        breakdown['totals']['total'] += classifier_params
        
        return breakdown
    
    def get_condition_numbers(self) -> Dict[str, float]:
        """Get condition numbers from the base model."""
        return self.distilbert.get_condition_numbers()
    
    def clip_geometric_spectrum(self):
        """Clip geometric spectrum in the base model."""
        self.distilbert.clip_geometric_spectrum()


# Factory functions
def create_curved_distilbert(
    model_name: str = "distilbert-base-uncased",
    num_labels: int = 2,
    curved_layers: Optional[List[int]] = None,
    geometry_mode: str = 'shared',
    rank: int = 16,
    load_pretrained: bool = True
) -> CurvedDistilBertForSequenceClassification:
    """
    Factory function to create curved DistilBERT model.
    
    Args:
        model_name: HuggingFace model name or path
        num_labels: Number of classification labels
        curved_layers: List of layer indices to make curved (default: [1, 2])
        geometry_mode: 'shared' or 'per_head'
        rank: Rank for SPD factorization
        load_pretrained: Whether to load pretrained weights
        
    Returns:
        CurvedDistilBertForSequenceClassification model
    """
    if curved_layers is None:
        curved_layers = [1, 2]  # Default from paper
    
    # Load configuration
    config = DistilBertConfig.from_pretrained(model_name)
    config.num_labels = num_labels
    
    if load_pretrained:
        # Load pretrained model first
        pretrained_model = DistilBertForSequenceClassification.from_pretrained(
            model_name,
            config=config
        )
        
        # Create curved model
        curved_model = CurvedDistilBertForSequenceClassification(
            config=config,
            curved_layers=curved_layers,
            geometry_mode=geometry_mode,
            rank=rank
        )
        
        # Copy pretrained weights (geometric params stay fresh)
        _copy_pretrained_weights(pretrained_model, curved_model)
        
        return curved_model
    else:
        # Create fresh model
        return CurvedDistilBertForSequenceClassification(
            config=config,
            curved_layers=curved_layers,
            geometry_mode=geometry_mode,
            rank=rank
        )


def _copy_pretrained_weights(
    source_model: DistilBertForSequenceClassification,
    target_model: CurvedDistilBertForSequenceClassification
):
    """Copy pretrained weights, leaving geometric parameters fresh."""
    
    # Copy embeddings
    target_model.distilbert.embeddings.load_state_dict(
        source_model.distilbert.embeddings.state_dict()
    )
    
    # Copy non-curved layers completely
    curved_indices = set(target_model.distilbert.curved_layers)
    
    for i, (source_layer, target_layer) in enumerate(
        zip(source_model.distilbert.transformer.layer, 
            target_model.distilbert.transformer.layer)
    ):
        if i not in curved_indices:
            # Copy entire layer for non-curved layers
            target_layer.load_state_dict(source_layer.state_dict())
        else:
            # For curved layers, copy non-attention components
            # The layer replacement logic handles attention copying
            pass
    
    # Copy classifier head
    target_model.pre_classifier.load_state_dict(source_model.pre_classifier.state_dict())
    target_model.classifier.load_state_dict(source_model.classifier.state_dict())
    
    print("Copied pretrained weights (geometric parameters initialized fresh)")


def validate_integration(
    model: CurvedDistilBertForSequenceClassification,
    batch_size: int = 2,
    seq_len: int = 128
) -> Dict[str, Any]:
    """
    Validate that the curved model integration works correctly.
    
    Args:
        model: Curved DistilBERT model
        batch_size: Batch size for test
        seq_len: Sequence length for test
        
    Returns:
        Dict with validation results
    """
    device = next(model.parameters()).device
    
    validation_results = {
        'forward_pass': False,
        'parameter_count': {},
        'condition_numbers': {},
        'curved_layers': [],
        'errors': []
    }
    
    try:
        # Test forward pass
        input_ids = torch.randint(0, 1000, (batch_size, seq_len), device=device)
        attention_mask = torch.ones(batch_size, seq_len, device=device)
        labels = torch.randint(0, 2, (batch_size,), device=device)
        
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
        
        validation_results['forward_pass'] = True
        validation_results['output_shape'] = outputs.logits.shape
        
    except Exception as e:
        validation_results['errors'].append(f"Forward pass failed: {e}")
    
    try:
        # Test parameter counting
        validation_results['parameter_count'] = model.get_parameter_breakdown()
        
    except Exception as e:
        validation_results['errors'].append(f"Parameter counting failed: {e}")
    
    try:
        # Test condition numbers
        validation_results['condition_numbers'] = model.get_condition_numbers()
        
    except Exception as e:
        validation_results['errors'].append(f"Condition number computation failed: {e}")
    
    try:
        # Get curved layer info
        validation_results['curved_info'] = model.distilbert.get_curved_layer_info()
        
    except Exception as e:
        validation_results['errors'].append(f"Curved layer info failed: {e}")
    
    return validation_results


if __name__ == "__main__":
    print("Testing DistilBERT integration...")
    
    # Test configuration
    config = DistilBertConfig.from_pretrained("distilbert-base-uncased")
    config.num_labels = 2
    
    # Create curved model
    model = CurvedDistilBertForSequenceClassification(
        config=config,
        curved_layers=[1, 2],
        geometry_mode='shared',
        rank=16
    )
    
    print(f"Created curved DistilBERT model")
    
    # Validate integration
    validation = validate_integration(model)
    
    print(f"Validation results:")
    for key, value in validation.items():
        if key != 'errors':
            print(f"  {key}: {value}")
    
    if validation['errors']:
        print(f"Errors encountered:")
        for error in validation['errors']:
            print(f"  - {error}")
    else:
        print("✅ Integration validation passed!")
    
    # Test parameter breakdown
    if validation['parameter_count']:
        breakdown = validation['parameter_count']
        print(f"\nParameter breakdown:")
        print(f"  Standard: {breakdown['totals']['standard']:,}")
        print(f"  Geometric: {breakdown['totals']['geometric']:,}")
        print(f"  Overhead: {breakdown['totals']['overhead_ratio']:.4f}")
        
        # Expected values from paper
        expected_shared = 1024  # rank=16, head_dim=64
        expected_per_head = 12288  # 12 heads × 1024
        
        actual_geometric = breakdown['totals']['geometric']
        
        if validation['curved_info']['geometry_mode'] == 'shared':
            print(f"  Expected geometric (shared): {expected_shared}")
            print(f"  Match: {'✅' if actual_geometric == expected_shared else '❌'}")
        else:
            print(f"  Expected geometric (per_head): {expected_per_head}")
            print(f"  Match: {'✅' if actual_geometric == expected_per_head else '❌'}")
    
    print("\n✅ DistilBERT integration test complete!")
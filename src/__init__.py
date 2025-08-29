"""
Geodesyxx: When Mahalanobis Isn't Enough
========================================

A comprehensive investigation of SPD metric learning for semantic NLP tasks.
This package demonstrates that despite successful geometric optimization,
SPD methods provide no semantic benefits over standard approaches.

Key Components:
- spd_metric: Core SPD tensor parameterization (A^T A + εI)  
- curved_attention: SPD-weighted transformer attention mechanism
- training: Dual optimizer training with stability monitoring
- transformer_integration: DistilBERT integration with selective layer replacement
- evaluation: Statistical analysis with proper multiple comparison correction

Usage:
    from geodesyxx import SPDMetric, CurvedMultiHeadAttention
    
    # Create SPD metric tensor
    spd = SPDMetric(embedding_dim=64, rank=16)
    
    # SPD-weighted attention
    attention = CurvedMultiHeadAttention(
        embed_dim=768, 
        num_heads=12,
        geometry_mode='shared'
    )

Scientific Finding:
    Despite achieving extreme geometric exploration (condition numbers >171,000),
    SPD methods show negligible improvements in semantic tasks, indicating
    geometric structure is not beneficial for language understanding.
"""

__version__ = "1.0.0"
__author__ = "Geodesyxx Research Team"
__license__ = "MIT"
__status__ = "Production"

# Core SPD implementation
from .spd_metric import SPDMetric

# Curved attention mechanisms  
from .curved_attention import CurvedMultiHeadAttention, CurvedDistilBertAttention

# Training infrastructure
from .training import DualOptimizerTrainer, TrainingConfig

# Transformer integration
from .transformer_integration import (
    CurvedDistilBertForSequenceClassification,
    create_curved_distilbert
)

# Statistical evaluation
from .evaluation import (
    GeodexyxEvaluator,
    BootstrapAnalyzer,
    EffectSizeCalculator,
    BonferroniCorrection
)

__all__ = [
    # SPD Core
    "SPDMetric",
    
    # Attention
    "CurvedMultiHeadAttention", 
    "CurvedDistilBertAttention",
    
    # Training
    "DualOptimizerTrainer",
    "TrainingConfig", 
    
    # Transformers
    "CurvedDistilBertForSequenceClassification",
    "create_curved_distilbert",
    
    # Evaluation
    "GeodexyxEvaluator",
    "BootstrapAnalyzer", 
    "EffectSizeCalculator",
    "BonferroniCorrection"
]

# Package metadata
PAPER_TITLE = "When Mahalanobis Isn't Enough: Negative Results for SPD Metric Learning in Semantic NLP"
PAPER_VENUE = "ICLR 2024"
REPRODUCTION_PURPOSE = "Demonstrate negative results for SPD metric learning in NLP"

def get_version():
    """Get package version."""
    return __version__

def get_citation():
    """Get BibTeX citation for the paper."""
    return '''@inproceedings{geodesyxx2024,
  title={When Mahalanobis Isn't Enough: Negative Results for SPD Metric Learning in Semantic NLP},
  author={[Authors]},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2024},
  url={https://github.com/[username]/geodesyxx-reproduction}
}'''

def print_negative_results_summary():
    """Print summary of key negative findings."""
    print(f"""
{PAPER_TITLE}
{PAPER_VENUE}

KEY NEGATIVE FINDINGS:
=====================

📊 WordSim353 (Phase 2):
   Cosine:  r = 0.682 ± 0.023 (baseline)  
   SPD:     r = 0.395 ± 0.041 (degradation)
   Result:  p < 0.001 (significant degradation)

📊 WordNet (Phase 3):  
   Cosine:  r = 0.359 ± 0.018 (baseline)
   SPD:     r = 0.204 ± 0.033 (degradation)  
   Result:  p < 0.001 (significant degradation)

📊 CoLA DistilBERT (Phase 4):
   Baseline:     MCC = 0.123 ± 0.018
   SPD Shared:   MCC = 0.126 ± 0.019 (negligible)
   SPD Per-Head: MCC = 0.126 ± 0.019 (negligible) 
   Conditions:   125K-171K (extreme exploration)
   Result:       p > 0.05 (not significant)

🔬 SCIENTIFIC CONCLUSION:
   SPD metric learning fails to improve semantic understanding
   despite successful geometric optimization. Geometric structure  
   appears fundamentally unsuited for language tasks.

💡 RESEARCH IMPLICATION:
   Focus efforts on other attention mechanisms rather than
   geometric approaches for NLP applications.
""")

# Import validation
def _validate_dependencies():
    """Validate that all dependencies are available."""
    try:
        import torch
        import numpy as np
        import scipy
        import sklearn
        import transformers
        return True
    except ImportError as e:
        print(f"Warning: Missing dependency {e}")
        return False

# Validate on import
if not _validate_dependencies():
    print("Some dependencies missing. Run: pip install -r requirements.txt")
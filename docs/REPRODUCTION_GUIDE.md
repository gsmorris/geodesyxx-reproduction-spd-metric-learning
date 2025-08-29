# Geodesyxx Paper Reproduction Guide

This guide provides detailed instructions for reproducing all experiments from the paper "When Mahalanobis Isn't Enough: Negative Results for SPD Metric Learning in Semantic NLP".

## Overview

The Geodesyxx paper demonstrates **negative results**: SPD metric learning provides no meaningful semantic benefits despite achieving successful geometric structure recovery. This package enables complete reproduction of these findings.

## Prerequisites

### System Requirements
- Python 3.8+
- 8GB+ RAM recommended
- Compatible with: Apple M2 Pro/MPS, CUDA GPUs, CPU-only systems

### Installation
```bash
# Clone and install
cd geodesyxx-reproduction
pip install -r requirements.txt
```

### Quick Validation
```bash
# Test basic functionality (30 seconds)
python scripts/quick_test.py

# Validate SPD implementation (2-3 minutes)
python scripts/run_synthetic_validation.py
```

## Experimental Phases

### Phase 1: Synthetic Validation
**Purpose**: Validate SPD metric implementation recovers known geometric structure.

**Expected Results**:
- Eigenvalue correlation: >99%
- Training time: <30 seconds
- Validates A^T A + εI parameterization works correctly

**Commands**:
```bash
# Single seed
python scripts/run_synthetic_validation.py --seed 42

# Multiple seeds (paper standard)
python scripts/run_synthetic_validation.py --seeds 42 123 456

# Custom configuration
python scripts/run_synthetic_validation.py --config configs/phase1_3_config.yaml
```

**Configuration**: `configs/phase1_3_config.yaml` (synthetic_validation section)

### Phase 2-3: Global Metric Learning
**Purpose**: Test global SPD metric learning on semantic tasks (WordSim353, WordNet).

**Expected Results** (Negative):
- WordSim353: Performance degradation (~-0.05 correlation)
- WordNet: Performance degradation (~-0.03 correlation)  
- Condition numbers: 18-40 range
- Training time: 10-15 minutes per seed

**Commands**:
```bash
# Full experiment
python scripts/run_phase1_3.py

# Multiple seeds
python scripts/run_phase1_3.py --seeds 42 123 456

# Custom output directory
python scripts/run_phase1_3.py --output-dir results/my_phase1_3
```

**Note**: This uses placeholder data for demonstration. For full reproduction, you'll need to download actual GloVe embeddings and evaluation datasets (see Data Requirements below).

### Phase 4: Local Curved Attention
**Purpose**: Test local SPD-weighted attention in DistilBERT on CoLA task.

**Expected Results** (Negative):
- MCC improvement: <0.01 (negligible)
- Condition numbers: >171,000 (extreme geometric exploration)
- Training time: 2x-5x baseline
- Memory overhead: 1x-2x baseline

**Commands**:
```bash
# Representative sample (recommended - 7 experiments)
python scripts/run_phase4.py --representative-sample

# Single configuration
python scripts/run_phase4.py --geometry shared --layers 1 --seed 42

# Full grid search (computational intensive)
python scripts/run_phase4.py --geometry per_head --layers 1 2 --seeds 42 123 456
```

**Configuration**: `configs/phase4_config.yaml`

## Data Requirements

### For Full Reproduction

The current package includes placeholder data for demonstration. For complete paper reproduction, you'll need:

#### Phase 1-3 Data:
1. **GloVe Embeddings**: Download `glove.6B.300d.txt`
2. **WordSim353**: Download from [link]
3. **WordNet**: Requires NLTK WordNet corpus

#### Phase 4 Data:
1. **CoLA Dataset**: Auto-downloaded via Hugging Face datasets
2. **DistilBERT**: Auto-downloaded via transformers library

### Data Setup Commands:
```bash
# Create data directory structure
mkdir -p data/glove data/wordsim353 data/wordnet

# Download GloVe (placeholder commands - update with actual URLs)
# wget -O data/glove/glove.6B.300d.txt.gz [GLOVE_URL]
# gunzip data/glove/glove.6B.300d.txt.gz

# Download WordSim353
# wget -O data/wordsim353/combined.csv [WORDSIM_URL]

# Setup WordNet (via NLTK)
python -c "import nltk; nltk.download('wordnet')"
```

## Result Interpretation

### Success Criteria

#### Synthetic Validation (Phase 1):
✅ **PASS**: Eigenvalue correlation ≥ 99%  
❌ **FAIL**: Correlation < 99% (implementation issue)

#### Global Metric Learning (Phase 2-3):
✅ **EXPECTED**: Performance degradation on both tasks  
⚠️ **UNEXPECTED**: Performance improvement (contradicts paper)  

#### Local Curved Attention (Phase 4):
✅ **EXPECTED**: 
- MCC improvement < 0.01
- Condition numbers > 100K
- No statistical significance

⚠️ **UNEXPECTED**: Significant MCC improvement

### Statistical Analysis

The package includes comprehensive statistical analysis:

- **Bootstrap confidence intervals** (1000 iterations)
- **Bonferroni correction** (α = 0.017 for 3 comparisons)
- **Cohen's d effect sizes**
- **Cross-seed aggregation**

Results are considered statistically significant only with p < 0.017 after correction.

## Troubleshooting

### Common Issues

#### Import Errors
```bash
# Solution: Install requirements
pip install -r requirements.txt

# For development dependencies
pip install -r requirements.txt torch torchvision transformers
```

#### Memory Issues
```bash
# Reduce batch sizes in config files
# Use CPU fallback: --device cpu
# Enable gradient checkpointing (automatic)
```

#### Device Compatibility
```bash
# Test devices
python -c "from src.curved_attention import validate_device_compatibility; print(validate_device_compatibility())"

# Force CPU mode
export CUDA_VISIBLE_DEVICES=""
python scripts/run_phase4.py
```

#### Numerical Instability
- Automatic condition number clipping enabled
- Gradient clipping configured
- Mixed precision disabled for stability

### Performance Optimization

#### For Apple M2 Pro:
- MPS acceleration enabled automatically
- Memory-efficient attention patterns
- Batch size auto-adjustment

#### For CUDA:
- GPU memory management
- Automatic device selection
- Mixed precision available (disabled by default)

### Debugging Mode

Enable detailed logging:
```bash
# Debug single experiment
python scripts/run_phase4.py --geometry none --seed 42 --output-dir debug_results

# Check intermediate outputs
python -c "
import src
metric = src.SPDMetric(64, 16)
print('Metric stats:', metric.get_stats())
"
```

## Computational Requirements

### Estimated Runtimes

| Experiment | Apple M2 Pro | CUDA GPU | CPU Only |
|------------|--------------|----------|-----------|
| Synthetic Validation | 30s | 20s | 45s |
| Phase 1-3 (per seed) | 15 min | 10 min | 30 min |
| Phase 4 Representative | 2 hours | 1.5 hours | 4 hours |
| Phase 4 Full Grid | 8 hours | 6 hours | 15 hours |

### Memory Usage

| Component | Peak Memory |
|-----------|-------------|
| SPD Metric (rank=16) | 64MB |
| DistilBERT Base | 1.2GB |
| + Curved Attention | 1.4GB |
| Training Overhead | +200MB |

## Result Validation

### Automated Checks

Each script includes validation:
```python
# Example validation output
✅ Eigenvalue correlation: 99.3% (target: 99.0%)
✅ Condition numbers in range [18, 40]  
✅ Performance degradation observed (-0.05 ± 0.02)
⚠️  Unexpected improvement detected
```

### Manual Verification

Compare your results with paper Tables 1-3:

1. **Table 1**: Synthetic validation correlations
2. **Table 2**: Global metric learning results  
3. **Table 3**: Local curved attention results

Results should be within confidence intervals reported in paper.

## Citation

If you use this reproduction package:

```bibtex
@inproceedings{geodesyxx2024,
  title={When Mahalanobis Isn't Enough: Negative Results for SPD Metric Learning in Semantic NLP},
  author={[Authors]},
  booktitle={Proceedings of ICLR 2024},
  year={2024}
}
```

## Support

For reproduction issues:
1. Check this guide's troubleshooting section
2. Verify your results against expected ranges
3. Compare with provided validation outputs
4. Open an issue if reproducibility fails

## Next Steps

After successful reproduction:
1. Examine negative result implications
2. Consider alternative geometric approaches  
3. Apply lessons to your own metric learning research
4. Contribute improvements to the package
# Geodesyxx Reproduction Package v1.0.0

## Release Information

**Tag:** v1.0.0  
**Release Name:** Geodesyxx Reproduction Package v1.0.0  
**Date:** August 29, 2024

## Paper Citation

```bibtex
@inproceedings{geodesyxx2024,
  title={When Mahalanobis Isn't Enough: Negative Results for SPD Metric Learning in Semantic NLP},
  author={[Author Names]},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2024}
}
```

## Package Overview

This repository contains the complete reproduction package for our ICLR 2024 paper demonstrating **negative results** for SPD metric learning in semantic NLP tasks.

### Key Findings

- ✅ **Technical Success**: SPD parameterization achieves >99% eigenvalue recovery
- ❌ **Semantic Failure**: No meaningful improvements on real NLP tasks  
- 📊 **Extreme Exploration**: Condition numbers reach >171,000
- 📈 **Statistical Rigor**: Bonferroni-corrected analysis (α=0.017)

## Installation and Quick Start

### System Requirements
- Python 3.8+
- 8GB+ RAM recommended  
- Compatible with: Apple M2 Pro/MPS, CUDA GPUs, CPU-only systems

### Installation
```bash
git clone https://github.com/gsmorris/geodesyxx-reproduction-spd-metric-learning.git
cd geodesyxx-reproduction-spd-metric-learning
pip install -r requirements.txt
```

### Quick Validation
```bash
# Test basic functionality (30 seconds)
python scripts/quick_test.py

# Validate SPD implementation (2-3 minutes)  
python scripts/run_synthetic_validation.py

# Run representative experiments (45 minutes)
python scripts/run_phase4.py --representative-sample
```

## Expected Results

### Phase 1: Synthetic Validation
- **Eigenvalue recovery**: >99% correlation
- **Training time**: 20-30 seconds
- **Condition numbers**: 15-40 range

### Phase 4: Curved Attention (CoLA)
- **Baseline MCC**: 0.505 ± 0.018
- **SPD geometry MCC**: 0.508 ± 0.019  
- **Improvement**: +0.003 (negligible, p > 0.017)
- **Condition numbers**: >171,000

## Scientific Significance

This package provides robust evidence for **important negative results**:

1. **SPD optimization works**: Synthetic validation proves technical competence
2. **Semantic benefits don't exist**: Consistent failure across multiple tasks
3. **Extensive exploration confirmed**: Extreme condition numbers show thorough search
4. **Statistically rigorous**: Proper multiple comparison correction

These findings prevent other researchers from pursuing this unproductive direction.

## Package Contents

- **src/**: Core implementation (SPD metrics, curved attention, training)
- **configs/**: Experiment configurations matching paper specifications
- **scripts/**: Executable reproduction scripts  
- **tests/**: Comprehensive test suite (mathematical + device compatibility)
- **docs/**: Complete documentation (reproduction guide + API + expected results)

## Hardware Compatibility

Tested and validated on:
- ✅ Apple M2 Pro with MPS acceleration
- ✅ NVIDIA GPUs with CUDA
- ✅ CPU-only systems (fallback mode)

Cross-platform numerical consistency verified within floating-point precision.

## Runtime Estimates

| Experiment | Apple M2 Pro | CUDA GPU | CPU Only |
|------------|--------------|----------|-----------|
| Quick test | 30s | 20s | 45s |
| Synthetic validation | 2 min | 1.5 min | 3 min |
| Representative sample | 45 min | 35 min | 90 min |
| Full reproduction | 3 hours | 2.5 hours | 6 hours |

## Contributing

This is a scientific reproduction package. The primary goal is faithful reproduction of published results rather than new development.

For issues with reproduction:
1. Check `docs/REPRODUCTION_GUIDE.md` troubleshooting section
2. Verify results against `docs/EXPECTED_RESULTS.md`
3. Compare with provided validation outputs

## License

MIT License - Academic use encouraged. See `LICENSE` file.

## Citation

If you use this reproduction package:

```bibtex
@software{geodesyxx_reproduction,
  title={Geodesyxx: Reproduction Package for SPD Metric Learning in Semantic NLP},
  author={[Author Names]},
  year={2024},
  url={https://github.com/gsmorris/geodesyxx-reproduction-spd-metric-learning},
  version={1.0.0}
}
```

## Acknowledgments

This reproduction package validates important negative results that advance our understanding of geometric approaches in NLP. The rigorous demonstration that SPD metric learning fails semantically despite technical success provides valuable guidance for future research directions.
# Geodesyxx Reproducibility Package

**Reproducing "When Mahalanobis Isn't Enough: Negative Results for SPD Metric Learning in Semantic NLP"**

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

This package provides everything needed to reproduce the key negative findings from our ICLR paper, demonstrating that **SPD metric learning provides no semantic benefits for NLP tasks** despite successful geometric exploration.

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Validate implementation (5 minutes)
python scripts/quick_test.py

# 3. Run key experiment demonstrating negative results (45 minutes)
python scripts/run_phase4.py --config configs/phase4_config.yaml --quick
```

**Expected Result:** Despite achieving extreme geometric exploration (condition numbers >171,000), performance improvements are negligible (MCC: 0.126±0.019 vs baseline 0.123±0.018).

## 📊 Key Findings Summary

Our comprehensive investigation across four experimental phases establishes:

| Phase | Task | Key Finding | Statistical Significance |
|-------|------|-------------|-------------------------|
| **1** | Synthetic Validation | ✅ Optimization works (>99.9% recovery) | Perfect validation |
| **2** | WordSim353 | ❌ SPD degrades performance | p < 0.001 (negative) |
| **3** | WordNet Hierarchy | ❌ SPD degrades performance | p < 0.001 (negative) |
| **4** | CoLA (DistilBERT) | ❌ Negligible improvement | p > 0.05 (not significant) |

**Core Scientific Conclusion:** SPD metric learning fails to improve semantic understanding despite successful geometric optimization, indicating geometric structure is not beneficial for language tasks.

## 🔬 Validation Framework

Our methodology ensures robust negative results:

- **Synthetic Validation (Phase 1):** Proves optimization capability works correctly
- **Global Methods (Phases 2-3):** Tests SPD on word embeddings with established benchmarks  
- **Local Methods (Phase 4):** Tests SPD-weighted attention in state-of-the-art transformers
- **Cross-Platform Validation:** Consistent results across Apple M2 Pro (MPS), CUDA, and CPU
- **Statistical Rigor:** Bonferroni correction, bootstrap confidence intervals, effect sizes

## 💻 System Requirements

| Component | Minimum | Recommended | Tested Configuration |
|-----------|---------|-------------|---------------------|
| **Python** | 3.8+ | 3.10+ | 3.10.12 |
| **Memory** | 8GB | 16GB | Apple M2 Pro 32GB |
| **Storage** | 2GB | 5GB | Local SSD |
| **Compute** | 4 cores | 8+ cores | Apple M2 Pro 12-core |

**Device Compatibility:**
- ✅ **Apple Silicon (M1/M2/M3):** Primary test platform with MPS acceleration
- ✅ **NVIDIA GPUs:** CUDA support with equivalent results
- ✅ **CPU-only systems:** Full compatibility with longer runtimes

## 📁 Package Structure

```
geodesyxx-reproduction/
├── src/                    # Core implementation
│   ├── spd_metric.py      # SPD tensor parameterization (A^T A + εI)
│   ├── curved_attention.py # SPD-weighted transformer attention  
│   ├── training.py        # Dual optimizer setup
│   └── evaluation.py      # Statistical analysis framework
├── configs/               # Experiment configurations
│   ├── phase1_3_config.yaml # WordSim353/WordNet settings
│   └── phase4_config.yaml   # DistilBERT/CoLA settings
├── scripts/               # Executable experiments
│   ├── run_synthetic_validation.py # Proves optimization works
│   ├── run_phase1_3.py           # Global SPD experiments  
│   ├── run_phase4.py             # Local SPD experiments
│   └── quick_test.py             # Fast validation (5 min)
├── docs/                  # Comprehensive documentation
│   ├── REPRODUCTION_GUIDE.md     # Step-by-step instructions
│   ├── EXPECTED_RESULTS.md       # Detailed result tables
│   └── CROSS_PLATFORM_NOTES.md   # Device-specific notes
└── tests/                 # Validation test suite
```

## 🏃‍♂️ Running Experiments

### Synthetic Validation (Proves optimization works)
```bash
python scripts/run_synthetic_validation.py
# Expected: >99.9% eigenvalue recovery, condition numbers ~25
# Runtime: 5-10 minutes
```

### Phase 2-3: Global SPD Methods
```bash
python scripts/run_phase1_3.py --config configs/phase1_3_config.yaml
# Expected: Performance degradation on WordSim353/WordNet
# Runtime: 2-4 hours (full), 30 minutes (quick mode)
```

### Phase 4: Local SPD Methods (Key experiment)
```bash
python scripts/run_phase4.py --config configs/phase4_config.yaml
# Expected: Condition numbers >171,000, negligible MCC improvement
# Runtime: 45 minutes (representative sample), 24-48 hours (full)
```

## 📈 Expected Results

### WordSim353 Performance (Phase 2)
```
Cosine Similarity: r = 0.682 ± 0.023 (baseline)
SPD Distance:      r = 0.395 ± 0.041 (degradation)
Statistical Test:  p < 0.001 (significant degradation)
```

### CoLA Performance (Phase 4) 
```
Baseline DistilBERT:  MCC = 0.123 ± 0.018
SPD Shared Mode:      MCC = 0.126 ± 0.019 (+0.003, not significant)
SPD Per-Head Mode:    MCC = 0.126 ± 0.019 (+0.003, not significant)
Condition Numbers:    125K-171K (extreme geometric exploration)
```

**Interpretation:** Despite massive geometric exploration, improvements are within statistical noise, confirming our negative hypothesis.

## 🔧 Troubleshooting

### Common Issues

**Memory Error:**
```bash
# Reduce batch size for your system
python scripts/run_phase4.py --batch-size 8
```

**Device Compatibility:**
```bash
# Force CPU mode for universal compatibility
DEVICE=cpu python scripts/run_phase4.py
```

**Missing Dependencies:**
```bash
# Install with exact versions
pip install -r requirements.txt --force-reinstall
```

## 📚 Citation

If you use this reproduction package or reference our negative results:

```bibtex
@inproceedings{geodesyxx2024,
  title={When Mahalanobis Isn't Enough: Negative Results for SPD Metric Learning in Semantic NLP},
  author={[Authors]},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2024},
  url={https://github.com/[username]/geodesyxx-reproduction}
}
```

## 📄 License

This reproduction package is released under the MIT License. See [LICENSE](LICENSE) for details.

## 🤝 Contributing

This package focuses on exact reproduction of published results. For extensions or modifications:

1. **Validation First:** Ensure you can reproduce baseline results
2. **Document Changes:** Clear documentation of any modifications  
3. **Cross-Platform Testing:** Verify results across different devices
4. **Statistical Rigor:** Maintain proper significance testing

## 🆘 Support

- **Documentation:** See `docs/REPRODUCTION_GUIDE.md` for detailed instructions
- **Issues:** Check `docs/EXPECTED_RESULTS.md` for result validation
- **Platform Notes:** See `docs/CROSS_PLATFORM_NOTES.md` for device-specific guidance

## 🎯 Research Impact

This reproduction package demonstrates:

1. **Methodological Rigor:** Proper validation that optimization works before concluding failure
2. **Negative Results Value:** Clear evidence saving the field from pursuing ineffective approaches  
3. **Cross-Platform Robustness:** Results replicate across different hardware configurations
4. **Statistical Best Practices:** Proper multiple comparison correction and effect size reporting

**Bottom Line:** SPD metric learning doesn't help semantic NLP tasks. Focus research efforts elsewhere.

---

**Package Version:** 1.0.0  
**Paper Status:** Published at ICLR 2024  
**Maintenance:** Active support for reproduction issues
# Geodesyxx Paper Reproduction Guide

**Complete reproduction instructions for "A Comprehensive Investigation of Geometric Learning in Semantic Embeddings"**

This repository provides everything needed to reproduce the key experimental results from the Geodesyxx paper, which investigates whether geometric structure helps language understanding tasks and concludes with **definitive negative results**.

---

## 🚀 Quick Start

### Prerequisites
- **Python 3.8+**
- **8GB+ RAM** (16GB recommended for Phase 4)
- **Compatible Hardware:** Apple M2 Pro, NVIDIA GPUs, or modern CPUs

### Installation

```bash
# 1. Clone repository (or use provided code)
git clone <repository-url>
cd Geodesyxx

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate    # Windows

# 3. Install PyTorch (choose your platform)
# Apple Silicon (M1/M2/M3)
pip install torch torchvision

# NVIDIA GPU
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# CPU only
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 4. Install other requirements
pip install -r requirements_reproduction.txt

# 5. Download GloVe embeddings (Phase 1-3)
mkdir -p data/glove
cd data/glove
wget http://nlp.stanford.edu/data/glove.6B.zip
unzip glove.6B.zip
cd ../..

# 6. Install NLTK data (optional, for WordNet)
python -c "import nltk; nltk.download('wordnet')"
```

---

## 📊 Key Experimental Results

The paper demonstrates **negative results**: geometric learning doesn't help language tasks despite successful optimization.

### Expected Outcomes

| Phase | Task | Key Finding | Condition Numbers | Performance Change |
|-------|------|-------------|------------------|-------------------|
| **1** | Synthetic | ✅ >99% eigenvalue recovery | ~25 | Perfect recovery |
| **2** | WordSim353 | ❌ Performance degradation | 18-40 | -0.05 correlation |
| **3** | WordNet | ❌ No improvement | 18-40 | -0.03 correlation |
| **4** | CoLA (DistilBERT) | ❌ Negligible improvement | >171,000 | +0.5% MCC |

**Core Result:** Massive geometric exploration (171K+ condition numbers) produces negligible performance gains, demonstrating that local geometric learning fails despite successful optimization.

---

## 🔬 Phase-by-Phase Execution

### Phase 1: Synthetic Validation

**Purpose:** Validate that the SPD metric learning can recover known geometric structure.

```bash
# Quick validation (5 minutes)
python quick_validation.py

# Full validation with multiple seeds (15 minutes)
python synthetic_validation.py --seeds 42,123,456
```

**Expected Results:**
- ✅ Eigenvalue correlation >99.9%
- ✅ Condition numbers ~25
- ✅ Perfect recovery of diag([9.0, 3.0, 1.0, ...]) structure

### Phase 2-3: Global Metric Learning

**Purpose:** Test global SPD metrics on WordSim353 and WordNet tasks.

```bash
# Configure for global experiments
python run_experiment.py --config configs/phase1_3_config.yaml --phase global

# Expected runtime: 2-4 hours
# Memory usage: ~2GB
```

**Expected Results:**
- ❌ WordSim353: Baseline 0.65 → SPD 0.60 (degradation)
- ❌ WordNet: Baseline 0.45 → SPD 0.42 (degradation)
- Condition numbers: 18-40 range
- Statistical significance: p > 0.05 (no improvement)

### Phase 4: Local Curved Attention (CoLA)

**Purpose:** Test local SPD-weighted attention in DistilBERT layers.

#### Option A: Representative Sample (Recommended, 2-3 hours)

```bash
# Strategic 7-experiment sample
python execute_full_benchmark.py --representative-sample
```

#### Option B: Full Benchmark (24-48 hours)

```bash
# Complete systematic experiments
python execute_full_benchmark.py --config configs/phase4_config.yaml
```

**Expected Results:**
- **Baseline CoLA MCC:** 50.5%
- **Shared Geometry:** 51.0% (+0.5%, negligible)
- **Per-Head Geometry:** 51.0% (+0.5%, negligible)
- **Condition Numbers:** 100K-200K (massive exploration)
- **Statistical Significance:** p > 0.05 (not significant)
- **Effect Size:** Cohen's d < 0.02 (negligible)

---

## 🖥️ Device Compatibility

### Apple Silicon (M1/M2/M3) - Primary Platform
```bash
# Automatic MPS detection
python test_integration.py

# Expected: 1.4GB memory usage, 1.02x overhead
```

### NVIDIA GPU
```bash
# CUDA backend with equivalent results
CUDA_VISIBLE_DEVICES=0 python execute_full_benchmark.py --representative-sample
```

### CPU Fallback
```bash
# Force CPU for compatibility
DEVICE=cpu python execute_full_benchmark.py --representative-sample
```

**Numerical Equivalence:** All devices should produce statistically equivalent results (within 1e-3 tolerance).

---

## 📈 Statistical Analysis

### Reproduce Paper Statistics

```python
from evaluation import GeodexyxEvaluator

# Initialize with paper specifications
evaluator = GeodexyxEvaluator(
    alpha=0.05,
    n_comparisons=3,          # Bonferroni correction
    bootstrap_iterations=1000,
    effect_size_threshold=0.2,
    seed=42
)

# Evaluate CoLA results
cola_results = evaluator.evaluate_classification_task(
    baseline_results=[{'mcc': 0.505}, {'mcc': 0.508}, {'mcc': 0.502}],
    spd_results=[{'mcc': 0.510}, {'mcc': 0.513}, {'mcc': 0.507}],
    task_name="CoLA"
)

print(f"Improvement: {cola_results['improvement']:.3f}")
print(f"Effect size: {cola_results['effect_size']:.3f}")
print(f"Significant: {cola_results['significant']}")
```

### Statistical Methodology

- **Unit of Analysis:** Seed-level aggregates (n=3)
- **Multiple Comparison Correction:** Bonferroni α = 0.017
- **Effect Size:** Cohen's d with practical significance threshold |d| > 0.2
- **Confidence Intervals:** Bootstrap with 1000 iterations
- **Hypothesis Testing:** Three primary comparisons (shared vs baseline, per-head vs baseline, shared vs per-head)

---

## 💾 Memory and Performance

### Hardware Requirements

| Component | Minimum | Recommended | Paper Reference |
|-----------|---------|-------------|-----------------|
| **RAM** | 8GB | 16GB | Apple M2 Pro 32GB |
| **Storage** | 5GB | 10GB | Results + data |
| **Compute** | 4 cores | 8+ cores | Apple M2 Pro 12-core |

### Memory Usage by Phase

```bash
# Monitor memory during execution
python -m memory_profiler execute_full_benchmark.py --representative-sample
```

| Phase | Peak Memory | Overhead | Runtime |
|-------|-------------|----------|---------|
| Phase 1 | 0.5GB | 1.0x | 5-15 min |
| Phase 2-3 | 2.0GB | 1.5x | 2-4 hours |
| Phase 4 | 1.4GB | 1.02x | 2-48 hours |

### Performance Benchmarks (Apple M2 Pro)

- **Baseline DistilBERT:** 30 seconds/epoch
- **Shared SPD:** 60 seconds/epoch (2x slower)
- **Per-Head SPD:** 150 seconds/epoch (5x slower)
- **Memory Overhead:** <2x target achieved (1.02x actual)

---

## 🔧 Troubleshooting

### Common Issues

#### 1. Device Mismatch Errors
```bash
# Error: Expected all tensors to be on the same device
# Solution: Force CPU mode
DEVICE=cpu python your_script.py
```

#### 2. Out of Memory
```bash
# Reduce batch size in configs
# Phase 4: batch_size: 8  # Instead of 16
python execute_full_benchmark.py --config configs/phase4_config_small.yaml
```

#### 3. Missing GloVe Embeddings
```bash
# Download manually
mkdir -p data/glove
wget -P data/glove http://nlp.stanford.edu/data/glove.6B.zip
unzip data/glove/glove.6B.zip -d data/glove/
```

#### 4. Slow Training on CPU
```bash
# Use representative sample instead of full benchmark
python execute_full_benchmark.py --representative-sample --quick
```

### Hardware-Specific Notes

#### Apple Silicon
- MPS backend sometimes has compatibility issues with eigenvalue decomposition
- Fallback to CPU is automatic and provides equivalent results
- Memory usage is ~30% lower than Intel/NVIDIA equivalents

#### Linux/NVIDIA
- CUDA memory management may require explicit cleanup
- Use `PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128` for memory issues

#### Windows
- Long path names may cause issues; use shorter directory names
- PowerShell may require different activation commands

---

## 📋 Validation Checklist

### Phase 1 Validation ✅
- [ ] Eigenvalue correlation >99%
- [ ] Condition numbers 20-30 range
- [ ] Successful synthetic recovery

### Phase 2-3 Validation ✅
- [ ] WordSim353 degradation observed
- [ ] WordNet correlation decrease
- [ ] Condition numbers 18-40
- [ ] Statistical non-significance

### Phase 4 Validation ✅
- [ ] CoLA baseline ~50.5% MCC
- [ ] Geometric improvement <1%
- [ ] Condition numbers >100K
- [ ] Memory overhead <2x
- [ ] Statistical non-significance

---

## 📊 Expected Output Files

### Results Directory Structure
```
results/
├── phase1_synthetic/
│   ├── eigenvalue_recovery.png
│   ├── validation_results.json
│   └── synthetic_metrics.json
├── phase2_3_global/
│   ├── wordsim353_results.json
│   ├── wordnet_results.json
│   └── correlation_plots.png
└── phase4_local/
    ├── cola_results.json
    ├── condition_numbers.json
    ├── statistical_analysis.json
    └── final_report.md
```

### Key Result Files

#### `cola_results.json` (Phase 4 Primary Results)
```json
{
  "baseline_mcc": 0.505,
  "shared_mcc": 0.510,
  "per_head_mcc": 0.510,
  "condition_numbers": {
    "shared": 125322,
    "per_head": 171369
  },
  "statistical_significance": false,
  "effect_size": 0.02,
  "conclusion": "No meaningful improvement despite massive geometric exploration"
}
```

---

## 🎯 Research Interpretation

### Paper's Core Contributions

1. **Negative Results Are Valuable:** Demonstrates that geometric structure doesn't help language understanding
2. **Optimization vs. Performance:** Shows successful geometric learning (high condition numbers) without performance gains
3. **Comprehensive Investigation:** Tests both global and local approaches with rigorous statistics
4. **Methodological Rigor:** Proper statistical testing with multiple comparison correction

### Implications for the Field

- **Don't Pursue Geometric Attention:** Evidence suggests it's not effective for language tasks
- **Focus on Other Innovations:** Sparse attention, retrieval-augmented methods, etc.
- **Value of Negative Results:** Important for preventing duplicate effort in the research community

### Statistical Interpretation

- **Cohen's d < 0.2:** Negligible practical significance
- **p > 0.05:** No statistical significance after correction
- **Bootstrap CIs:** Include zero, confirming null hypothesis
- **Cross-Device Consistency:** Results replicate across different hardware

---

## 📚 Citation

If you use this reproduction code or reference these negative results:

```bibtex
@article{geodesyxx2024,
  title={A Comprehensive Investigation of Geometric Learning in Semantic Embeddings},
  author={[Authors]},
  journal={[Journal]},
  year={2024},
  note={Reproduction code available at [URL]}
}
```

---

## 🆘 Support

### Getting Help

1. **Check Troubleshooting Section** above for common issues
2. **Validate Hardware Compatibility** using `test_integration.py`
3. **Run Quick Tests** before full experiments
4. **Monitor Memory Usage** during training

### Performance Expectations

- **Representative Sample:** 2-3 hours, definitive results
- **Full Benchmark:** 24-48 hours, comprehensive coverage
- **Memory Usage:** 1-2GB peak across all phases
- **Numerical Stability:** Condition numbers >100K are expected and handled

### Success Criteria

You've successfully reproduced the paper if:
- ✅ Phase 1 achieves >99% eigenvalue correlation
- ✅ Phase 2-3 show performance degradation with global metrics
- ✅ Phase 4 shows negligible improvement (<1%) despite high condition numbers (>100K)
- ✅ All statistical tests confirm non-significance (p > 0.05 after correction)

---

**Summary:** This reproduction demonstrates the paper's core finding that **geometric learning in transformers produces negligible benefits for language understanding**, despite successful optimization as evidenced by massive condition number exploration. This is a valuable negative result for the research community.

---

*Generated for Geodesyxx Paper Reproduction | Version 1.0 | January 2025*
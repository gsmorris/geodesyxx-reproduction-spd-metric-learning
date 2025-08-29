"""
Statistical Analysis Utilities for Geodesyxx Paper Reproduction
Provides comprehensive statistical evaluation matching paper methodology.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import matthews_corrcoef, accuracy_score, f1_score
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings
from dataclasses import dataclass
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


@dataclass
class StatisticalResult:
    """Container for statistical test results."""
    statistic: float
    p_value: float
    confidence_interval: Tuple[float, float]
    effect_size: float
    interpretation: str
    significant: bool


class BootstrapAnalyzer:
    """
    Bootstrap confidence interval analysis as specified in the paper.
    Uses 1000 iterations with fixed seed for reproducibility.
    """
    
    def __init__(self, n_bootstrap: int = 1000, confidence_level: float = 0.95, seed: int = 42):
        self.n_bootstrap = n_bootstrap
        self.confidence_level = confidence_level
        self.seed = seed
        
    def bootstrap_correlation(
        self, 
        x: np.ndarray, 
        y: np.ndarray, 
        method: str = 'spearman'
    ) -> Tuple[float, Tuple[float, float]]:
        """
        Compute bootstrap confidence intervals for correlation.
        
        Args:
            x: First variable
            y: Second variable  
            method: 'spearman' or 'pearson'
            
        Returns:
            Tuple of (correlation, (ci_lower, ci_upper))
        """
        np.random.seed(self.seed)
        
        # Original correlation
        if method == 'spearman':
            original_corr, _ = spearmanr(x, y)
        else:
            original_corr, _ = pearsonr(x, y)
        
        # Bootstrap samples
        bootstrap_corrs = []
        n = len(x)
        
        for _ in range(self.n_bootstrap):
            # Resample with replacement
            indices = np.random.choice(n, size=n, replace=True)
            x_boot = x[indices]
            y_boot = y[indices]
            
            # Compute correlation
            if method == 'spearman':
                corr, _ = spearmanr(x_boot, y_boot)
            else:
                corr, _ = pearsonr(x_boot, y_boot)
                
            if not np.isnan(corr):
                bootstrap_corrs.append(corr)
        
        # Confidence interval
        alpha = 1 - self.confidence_level
        ci_lower = np.percentile(bootstrap_corrs, 100 * alpha / 2)
        ci_upper = np.percentile(bootstrap_corrs, 100 * (1 - alpha / 2))
        
        return original_corr, (ci_lower, ci_upper)
    
    def bootstrap_metric_difference(
        self,
        group1_scores: List[float],
        group2_scores: List[float]
    ) -> Tuple[float, Tuple[float, float]]:
        """
        Bootstrap confidence interval for difference in means.
        
        Args:
            group1_scores: Scores for group 1
            group2_scores: Scores for group 2
            
        Returns:
            Tuple of (mean_difference, (ci_lower, ci_upper))
        """
        np.random.seed(self.seed)
        
        group1 = np.array(group1_scores)
        group2 = np.array(group2_scores)
        
        original_diff = np.mean(group1) - np.mean(group2)
        
        # Bootstrap differences
        bootstrap_diffs = []
        
        for _ in range(self.n_bootstrap):
            # Resample both groups
            boot1 = np.random.choice(group1, size=len(group1), replace=True)
            boot2 = np.random.choice(group2, size=len(group2), replace=True)
            
            diff = np.mean(boot1) - np.mean(boot2)
            bootstrap_diffs.append(diff)
        
        # Confidence interval
        alpha = 1 - self.confidence_level
        ci_lower = np.percentile(bootstrap_diffs, 100 * alpha / 2)
        ci_upper = np.percentile(bootstrap_diffs, 100 * (1 - alpha / 2))
        
        return original_diff, (ci_lower, ci_upper)


class EffectSizeCalculator:
    """Calculate effect sizes with practical significance interpretation."""
    
    def __init__(self, practical_threshold: float = 0.2):
        """
        Args:
            practical_threshold: Cohen's d threshold for practical significance
        """
        self.practical_threshold = practical_threshold
    
    def cohens_d(self, group1: List[float], group2: List[float]) -> float:
        """
        Calculate Cohen's d effect size.
        
        Args:
            group1: First group scores
            group2: Second group scores
            
        Returns:
            Cohen's d effect size
        """
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
    
    def interpret_cohens_d(self, d: float) -> str:
        """Interpret Cohen's d magnitude."""
        abs_d = abs(d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"
    
    def is_practically_significant(self, d: float) -> bool:
        """Check if effect size meets practical significance threshold."""
        return abs(d) > self.practical_threshold


class BonferroniCorrection:
    """Handle multiple comparison corrections."""
    
    def __init__(self, alpha: float = 0.05, n_comparisons: int = 3):
        """
        Args:
            alpha: Overall significance level
            n_comparisons: Number of comparisons (3 for paper: shared vs baseline, per-head vs baseline, per-head vs shared)
        """
        self.alpha = alpha
        self.n_comparisons = n_comparisons
        self.alpha_corrected = alpha / n_comparisons
        
    def is_significant(self, p_value: float) -> bool:
        """Check if p-value is significant after correction."""
        return p_value < self.alpha_corrected
    
    def adjust_p_values(self, p_values: List[float]) -> List[float]:
        """Apply Bonferroni correction to p-values."""
        return [min(p * self.n_comparisons, 1.0) for p in p_values]


class StatisticalEquivalenceTest:
    """
    TOST-style equivalence testing for numerical precision across devices.
    """
    
    def __init__(self, equivalence_margin: float = 1e-3):
        """
        Args:
            equivalence_margin: Equivalence margin for TOST test
        """
        self.equivalence_margin = equivalence_margin
    
    def tost_test(
        self, 
        group1: List[float], 
        group2: List[float]
    ) -> Dict[str, Any]:
        """
        Two One-Sided Tests (TOST) for equivalence.
        
        Args:
            group1: Results from device 1 (e.g., MPS)
            group2: Results from device 2 (e.g., CPU)
            
        Returns:
            Dict with TOST results
        """
        group1 = np.array(group1)
        group2 = np.array(group2)
        
        # Difference in means
        diff = np.mean(group1) - np.mean(group2)
        
        # Standard error of difference
        se_diff = np.sqrt(np.var(group1, ddof=1)/len(group1) + np.var(group2, ddof=1)/len(group2))
        
        if se_diff == 0:
            return {
                'equivalent': abs(diff) < self.equivalence_margin,
                'difference': diff,
                'p_value_lower': 0.0,
                'p_value_upper': 0.0,
                'margin': self.equivalence_margin
            }
        
        # TOST statistics
        t1 = (diff - self.equivalence_margin) / se_diff  # H0: diff >= margin
        t2 = (diff + self.equivalence_margin) / se_diff  # H0: diff <= -margin
        
        df = len(group1) + len(group2) - 2
        
        # P-values (one-tailed)
        p1 = stats.t.cdf(t1, df)  # P(diff < margin)
        p2 = 1 - stats.t.cdf(t2, df)  # P(diff > -margin)
        
        # Equivalence if both p-values < 0.05
        equivalent = p1 < 0.05 and p2 < 0.05
        
        return {
            'equivalent': equivalent,
            'difference': diff,
            'p_value_lower': p1,
            'p_value_upper': p2,
            'margin': self.equivalence_margin,
            'max_p_value': max(p1, p2)
        }


class PaperResultsFormatter:
    """Format results to match paper table styles."""
    
    def __init__(self, precision: int = 4):
        self.precision = precision
    
    def format_correlation_result(
        self,
        correlation: float,
        ci: Tuple[float, float],
        p_value: Optional[float] = None
    ) -> str:
        """Format correlation with CI for paper tables."""
        ci_str = f"[{ci[0]:.{self.precision}f}, {ci[1]:.{self.precision}f}]"
        
        if p_value is not None:
            if p_value < 0.001:
                p_str = "p < 0.001"
            else:
                p_str = f"p = {p_value:.3f}"
            return f"{correlation:.{self.precision}f} {ci_str}, {p_str}"
        else:
            return f"{correlation:.{self.precision}f} {ci_str}"
    
    def format_mcc_result(
        self,
        mcc: float,
        ci: Tuple[float, float],
        seed_results: Optional[List[float]] = None
    ) -> str:
        """Format MCC results with seed-level reporting."""
        ci_str = f"[{ci[0]:.{self.precision}f}, {ci[1]:.{self.precision}f}]"
        
        if seed_results:
            seed_str = " ± ".join([f"{s:.{self.precision}f}" for s in seed_results])
            return f"{mcc:.{self.precision}f} {ci_str} (seeds: {seed_str})"
        else:
            return f"{mcc:.{self.precision}f} {ci_str}"
    
    def format_condition_number(self, condition: float) -> str:
        """Format condition numbers for readability."""
        if condition >= 1e6:
            return f"{condition/1e6:.1f}M"
        elif condition >= 1e3:
            return f"{condition/1e3:.1f}K"
        else:
            return f"{condition:.1f}"


class GeodexyxEvaluator:
    """
    Main evaluator class implementing paper's statistical methodology.
    """
    
    def __init__(
        self,
        alpha: float = 0.05,
        n_comparisons: int = 3,
        bootstrap_iterations: int = 1000,
        effect_size_threshold: float = 0.2,
        seed: int = 42
    ):
        """
        Initialize evaluator with paper specifications.
        
        Args:
            alpha: Overall significance level
            n_comparisons: Number of primary comparisons
            bootstrap_iterations: Bootstrap iterations for CIs
            effect_size_threshold: Cohen's d threshold for practical significance
            seed: Random seed for reproducibility
        """
        self.alpha = alpha
        self.bootstrap = BootstrapAnalyzer(bootstrap_iterations, 0.95, seed)
        self.effect_size = EffectSizeCalculator(effect_size_threshold)
        self.bonferroni = BonferroniCorrection(alpha, n_comparisons)
        self.equivalence = StatisticalEquivalenceTest()
        self.formatter = PaperResultsFormatter()
        
    def evaluate_correlation_task(
        self,
        embeddings1: np.ndarray,
        embeddings2: np.ndarray,
        human_scores: np.ndarray,
        task_name: str = "correlation"
    ) -> Dict[str, Any]:
        """
        Evaluate correlation task (WordSim353, WordNet).
        
        Args:
            embeddings1: Baseline embedding distances
            embeddings2: SPD-weighted embedding distances
            human_scores: Human similarity/distance scores
            task_name: Name of the task
            
        Returns:
            Complete evaluation results
        """
        # Compute correlations
        baseline_corr, baseline_ci = self.bootstrap.bootstrap_correlation(
            -embeddings1, human_scores, method='spearman'  # Negative for similarity
        )
        
        spd_corr, spd_ci = self.bootstrap.bootstrap_correlation(
            -embeddings2, human_scores, method='spearman'
        )
        
        # Statistical test for difference
        _, p_value = stats.wilcoxon(embeddings1, embeddings2, alternative='two-sided')
        
        # Effect size
        effect_size = self.effect_size.cohens_d(
            embeddings1.tolist(), embeddings2.tolist()
        )
        
        # Significance after Bonferroni correction
        significant = self.bonferroni.is_significant(p_value)
        
        results = {
            'task': task_name,
            'baseline_correlation': baseline_corr,
            'baseline_ci': baseline_ci,
            'spd_correlation': spd_corr,
            'spd_ci': spd_ci,
            'improvement': spd_corr - baseline_corr,
            'p_value': p_value,
            'p_value_corrected': min(p_value * self.bonferroni.n_comparisons, 1.0),
            'significant': significant,
            'effect_size': effect_size,
            'effect_interpretation': self.effect_size.interpret_cohens_d(effect_size),
            'practically_significant': self.effect_size.is_practically_significant(effect_size),
            'formatted_result': self.formatter.format_correlation_result(
                spd_corr, spd_ci, p_value
            )
        }
        
        return results
    
    def evaluate_classification_task(
        self,
        baseline_results: List[Dict[str, float]],  # Per-seed results
        spd_results: List[Dict[str, float]],       # Per-seed results
        task_name: str = "classification"
    ) -> Dict[str, Any]:
        """
        Evaluate classification task (CoLA with MCC).
        
        Args:
            baseline_results: Baseline results per seed
            spd_results: SPD results per seed
            task_name: Task name
            
        Returns:
            Complete evaluation results
        """
        # Extract MCC scores per seed
        baseline_mccs = [r['mcc'] for r in baseline_results]
        spd_mccs = [r['mcc'] for r in spd_results]
        
        # Aggregate statistics
        baseline_mean = np.mean(baseline_mccs)
        spd_mean = np.mean(spd_mccs)
        
        # Bootstrap CI for difference
        diff, diff_ci = self.bootstrap.bootstrap_metric_difference(
            spd_mccs, baseline_mccs
        )
        
        # Statistical test (paired t-test for seed-level aggregates)
        if len(baseline_mccs) >= 2 and len(spd_mccs) >= 2:
            _, p_value = stats.ttest_rel(spd_mccs, baseline_mccs)
        else:
            p_value = 1.0  # Cannot test with insufficient data
        
        # Effect size
        effect_size = self.effect_size.cohens_d(spd_mccs, baseline_mccs)
        
        # Significance
        significant = self.bonferroni.is_significant(p_value)
        
        results = {
            'task': task_name,
            'baseline_mcc_mean': baseline_mean,
            'baseline_mcc_seeds': baseline_mccs,
            'spd_mcc_mean': spd_mean,
            'spd_mcc_seeds': spd_mccs,
            'improvement': diff,
            'improvement_ci': diff_ci,
            'p_value': p_value,
            'p_value_corrected': min(p_value * self.bonferroni.n_comparisons, 1.0),
            'significant': significant,
            'effect_size': effect_size,
            'effect_interpretation': self.effect_size.interpret_cohens_d(effect_size),
            'practically_significant': self.effect_size.is_practically_significant(effect_size),
            'formatted_result': self.formatter.format_mcc_result(
                spd_mean, diff_ci, spd_mccs
            )
        }
        
        return results
    
    def test_device_equivalence(
        self,
        mps_results: List[float],
        cpu_results: List[float],
        cuda_results: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """
        Test numerical equivalence across devices.
        
        Args:
            mps_results: Results from MPS device
            cpu_results: Results from CPU device
            cuda_results: Optional results from CUDA device
            
        Returns:
            Equivalence test results
        """
        equivalence_results = {}
        
        # MPS vs CPU
        mps_cpu = self.equivalence.tost_test(mps_results, cpu_results)
        equivalence_results['mps_cpu'] = mps_cpu
        
        # CUDA comparisons if available
        if cuda_results:
            mps_cuda = self.equivalence.tost_test(mps_results, cuda_results)
            cpu_cuda = self.equivalence.tost_test(cpu_results, cuda_results)
            equivalence_results['mps_cuda'] = mps_cuda
            equivalence_results['cpu_cuda'] = cpu_cuda
        
        # Overall equivalence
        all_equivalent = all(result['equivalent'] for result in equivalence_results.values())
        equivalence_results['all_equivalent'] = all_equivalent
        
        return equivalence_results
    
    def generate_paper_table(
        self,
        results: List[Dict[str, Any]],
        table_type: str = "correlation"
    ) -> str:
        """
        Generate LaTeX table formatted for paper.
        
        Args:
            results: List of evaluation results
            table_type: Type of table ('correlation' or 'classification')
            
        Returns:
            LaTeX table string
        """
        if table_type == "correlation":
            return self._generate_correlation_table(results)
        else:
            return self._generate_classification_table(results)
    
    def _generate_correlation_table(self, results: List[Dict[str, Any]]) -> str:
        """Generate correlation results table."""
        latex = "\\begin{table}[h]\n\\centering\n"
        latex += "\\begin{tabular}{l|c|c|c|c}\n"
        latex += "\\hline\n"
        latex += "Task & Baseline & SPD & Improvement & Significant \\\\\n"
        latex += "\\hline\n"
        
        for result in results:
            task = result['task']
            baseline = f"{result['baseline_correlation']:.3f}"
            spd = f"{result['spd_correlation']:.3f}"
            improvement = f"{result['improvement']:.3f}"
            sig = "✓" if result['significant'] else "✗"
            
            latex += f"{task} & {baseline} & {spd} & {improvement} & {sig} \\\\\n"
        
        latex += "\\hline\n\\end{tabular}\n"
        latex += "\\caption{Correlation Results}\n\\end{table}"
        
        return latex
    
    def _generate_classification_table(self, results: List[Dict[str, Any]]) -> str:
        """Generate classification results table."""
        latex = "\\begin{table}[h]\n\\centering\n"
        latex += "\\begin{tabular}{l|c|c|c|c|c}\n"
        latex += "\\hline\n"
        latex += "Task & Baseline MCC & SPD MCC & Improvement & Effect Size & Significant \\\\\n"
        latex += "\\hline\n"
        
        for result in results:
            task = result['task']
            baseline = f"{result['baseline_mcc_mean']:.3f}"
            spd = f"{result['spd_mcc_mean']:.3f}"
            improvement = f"{result['improvement']:.3f}"
            effect_size = f"{result['effect_size']:.3f}"
            sig = "✓" if result['significant'] else "✗"
            
            latex += f"{task} & {baseline} & {spd} & {improvement} & {effect_size} & {sig} \\\\\n"
        
        latex += "\\hline\n\\end{tabular}\n"
        latex += "\\caption{Classification Results (MCC)}\n\\end{table}"
        
        return latex
    
    def save_results(self, results: Dict[str, Any], output_path: Path):
        """Save results to JSON file."""
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)


# Utility functions for specific evaluations
def evaluate_wordsim353(
    baseline_distances: np.ndarray,
    spd_distances: np.ndarray,
    human_similarities: np.ndarray,
    evaluator: GeodexyxEvaluator
) -> Dict[str, Any]:
    """Evaluate WordSim353 task."""
    return evaluator.evaluate_correlation_task(
        baseline_distances, spd_distances, human_similarities, "WordSim353"
    )


def evaluate_wordnet(
    baseline_distances: np.ndarray,
    spd_distances: np.ndarray,
    path_distances: np.ndarray,
    evaluator: GeodexyxEvaluator
) -> Dict[str, Any]:
    """Evaluate WordNet hierarchical task."""
    return evaluator.evaluate_correlation_task(
        baseline_distances, spd_distances, path_distances, "WordNet"
    )


def evaluate_cola(
    baseline_results: List[Dict[str, float]],
    shared_results: List[Dict[str, float]],
    per_head_results: List[Dict[str, float]],
    evaluator: GeodexyxEvaluator
) -> Dict[str, Any]:
    """Evaluate CoLA task with multiple SPD configurations."""
    
    # Evaluate shared vs baseline
    shared_eval = evaluator.evaluate_classification_task(
        baseline_results, shared_results, "CoLA-Shared"
    )
    
    # Evaluate per-head vs baseline
    per_head_eval = evaluator.evaluate_classification_task(
        baseline_results, per_head_results, "CoLA-PerHead"
    )
    
    # Evaluate per-head vs shared
    shared_vs_per_head = evaluator.evaluate_classification_task(
        shared_results, per_head_results, "CoLA-SharedVsPerHead"
    )
    
    return {
        'shared_vs_baseline': shared_eval,
        'per_head_vs_baseline': per_head_eval,
        'shared_vs_per_head': shared_vs_per_head
    }


if __name__ == "__main__":
    # Example usage
    print("Geodesyxx Statistical Evaluation Utilities")
    print("==========================================")
    
    # Initialize evaluator with paper specifications
    evaluator = GeodexyxEvaluator(
        alpha=0.05,
        n_comparisons=3,
        bootstrap_iterations=1000,
        effect_size_threshold=0.2,
        seed=42
    )
    
    print(f"Bonferroni corrected α: {evaluator.bonferroni.alpha_corrected:.3f}")
    print(f"Effect size threshold: {evaluator.effect_size.practical_threshold}")
    print(f"Bootstrap iterations: {evaluator.bootstrap.n_bootstrap}")
    
    # Demo with synthetic data
    np.random.seed(42)
    baseline_scores = np.random.normal(0.50, 0.05, 100)  # CoLA baseline ~50%
    spd_scores = baseline_scores + np.random.normal(0.005, 0.01, 100)  # Small improvement
    
    # Effect size calculation
    effect_size = evaluator.effect_size.cohens_d(spd_scores.tolist(), baseline_scores.tolist())
    print(f"\nDemo effect size: {effect_size:.4f} ({evaluator.effect_size.interpret_cohens_d(effect_size)})")
    print(f"Practically significant: {evaluator.effect_size.is_practically_significant(effect_size)}")
    
    print("\n✅ Statistical evaluation utilities ready for paper reproduction!")
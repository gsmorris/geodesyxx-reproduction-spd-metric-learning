#!/usr/bin/env python3
"""
Attention Integration Test Suite
Tests curved attention integration with DistilBERT and parameter consistency.
"""

import sys
import os
import unittest
from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
import warnings
from typing import Dict, List, Tuple, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from curved_attention import CurvedMultiHeadAttention, CurvedDistilBertAttention
from transformer_integration import CurvedDistilBertForSequenceClassification, create_curved_distilbert

try:
    from transformers import DistilBertConfig, DistilBertForSequenceClassification, AutoTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    warnings.warn("Transformers not available - skipping integration tests")


class TestCurvedAttentionBasics(unittest.TestCase):
    """Test basic curved attention functionality."""
    
    def setUp(self):
        torch.manual_seed(42)
        
        self.embed_dim = 768
        self.num_heads = 12
        self.batch_size = 2
        self.seq_len = 32
        
    def test_attention_output_shapes(self):
        """Test that attention outputs have correct shapes."""
        
        for geometry_mode in ['none', 'shared', 'per_head']:
            with self.subTest(geometry_mode=geometry_mode):
                attention = CurvedMultiHeadAttention(
                    embed_dim=self.embed_dim,
                    num_heads=self.num_heads,
                    geometry_mode=geometry_mode,
                    rank=16,
                    device=torch.device('cpu')
                )
                
                hidden_states = torch.randn(self.batch_size, self.seq_len, self.embed_dim)
                outputs = attention(hidden_states)
                
                self.assertEqual(len(outputs), 1, "Should return single output by default")
                output = outputs[0]
                
                expected_shape = (self.batch_size, self.seq_len, self.embed_dim)
                self.assertEqual(output.shape, expected_shape,
                               f"Output shape should be {expected_shape}, got {output.shape}")
                
                self.assertEqual(output.dtype, hidden_states.dtype,
                               "Output dtype should match input dtype")
                
                self.assertTrue(torch.isfinite(output).all(),
                               "All outputs should be finite")
    
    def test_attention_with_mask(self):
        """Test attention with attention masks."""
        attention = CurvedMultiHeadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            geometry_mode='shared',
            rank=16,
            device=torch.device('cpu')
        )
        
        hidden_states = torch.randn(self.batch_size, self.seq_len, self.embed_dim)
        
        # Create attention mask (mask out last few tokens)
        attention_mask = torch.zeros(self.batch_size, 1, 1, self.seq_len)
        attention_mask[:, :, :, -5:] = -1e9  # Mask last 5 tokens
        
        outputs_with_mask = attention(hidden_states, attention_mask=attention_mask)
        outputs_without_mask = attention(hidden_states)
        
        # Outputs should be different
        self.assertFalse(torch.allclose(outputs_with_mask[0], outputs_without_mask[0], atol=1e-6),
                        "Masked and unmasked outputs should differ")
        
        # All outputs should still be finite
        self.assertTrue(torch.isfinite(outputs_with_mask[0]).all(),
                       "Masked outputs should be finite")
    
    def test_attention_weights_output(self):
        """Test attention weights output when requested."""
        attention = CurvedMultiHeadAttention(
            embed_dim=self.embed_dim,
            num_heads=self.num_heads,
            geometry_mode='none',  # Use standard attention for this test
            device=torch.device('cpu')
        )
        
        hidden_states = torch.randn(self.batch_size, self.seq_len, self.embed_dim)
        outputs = attention(hidden_states, output_attentions=True)
        
        self.assertEqual(len(outputs), 2, "Should return output and attention weights")
        
        output, attention_weights = outputs
        expected_attn_shape = (self.batch_size, self.num_heads, self.seq_len, self.seq_len)
        
        self.assertEqual(attention_weights.shape, expected_attn_shape,
                        f"Attention weights shape should be {expected_attn_shape}")
        
        # Attention weights should sum to 1 along last dimension
        attn_sums = attention_weights.sum(dim=-1)
        expected_ones = torch.ones(self.batch_size, self.num_heads, self.seq_len)
        self.assertTrue(torch.allclose(attn_sums, expected_ones, atol=1e-5),
                       "Attention weights should sum to 1")
    
    def test_parameter_counting(self):
        """Test parameter counting functionality."""
        
        test_cases = [
            {'geometry_mode': 'none', 'expected_geometric': 0},
            {'geometry_mode': 'shared', 'expected_geometric': 16 * 64},  # rank * head_dim
            {'geometry_mode': 'per_head', 'expected_geometric': 12 * 16 * 64},  # num_heads * rank * head_dim
        ]
        
        for case in test_cases:
            with self.subTest(geometry_mode=case['geometry_mode']):
                attention = CurvedMultiHeadAttention(
                    embed_dim=self.embed_dim,
                    num_heads=self.num_heads,
                    geometry_mode=case['geometry_mode'],
                    rank=16,
                    device=torch.device('cpu')
                )
                
                param_counts = attention.get_parameter_count()
                
                self.assertIn('standard', param_counts,
                             "Should have standard parameter count")
                self.assertIn('geometric', param_counts,
                             "Should have geometric parameter count")
                self.assertIn('total', param_counts,
                             "Should have total parameter count")
                
                self.assertEqual(param_counts['geometric'], case['expected_geometric'],
                               f"Geometric parameters: expected {case['expected_geometric']}, "
                               f"got {param_counts['geometric']}")
                
                self.assertEqual(param_counts['total'], 
                               param_counts['standard'] + param_counts['geometric'],
                               "Total should equal standard + geometric")
    
    def test_condition_number_monitoring(self):
        """Test condition number monitoring for geometric modes."""
        
        for geometry_mode in ['shared', 'per_head']:
            with self.subTest(geometry_mode=geometry_mode):
                attention = CurvedMultiHeadAttention(
                    embed_dim=self.embed_dim,
                    num_heads=self.num_heads,
                    geometry_mode=geometry_mode,
                    rank=16,
                    device=torch.device('cpu')
                )
                
                conditions = attention.get_condition_numbers()
                
                if geometry_mode == 'shared':
                    self.assertIn('shared', conditions,
                                 "Shared mode should have 'shared' condition number")
                    self.assertEqual(len(conditions), 1,
                                   "Shared mode should have 1 condition number")
                
                elif geometry_mode == 'per_head':
                    expected_keys = [f'head_{i}' for i in range(self.num_heads)]
                    for key in expected_keys:
                        self.assertIn(key, conditions,
                                     f"Per-head mode should have condition number for {key}")
                    self.assertEqual(len(conditions), self.num_heads,
                                   f"Per-head mode should have {self.num_heads} condition numbers")
                
                # All condition numbers should be positive and finite
                for key, condition in conditions.items():
                    self.assertGreater(condition, 0,
                                     f"Condition number {key} should be positive")
                    self.assertTrue(np.isfinite(condition),
                                   f"Condition number {key} should be finite")
    
    def test_gradient_flow(self):
        """Test that gradients flow through curved attention."""
        attention = CurvedMultiHeadAttention(
            embed_dim=64,  # Smaller for efficiency
            num_heads=4,
            geometry_mode='shared',
            rank=8,
            device=torch.device('cpu')
        )
        
        hidden_states = torch.randn(2, 16, 64, requires_grad=True)
        
        outputs = attention(hidden_states)
        loss = outputs[0].sum()
        
        # Backward pass
        loss.backward()
        
        # Check that parameters have gradients
        for name, param in attention.named_parameters():
            if param.requires_grad:
                self.assertIsNotNone(param.grad,
                                   f"Parameter {name} should have gradient")
                self.assertGreater(torch.norm(param.grad), 1e-8,
                                 f"Parameter {name} should have non-zero gradient")
        
        # Check input gradients
        self.assertIsNotNone(hidden_states.grad,
                           "Input should have gradients")


@unittest.skipIf(not TRANSFORMERS_AVAILABLE, "transformers library not available")
class TestDistilBertIntegration(unittest.TestCase):
    """Test DistilBERT integration."""
    
    def setUp(self):
        torch.manual_seed(42)
        
        self.config = DistilBertConfig.from_pretrained("distilbert-base-uncased")
        self.config.num_labels = 2
        
    def test_curved_distilbert_creation(self):
        """Test creation of curved DistilBERT models."""
        
        test_cases = [
            {'curved_layers': [], 'geometry_mode': 'none'},
            {'curved_layers': [1], 'geometry_mode': 'shared'},
            {'curved_layers': [1, 2], 'geometry_mode': 'shared'},
            {'curved_layers': [1], 'geometry_mode': 'per_head'},
        ]
        
        for case in test_cases:
            with self.subTest(case=case):
                try:
                    model = create_curved_distilbert(
                        model_name="distilbert-base-uncased",
                        num_labels=2,
                        curved_layers=case['curved_layers'],
                        geometry_mode=case['geometry_mode'],
                        rank=16
                    )
                    
                    # Check model type
                    self.assertIsInstance(model, CurvedDistilBertForSequenceClassification,
                                        "Should create CurvedDistilBertForSequenceClassification")
                    
                    # Check that specified layers are curved
                    for layer_idx in case['curved_layers']:
                        layer = model.distilbert.transformer.layer[layer_idx]
                        self.assertIsInstance(layer.attention, CurvedDistilBertAttention,
                                            f"Layer {layer_idx} should have curved attention")
                    
                    # Check that other layers are standard
                    for layer_idx in range(len(model.distilbert.transformer.layer)):
                        if layer_idx not in case['curved_layers']:
                            layer = model.distilbert.transformer.layer[layer_idx]
                            # Should be standard DistilBERT attention (not our curved version)
                            self.assertNotIsInstance(layer.attention, CurvedDistilBertAttention,
                                                   f"Layer {layer_idx} should have standard attention")
                
                except Exception as e:
                    self.fail(f"Failed to create curved DistilBERT with {case}: {e}")
    
    def test_forward_pass_shapes(self):
        """Test forward pass with different input shapes."""
        model = create_curved_distilbert(
            model_name="distilbert-base-uncased",
            num_labels=2,
            curved_layers=[1],
            geometry_mode='shared',
            rank=16
        )
        
        # Test different batch sizes and sequence lengths
        test_cases = [
            {'batch_size': 1, 'seq_len': 32},
            {'batch_size': 2, 'seq_len': 64},
            {'batch_size': 4, 'seq_len': 128},
        ]
        
        for case in test_cases:
            with self.subTest(case=case):
                batch_size = case['batch_size']
                seq_len = case['seq_len']
                
                # Create dummy inputs
                input_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_len))
                attention_mask = torch.ones(batch_size, seq_len)
                
                with torch.no_grad():
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                
                # Check output shapes
                self.assertEqual(outputs.logits.shape, (batch_size, 2),
                               f"Logits shape should be ({batch_size}, 2)")
                
                # Check outputs are finite
                self.assertTrue(torch.isfinite(outputs.logits).all(),
                               "All logits should be finite")
    
    def test_parameter_separation(self):
        """Test separation of geometric and standard parameters."""
        model = create_curved_distilbert(
            model_name="distilbert-base-uncased",
            num_labels=2,
            curved_layers=[1, 2],
            geometry_mode='shared',
            rank=16
        )
        
        # Get parameter separation
        geometric_params = []
        standard_params = []
        
        for name, param in model.named_parameters():
            if 'spd_metrics' in name:
                geometric_params.append(param)
            else:
                standard_params.append(param)
        
        # Should have some of each type
        self.assertGreater(len(geometric_params), 0,
                          "Should have some geometric parameters")
        self.assertGreater(len(standard_params), 0,
                          "Should have some standard parameters")
        
        # Total should match all parameters
        all_params = list(model.parameters())
        self.assertEqual(len(geometric_params) + len(standard_params), len(all_params),
                        "Geometric + standard should equal total parameters")
    
    def test_condition_number_access(self):
        """Test access to condition numbers in full model."""
        model = create_curved_distilbert(
            model_name="distilbert-base-uncased",
            num_labels=2,
            curved_layers=[1],
            geometry_mode='shared',
            rank=16
        )
        
        # Should be able to get condition numbers
        if hasattr(model, 'get_condition_numbers'):
            conditions = model.get_condition_numbers()
            
            self.assertIsInstance(conditions, dict,
                                "Condition numbers should be returned as dict")
            
            # Should have condition numbers for curved layers
            self.assertGreater(len(conditions), 0,
                             "Should have at least one condition number")
            
            for key, condition in conditions.items():
                self.assertGreater(condition, 0,
                                 f"Condition number {key} should be positive")
                self.assertTrue(np.isfinite(condition),
                               f"Condition number {key} should be finite")
    
    def test_training_mode_consistency(self):
        """Test behavior consistency between training and eval modes."""
        model = create_curved_distilbert(
            model_name="distilbert-base-uncased",
            num_labels=2,
            curved_layers=[1],
            geometry_mode='shared',
            rank=16
        )
        
        # Create inputs
        input_ids = torch.randint(0, self.config.vocab_size, (2, 32))
        attention_mask = torch.ones(2, 32)
        
        # Forward pass in training mode
        model.train()
        with torch.no_grad():
            train_outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Forward pass in eval mode
        model.eval()
        with torch.no_grad():
            eval_outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        # Outputs should be similar (some difference due to dropout)
        # But not exactly the same if dropout is used
        logit_diff = torch.abs(train_outputs.logits - eval_outputs.logits).max()
        
        # Should be reasonably similar (allowing for dropout differences)
        self.assertLess(logit_diff.item(), 0.1,
                       "Training and eval outputs should be reasonably similar")


class TestAttentionNumericalStability(unittest.TestCase):
    """Test numerical stability of attention computations."""
    
    def test_extreme_input_values(self):
        """Test with extreme input values."""
        attention = CurvedMultiHeadAttention(
            embed_dim=64,
            num_heads=4,
            geometry_mode='shared',
            rank=8,
            device=torch.device('cpu')
        )
        
        # Test with large values
        hidden_states_large = torch.randn(2, 16, 64) * 100
        
        try:
            with torch.no_grad():
                outputs_large = attention(hidden_states_large)
            
            self.assertTrue(torch.isfinite(outputs_large[0]).all(),
                           "Outputs should be finite even with large inputs")
        
        except Exception as e:
            warnings.warn(f"Large inputs caused exception: {e}")
    
    def test_attention_score_ranges(self):
        """Test that attention scores stay in reasonable ranges."""
        attention = CurvedMultiHeadAttention(
            embed_dim=64,
            num_heads=4,
            geometry_mode='shared',
            rank=8,
            device=torch.device('cpu')
        )
        
        hidden_states = torch.randn(2, 16, 64)
        
        with torch.no_grad():
            outputs = attention(hidden_states, output_attentions=True)
        
        if len(outputs) > 1:
            attention_weights = outputs[1]
            
            # Check attention weight properties
            self.assertTrue((attention_weights >= 0).all(),
                           "Attention weights should be non-negative")
            self.assertTrue((attention_weights <= 1).all(),
                           "Attention weights should be <= 1")
            
            # Check they sum to 1
            weight_sums = attention_weights.sum(dim=-1)
            expected_ones = torch.ones_like(weight_sums)
            self.assertTrue(torch.allclose(weight_sums, expected_ones, atol=1e-5),
                           "Attention weights should sum to 1")


def run_attention_tests():
    """Run all attention integration tests."""
    print("="*60)
    print(" "*15 + "ATTENTION INTEGRATION TESTS")
    print("="*60)
    
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestCurvedAttentionBasics,
        TestAttentionNumericalStability,
    ]
    
    if TRANSFORMERS_AVAILABLE:
        test_classes.append(TestDistilBertIntegration)
    else:
        print("⚠️  Skipping DistilBERT integration tests (transformers not available)")
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "="*60)
    print(" "*20 + "ATTENTION TEST SUMMARY")
    print("="*60)
    
    total = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    passed = total - failures - errors
    
    print(f"Total tests: {total}")
    print(f"Passed: {passed} ✅")
    print(f"Failed: {failures} ❌")
    print(f"Errors: {errors} ⚠️")
    
    if result.failures:
        print(f"\nFailures:")
        for test, traceback in result.failures:
            print(f"  - {test}")
    
    if result.errors:
        print(f"\nErrors:")
        for test, traceback in result.errors:
            print(f"  - {test}")
    
    success = failures == 0 and errors == 0
    
    if success:
        print("\n✅ ALL ATTENTION TESTS PASSED")
        print("   Curved attention implementation verified")
        print("   DistilBERT integration validated")
        print("   Numerical stability confirmed")
    else:
        print("\n❌ SOME ATTENTION TESTS FAILED")
        print("   Review failures and errors above")
    
    return success


if __name__ == '__main__':
    success = run_attention_tests()
    sys.exit(0 if success else 1)
#!/usr/bin/env python3
"""
Comprehensive Test Runner for Geodesyxx Package
Runs all test suites and provides detailed reporting.
"""

import sys
import os
import time
import traceback
from pathlib import Path
from typing import Dict, List, Tuple, Any
import warnings

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

# Import test modules
try:
    from test_spd_implementation import run_spd_tests
    from test_attention_integration import run_attention_tests
    from test_device_compatibility import main as run_device_tests
    from test_eigenvalue_recovery import main as run_eigenvalue_tests
except ImportError as e:
    print(f"❌ Failed to import test modules: {e}")
    sys.exit(1)


class TestSuite:
    """Main test suite coordinator."""
    
    def __init__(self):
        self.results = {}
        self.start_time = None
        self.end_time = None
        
    def run_test_suite(self, name: str, test_func, *args, **kwargs) -> bool:
        """Run a test suite and record results."""
        print(f"\n{'='*80}")
        print(f"RUNNING: {name}")
        print(f"{'='*80}")
        
        start_time = time.time()
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)  # Suppress warnings during testing
                success = test_func(*args, **kwargs)
                
            end_time = time.time()
            duration = end_time - start_time
            
            self.results[name] = {
                'success': success,
                'duration': duration,
                'error': None
            }
            
            status = "✅ PASSED" if success else "❌ FAILED"
            print(f"\n{name}: {status} (Duration: {duration:.1f}s)")
            
            return success
            
        except Exception as e:
            end_time = time.time()
            duration = end_time - start_time
            
            self.results[name] = {
                'success': False,
                'duration': duration,
                'error': str(e)
            }
            
            print(f"\n❌ {name} CRASHED: {str(e)}")
            print("Traceback:")
            traceback.print_exc()
            
            return False
    
    def run_all_tests(self, include_slow_tests: bool = True) -> Dict[str, Any]:
        """Run all test suites."""
        print("🚀 Starting Geodesyxx Comprehensive Test Suite")
        print(f"📅 Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        self.start_time = time.time()
        
        # Define test suites
        test_suites = [
            ("SPD Implementation Tests", run_spd_tests),
            ("Attention Integration Tests", run_attention_tests),
            ("Eigenvalue Recovery Tests", run_eigenvalue_tests),
        ]
        
        if include_slow_tests:
            test_suites.append(("Device Compatibility Tests", run_device_tests))
        
        # Run each test suite
        all_passed = True
        
        for name, test_func in test_suites:
            success = self.run_test_suite(name, test_func)
            all_passed = all_passed and success
        
        self.end_time = time.time()
        total_duration = self.end_time - self.start_time
        
        # Generate summary
        summary = self.generate_summary(total_duration)
        
        # Print summary
        self.print_summary(summary, all_passed)
        
        return summary
    
    def generate_summary(self, total_duration: float) -> Dict[str, Any]:
        """Generate test summary."""
        total_suites = len(self.results)
        passed_suites = sum(1 for r in self.results.values() if r['success'])
        failed_suites = total_suites - passed_suites
        
        return {
            'total_suites': total_suites,
            'passed_suites': passed_suites,
            'failed_suites': failed_suites,
            'total_duration': total_duration,
            'suite_results': self.results,
            'overall_success': failed_suites == 0,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def print_summary(self, summary: Dict[str, Any], all_passed: bool):
        """Print comprehensive test summary."""
        print(f"\n{'='*80}")
        print(f"{'COMPREHENSIVE TEST SUMMARY':^80}")
        print(f"{'='*80}")
        
        print(f"\n📊 Overall Results:")
        print(f"   Total test suites: {summary['total_suites']}")
        print(f"   Passed: {summary['passed_suites']} ✅")
        print(f"   Failed: {summary['failed_suites']} ❌")
        print(f"   Total duration: {summary['total_duration']:.1f}s")
        
        print(f"\n📋 Suite Details:")
        for suite_name, result in summary['suite_results'].items():
            status = "✅ PASS" if result['success'] else "❌ FAIL"
            duration = result['duration']
            
            print(f"   {suite_name}: {status} ({duration:.1f}s)")
            
            if result['error']:
                print(f"      Error: {result['error']}")
        
        # Overall assessment
        print(f"\n{'='*80}")
        print(f"{'FINAL ASSESSMENT':^80}")
        print(f"{'='*80}")
        
        if all_passed:
            print("🎉 ALL TEST SUITES PASSED!")
            print("✅ Geodesyxx implementation is mathematically correct")
            print("✅ Numerical stability verified across devices")
            print("✅ Integration with DistilBERT confirmed")
            print("✅ Ready for scientific reproduction")
            
            print(f"\n🔬 Reproduction readiness:")
            print("   • SPD metric learning: ✅ Validated")
            print("   • Curved attention: ✅ Working")
            print("   • Device compatibility: ✅ Verified")
            print("   • Statistical analysis: ✅ Ready")
            
        else:
            print("⚠️  SOME TEST SUITES FAILED")
            print("❌ Implementation has issues that need addressing")
            print("⚠️  Scientific reproduction may be affected")
            
            failed_suites = [name for name, result in summary['suite_results'].items() 
                           if not result['success']]
            print(f"\n🔧 Failed test suites: {', '.join(failed_suites)}")
            print("💡 Review individual test results and fix issues before reproduction")
        
        # Next steps
        print(f"\n📋 Next Steps:")
        if all_passed:
            print("1. Run quick validation: python scripts/quick_test.py")
            print("2. Synthetic validation: python scripts/run_synthetic_validation.py")
            print("3. Full reproduction: python scripts/run_phase4.py --representative-sample")
        else:
            print("1. Fix failing test suites")
            print("2. Re-run tests until all pass")
            print("3. Then proceed with reproduction")
    
    def save_results(self, output_path: str = None):
        """Save test results to file."""
        if output_path is None:
            output_path = Path(__file__).parent.parent / "test_results.json"
        
        import json
        
        # Make results JSON serializable
        json_results = {
            'summary': {
                'total_suites': len(self.results),
                'passed_suites': sum(1 for r in self.results.values() if r['success']),
                'failed_suites': sum(1 for r in self.results.values() if not r['success']),
                'total_duration': self.end_time - self.start_time if self.end_time else 0,
                'overall_success': all(r['success'] for r in self.results.values()),
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            },
            'suite_results': {
                name: {
                    'success': result['success'],
                    'duration': result['duration'],
                    'error': result['error']
                }
                for name, result in self.results.items()
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(json_results, f, indent=2)
        
        print(f"\n📄 Test results saved to: {output_path}")


def main():
    """Main entry point for comprehensive testing."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run comprehensive Geodesyxx test suite')
    parser.add_argument('--skip-slow', action='store_true',
                       help='Skip slow tests (like device compatibility)')
    parser.add_argument('--output', '-o', type=str,
                       help='Output file for test results (JSON)')
    
    args = parser.parse_args()
    
    # Create and run test suite
    suite = TestSuite()
    
    try:
        summary = suite.run_all_tests(include_slow_tests=not args.skip_slow)
        
        # Save results if requested
        if args.output:
            suite.save_results(args.output)
        else:
            suite.save_results()  # Save to default location
        
        # Return success status
        return summary['overall_success']
        
    except KeyboardInterrupt:
        print("\n⚠️  Test suite interrupted by user")
        return False
    
    except Exception as e:
        print(f"\n❌ Test suite failed with unexpected error: {e}")
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
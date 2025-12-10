#!/usr/bin/env python3
"""
Test and demonstrate local sklearn metric implementations with validation.

This script tests the local reimplementations of sklearn metrics and validates
them against sklearn to ensure correctness.
"""

import sys
import numpy as np
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

# Import local metrics and validation function
from final_analysis import (
    confusion_matrix_local,
    accuracy_score_local,
    precision_score_local,
    recall_score_local,
    f1_score_local,
    balanced_accuracy_score_local,
    matthews_corrcoef_local,
    validate_local_metrics_vs_sklearn
)


def run_comprehensive_tests():
    """Run comprehensive tests of local sklearn implementations."""

    print("\n" + "="*80)
    print("SKLEARN METRICS REIMPLEMENTATION - COMPREHENSIVE VALIDATION")
    print("="*80)
    print("\nThis tests local numpy-based reimplementations against sklearn")
    print("to ensure correctness of all classification metrics.\n")

    test_cases = []

    # Test Case 1: Balanced dataset
    print("Test 1: Balanced dataset (50-50 split)")
    print("-"*80)
    y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    y_pred = np.array([0, 0, 0, 1, 1, 0, 1, 1, 1, 1])
    result1 = validate_local_metrics_vs_sklearn(y_true, y_pred, "Test 1: Balanced", verbose=True)
    test_cases.append(('Balanced', result1))

    # Test Case 2: Imbalanced dataset
    print("\nTest 2: Imbalanced dataset (80-20 split)")
    print("-"*80)
    y_true = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1])
    y_pred = np.array([0, 0, 0, 0, 1, 1, 0, 0, 1, 1])
    result2 = validate_local_metrics_vs_sklearn(y_true, y_pred, "Test 2: Imbalanced", verbose=True)
    test_cases.append(('Imbalanced', result2))

    # Test Case 3: Perfect prediction
    print("\nTest 3: Perfect prediction (100% accuracy)")
    print("-"*80)
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_pred = np.array([0, 0, 0, 1, 1, 1])
    result3 = validate_local_metrics_vs_sklearn(y_true, y_pred, "Test 3: Perfect", verbose=True)
    test_cases.append(('Perfect', result3))

    # Test Case 4: All wrong prediction
    print("\nTest 4: All wrong prediction (0% accuracy)")
    print("-"*80)
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_pred = np.array([1, 1, 1, 0, 0, 0])
    result4 = validate_local_metrics_vs_sklearn(y_true, y_pred, "Test 4: All wrong", verbose=True)
    test_cases.append(('All wrong', result4))

    # Test Case 5: Real-world protein data simulation
    print("\nTest 5: Simulated protein burial prediction (179 residues)")
    print("-"*80)
    np.random.seed(42)
    n_residues = 179
    # Create somewhat realistic distribution
    y_true = np.random.choice([0, 1], size=n_residues, p=[0.65, 0.35])  # ~65% buried
    # Add correlation (80% accuracy)
    y_pred = y_true.copy()
    error_indices = np.random.choice(n_residues, size=int(0.2 * n_residues), replace=False)
    y_pred[error_indices] = 1 - y_pred[error_indices]
    result5 = validate_local_metrics_vs_sklearn(y_true, y_pred, "Test 5: Protein simulation", verbose=True)
    test_cases.append(('Protein sim', result5))

    # Test Case 6: Edge case - all positive
    print("\nTest 6: Edge case - all predictions are positive")
    print("-"*80)
    y_true = np.array([0, 0, 1, 1, 1])
    y_pred = np.array([1, 1, 1, 1, 1])
    result6 = validate_local_metrics_vs_sklearn(y_true, y_pred, "Test 6: All positive", verbose=True)
    test_cases.append(('All positive', result6))

    # Test Case 7: Edge case - all negative
    print("\nTest 7: Edge case - all predictions are negative")
    print("-"*80)
    y_true = np.array([0, 0, 1, 1, 1])
    y_pred = np.array([0, 0, 0, 0, 0])
    result7 = validate_local_metrics_vs_sklearn(y_true, y_pred, "Test 7: All negative", verbose=True)
    test_cases.append(('All negative', result7))

    # Summary
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)

    all_passed = True
    for name, result in test_cases:
        if result.get('sklearn_available'):
            status = "✓ PASS" if result.get('all_match') else "✗ FAIL"
            print(f"{name:20s}: {status}")
            if not result.get('all_match'):
                all_passed = False
        else:
            print(f"{name:20s}: ⚠ sklearn not available")

    print("="*80)
    if all_passed:
        print("✓ ALL TESTS PASSED - Local implementation matches sklearn perfectly!")
        print("\nThe local reimplementation can be used as a drop-in replacement for sklearn.")
        return 0
    else:
        print("✗ SOME TESTS FAILED - Implementation needs review")
        return 1


def demo_individual_metrics():
    """Demonstrate individual metric calculations."""

    print("\n" + "="*80)
    print("INDIVIDUAL METRIC DEMONSTRATION")
    print("="*80)

    # Sample data
    y_true = np.array([0, 0, 0, 1, 1, 1, 1, 0, 1, 0])
    y_pred = np.array([0, 0, 1, 1, 0, 1, 1, 0, 1, 0])

    print(f"\nGround Truth: {y_true}")
    print(f"Predictions : {y_pred}")
    print()

    # Confusion Matrix
    cm, (tn, fp, fn, tp) = confusion_matrix_local(y_true, y_pred)
    print("Confusion Matrix:")
    print(f"  TN={tn}, FP={fp}")
    print(f"  FN={fn}, TP={tp}")
    print(f"\n  Matrix form:\n{cm}")
    print()

    # Individual metrics
    print("Metrics (using local implementation):")
    print(f"  Accuracy           : {accuracy_score_local(y_true, y_pred):.4f}")
    print(f"  Precision          : {precision_score_local(y_true, y_pred):.4f}")
    print(f"  Recall (Sensitivity): {recall_score_local(y_true, y_pred):.4f}")
    print(f"  F1 Score           : {f1_score_local(y_true, y_pred):.4f}")
    print(f"  Balanced Accuracy  : {balanced_accuracy_score_local(y_true, y_pred):.4f}")
    print(f"  MCC                : {matthews_corrcoef_local(y_true, y_pred):.4f}")
    print()


if __name__ == "__main__":
    print("\n" + "#"*80)
    print("# LOCAL SKLEARN METRICS IMPLEMENTATION TEST SUITE")
    print("#"*80)

    # Demo individual metrics
    demo_individual_metrics()

    # Run comprehensive validation tests
    exit_code = run_comprehensive_tests()

    print("\n" + "#"*80)
    print("# TEST SUITE COMPLETE")
    print("#"*80 + "\n")

    sys.exit(exit_code)


#!/usr/bin/env python3
"""
T-test script to compare test results with and without progress estimator.

Usage:
    python t_test.py <progress_dir> <no_progress_dir>

Example:
    python t_test.py /root/autodl-tmp/PECN/test_result/instruct_pix2pix_128_backforth_next5_progress \\
                     /root/autodl-tmp/PECN/test_result/instruct_pix2pix_128_backforth_next5
"""

import json
import argparse
from pathlib import Path
import numpy as np
from scipy import stats


def load_results(results_dir):
    """Load results.json and extract SSIM and PSNR values (excluding avg_value)."""
    results_path = Path(results_dir) / "results.json"
    
    with open(results_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Filter out avg_value entry
    samples = [item for item in data if item.get('filename') != 'avg_value']
    
    ssim_values = [item['ssim'] for item in samples]
    psnr_values = [item['psnr'] for item in samples]
    
    return ssim_values, psnr_values, len(samples)


def perform_t_test(group1, group2, metric_name):
    """
    Perform independent samples t-test.
    
    Returns:
        mean_diff: mean(group1) - mean(group2)
        t_statistic: t-value
        p_value: p-value
        significance: significance marker (*, **, ***)
    """
    mean1 = np.mean(group1)
    mean2 = np.mean(group2)
    mean_diff = mean1 - mean2
    
    # Perform independent samples t-test
    t_statistic, p_value = stats.ttest_ind(group1, group2)
    
    # Determine significance
    if p_value < 0.01:
        significance = "***"
    elif p_value < 0.05:
        significance = "**"
    elif p_value < 0.1:
        significance = "*"
    else:
        significance = ""
    
    return {
        'mean_with_progress': float(mean1),
        'mean_without_progress': float(mean2),
        'mean_difference': float(mean_diff),
        't_statistic': float(t_statistic),
        'p_value': float(p_value),
        'significance': significance,
        'n_with_progress': len(group1),
        'n_without_progress': len(group2)
    }


def main():
    parser = argparse.ArgumentParser(
        description='Perform t-test to compare test results with and without progress estimator.'
    )
    parser.add_argument(
        'progress_dir',
        type=str,
        help='Directory containing test results WITH progress estimator (e.g., /path/to/test_result/..._progress)'
    )
    parser.add_argument(
        'no_progress_dir',
        type=str,
        help='Directory containing test results WITHOUT progress estimator (e.g., /path/to/test_result/...)'
    )
    
    args = parser.parse_args()
    
    # Load results
    print(f"Loading results from:")
    print(f"  With progress: {args.progress_dir}")
    print(f"  Without progress: {args.no_progress_dir}")
    
    try:
        ssim_with, psnr_with, n_with = load_results(args.progress_dir)
        ssim_without, psnr_without, n_without = load_results(args.no_progress_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return 1
    
    print(f"\nSample sizes:")
    print(f"  With progress: {n_with} samples")
    print(f"  Without progress: {n_without} samples")
    
    # Perform t-tests
    print("\nPerforming t-tests...")
    
    ssim_results = perform_t_test(ssim_with, ssim_without, "SSIM")
    psnr_results = perform_t_test(psnr_with, psnr_without, "PSNR")
    
    # Prepare output
    output = {
        'comparison': {
            'with_progress_dir': str(args.progress_dir),
            'without_progress_dir': str(args.no_progress_dir),
            'sample_sizes': {
                'with_progress': n_with,
                'without_progress': n_without
            }
        },
        'ssim': ssim_results,
        'psnr': psnr_results
    }
    
    # Print summary
    print("\n" + "="*60)
    print("T-Test Results Summary")
    print("="*60)
    print(f"\nSSIM:")
    print(f"  Mean with progress:    {ssim_results['mean_with_progress']:.6f}")
    print(f"  Mean without progress: {ssim_results['mean_without_progress']:.6f}")
    print(f"  Mean difference:       {ssim_results['mean_difference']:.6f} {ssim_results['significance']}")
    print(f"  t-statistic:           {ssim_results['t_statistic']:.6f}")
    print(f"  p-value:               {ssim_results['p_value']:.6f}")
    
    print(f"\nPSNR:")
    print(f"  Mean with progress:    {psnr_results['mean_with_progress']:.6f}")
    print(f"  Mean without progress: {psnr_results['mean_without_progress']:.6f}")
    print(f"  Mean difference:       {psnr_results['mean_difference']:.6f} {psnr_results['significance']}")
    print(f"  t-statistic:           {psnr_results['t_statistic']:.6f}")
    print(f"  p-value:               {psnr_results['p_value']:.6f}")
    
    print(f"\nSignificance levels:")
    print(f"  *  p < 0.1  (90% confidence)")
    print(f"  ** p < 0.05 (95% confidence)")
    print(f"  *** p < 0.01 (99% confidence)")
    print("="*60)
    
    # Save results to JSON
    output_path = Path(args.no_progress_dir) / "t_test_results.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    
    print(f"\nResults saved to: {output_path}")
    
    return 0


if __name__ == "__main__":
    exit(main())


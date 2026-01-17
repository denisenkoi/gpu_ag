#!/usr/bin/env python3
"""
Grid search for self-correlation penalty parameters.
Tests threshold (clamp) and weight on 2 degraded wells.
"""
import os
import subprocess
import sys

def main():
    wells = ['Well1675~EGFDL', 'Well239~EGFDL']

    # Grid search parameters
    thresholds = [0, 12, 14, 16]  # 0 = disabled
    weights = [0, 0.1, 0.5, 1.0]

    print("Grid Search: Self-Correlation Penalty")
    print("=" * 80)
    print(f"Wells: {wells}")
    print(f"Thresholds: {thresholds}")
    print(f"Weights: {weights}")
    print()

    results = []

    for thresh in thresholds:
        for weight in weights:
            if thresh == 0 and weight > 0:
                continue
            if thresh > 0 and weight == 0:
                continue

            print(f"\n--- threshold={thresh}, weight={weight} ---")

            # Set environment
            env = os.environ.copy()
            env['USE_PSEUDO_TYPELOG'] = 'true'
            env['SELFCORR_THRESHOLD'] = str(thresh)
            env['SELFCORR_WEIGHT'] = str(weight)
            env['CUDA_VISIBLE_DEVICES'] = '1'

            # Run full_well_optimizer for 2 wells
            cmd = [
                'python', 'full_well_optimizer.py',
                '--wells'] + wells + [
                '--angle-range', '2.0',
                '--angle-step', '0.2',
                '--mse-weight', '5',
                '--description', f'selfcorr_t{thresh}_w{weight}'
            ]

            try:
                result = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=600)
                output = result.stdout + result.stderr

                # Parse results from output
                for line in output.split('\n'):
                    for well in wells:
                        if well in line and ('opt=' in line or 'error=' in line):
                            print(f"  {line.strip()}")
                            # Try to extract error
                            if 'opt=' in line:
                                parts = line.split('opt=')[1].split()[0]
                                try:
                                    err = float(parts.replace('m', ''))
                                    results.append({
                                        'well': well,
                                        'threshold': thresh,
                                        'weight': weight,
                                        'error': err
                                    })
                                except:
                                    pass
            except subprocess.TimeoutExpired:
                print(f"  TIMEOUT")
            except Exception as e:
                print(f"  ERROR: {e}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    for well in wells:
        print(f"\n{well}:")
        well_results = [r for r in results if r['well'] == well]
        well_results.sort(key=lambda x: abs(x['error']))
        for r in well_results:
            print(f"  thresh={r['threshold']:>2}, weight={r['weight']:.1f} -> error={r['error']:+.2f}m")


if __name__ == '__main__':
    main()

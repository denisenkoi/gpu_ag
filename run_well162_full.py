#!/usr/bin/env python3
"""Run full Well162 slicing and save results."""

import sys
import os

# Setup paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from true_gpu_slicer import main
import argparse

if __name__ == '__main__':
    sys.argv = ['true_gpu_slicer.py', '--well', 'Well162~EGFDL', '--slice-step', '30']
    main()

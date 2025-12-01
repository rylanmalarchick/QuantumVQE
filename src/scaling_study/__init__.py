"""
VQE Scaling Study - CPU vs GPU Performance Analysis

This module benchmarks VQE performance across different qubit counts
to find the crossover point where GPU outperforms CPU+JIT.

Hardware Target:
    - CPU: 2x AMD EPYC 9654 (192 cores), 1.5TB RAM
    - GPU: 4x NVIDIA H100 PCIe (81GB each)
"""

__version__ = "1.0.0"

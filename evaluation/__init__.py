"""Evaluation and benchmarking package for ProGlint Footfall Intelligence."""

from evaluation.benchmark import run_benchmark

# Compatibility aliases
BenchmarkSuite = run_benchmark
run_comparative_benchmark = run_benchmark

__all__ = ["run_benchmark", "BenchmarkSuite", "run_comparative_benchmark"]

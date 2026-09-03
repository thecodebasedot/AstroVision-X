"""Checks against tools astronomers already trust."""

from .benchmark import (
    BenchmarkResult,
    available_tools,
    benchmark_field,
    run_photutils,
    run_sep,
)

__all__ = ["BenchmarkResult", "available_tools", "benchmark_field",
           "run_photutils", "run_sep"]

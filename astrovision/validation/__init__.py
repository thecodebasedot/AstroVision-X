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
from .morphology_benchmark import (MorphologyBenchmark, benchmark_morphology,  # noqa: E402
                                   compare_morphology, statmorph_available)

__all__ += ["MorphologyBenchmark", "benchmark_morphology", "compare_morphology",
            "statmorph_available"]

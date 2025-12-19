"""Performance benchmarks for the MPPI controller implementations.

Usage:
    # Run benchmarks (default console output)
    pytest tests/benchmarks/ --benchmark-enable

    # Generate JSON and pretty report
    pytest tests/benchmarks/ --benchmark-enable --benchmark-json=benchmark.json
    python tests/benchmarks/report.py benchmark.json

    # Customize output columns and sorting
    pytest tests/benchmarks/ --benchmark-enable \
        --benchmark-columns=min,mean,max,ops \
        --benchmark-sort=mean

    # Save baseline for comparison
    pytest tests/benchmarks/ --benchmark-enable --benchmark-autosave
    # ...make changes...
    pytest tests/benchmarks/ --benchmark-enable --benchmark-compare=0001
"""

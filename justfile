# Run benchmarks and save to JSON
bench:
    uv run pytest -m benchmark --benchmark-json=benchmark.json

bench-baseline:
    uv run pytest -m benchmark --benchmark-autosave

# Generate report for benchmark results
bench-report *args:
    uv run python tests/benchmarks/report.py {{ args }}

# Run benchmarks then generate report
bench-and-report *args: bench
    uv run python tests/benchmarks/report.py show benchmark.json {{ args }}

# Run benchmarks with Codspeed integration
bench-validate:
  uv run pytest --codspeed -m benchmark

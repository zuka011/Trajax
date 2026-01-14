set shell := ["bash", "-c"]
set windows-shell := ["cmd", "/c"]

default_modules := ". visualizer"

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

[unix]
check modules=default_modules:
    for dir in {{modules}}; do \
        echo "=== Checking $dir ===" && \
        pushd "$dir" && \
        source .venv/bin/activate && \
        ruff check --fix && \
        ruff format && \
        pyright && \
        pytest && \
        popd; \
    done

[windows]
check modules=default_modules:
    for %d in ({{modules}}) do @echo === Checking %d === && pushd %d && .venv\Scripts\activate.bat && ruff check --fix && ruff format && pyright && pytest && popd

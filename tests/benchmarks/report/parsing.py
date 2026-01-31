from typing import Any, Final

from .models import Benchmark, BenchmarkData, ExportData, ParsedBenchmark


BACKENDS: Final = {"NumPy", "JAX", "JAX (CPU)", "JAX (GPU)"}


def implementation(benchmark: Benchmark) -> str:
    if benchmark.params and benchmark.params.id:
        return benchmark.params.id

    if benchmark.param:
        parts = benchmark.param.split("-")
        for part in parts:
            if part in BACKENDS:
                return part

        return parts[0]

    if "[" in benchmark.name:
        return benchmark.name.split("[")[1].rstrip("]").split("-")[0]

    return benchmark.name


def group(benchmark: Benchmark) -> str:
    return benchmark.group or "default"


def groups(benchmarks: list[Benchmark]) -> dict[str, list[Benchmark]]:
    result: dict[str, list[Benchmark]] = {}

    for benchmark in benchmarks:
        result.setdefault(group(benchmark), []).append(benchmark)

    return result


def parameters(benchmark: Benchmark) -> dict[str, Any]:
    params: dict[str, Any] = {}

    if not benchmark.param:
        return params

    parts = benchmark.param.split("-")
    for part in parts:
        if "=" in part:
            key, value = part.split("=", 1)
            params[key] = parse_value(value)

        elif part not in BACKENDS:
            params.setdefault("extra", []).append(part)

    if extra := params.pop("extra", []):
        params["label"] = " ".join(extra)

    return params


def parse_value(value: str) -> int | float | str:
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def parsed_benchmark(benchmark: Benchmark) -> ParsedBenchmark:
    return ParsedBenchmark(
        name=benchmark.name,
        group=group(benchmark),
        implementation=implementation(benchmark),
        parameters=parameters(benchmark),
        mean=benchmark.stats.mean,
        min=benchmark.stats.min,
        max=benchmark.stats.max,
        stddev=benchmark.stats.stddev,
        median=benchmark.stats.median,
        rounds=benchmark.stats.rounds,
        ops=benchmark.stats.ops,
        iqr=benchmark.stats.iqr,
    )


def export_data(data: BenchmarkData) -> ExportData:
    parsed = [parsed_benchmark(b) for b in data.benchmarks]
    summary = build_summary(data.benchmarks)
    machine_dict = build_machine_info(data)
    commit_dict = build_commit_info(data)

    return ExportData(
        machine_info=machine_dict,
        commit_info=commit_dict,
        benchmarks=parsed,
        summary=summary,
    )


def build_summary(benchmarks: list[Benchmark]) -> dict[str, Any]:
    times: dict[str, list[float]] = {}

    for benchmark in benchmarks:
        times.setdefault(implementation(benchmark), []).append(benchmark.stats.mean)

    averages = {k: sum(v) / len(v) for k, v in times.items()}
    sorted_implementations = sorted(averages.items(), key=lambda x: x[1])

    fastest_name, fastest_avg = (
        sorted_implementations[0] if sorted_implementations else ("", 0.0)
    )

    return {
        "total_benchmarks": len(benchmarks),
        "implementations": list(times.keys()),
        "averages": {
            impl: {
                "mean_seconds": avg,
                "relative_to_fastest": 1.0
                if impl == fastest_name
                else avg / fastest_avg,
            }
            for impl, avg in averages.items()
        },
        "fastest": fastest_name,
    }


def build_machine_info(data: BenchmarkData) -> dict[str, Any]:
    if not data.machine_info:
        return {}

    return {
        "python_version": data.machine_info.python_version,
        "platform": data.machine_info.platform,
        "cpu": data.machine_info.cpu,
        "gpu": data.machine_info.gpu,
    }


def build_commit_info(data: BenchmarkData) -> dict[str, Any]:
    if not data.commit_info:
        return {}

    return {"id": data.commit_info.id}

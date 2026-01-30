from rich.text import Text


def time(seconds: float) -> str:
    if seconds < 1e-6:
        return f"{seconds * 1e9:.2f} ns"
    if seconds < 1e-3:
        return f"{seconds * 1e6:.2f} µs"
    if seconds < 1:
        return f"{seconds * 1e3:.2f} ms"
    return f"{seconds:.2f} s"


def operations(ops: float) -> str:
    if ops >= 1_000_000:
        return f"{ops / 1_000_000:.1f}M"
    if ops >= 1_000:
        return f"{ops / 1_000:.1f}K"
    return f"{ops:.1f}"


def comparison(value: float, baseline: float) -> Text:
    if baseline <= 0:
        return Text("N/A", style="dim")

    ratio = value / baseline
    if ratio <= 1.0:
        return Text(f"✓ {ratio:.2f}x", style="green")
    if ratio <= 1.2:
        return Text(f"~ {ratio:.2f}x", style="yellow")
    return Text(f"✗ {ratio:.2f}x", style="red")

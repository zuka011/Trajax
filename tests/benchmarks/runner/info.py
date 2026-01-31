import re
import asyncio
from typing import Any, Mapping


async def raw_gpu_info() -> bytes | None:
    proc = await asyncio.create_subprocess_exec(
        "nvidia-smi",
        "--query-gpu="
        "index,"
        "name,"
        "driver_version,"
        "pci.bus_id,"
        "compute_cap,"
        "memory.total,"
        "memory.free,"
        "clocks.max.sm,"
        "clocks.max.memory,"
        "power.limit,"
        "power.max_limit,"
        "temperature.gpu,"
        "fan.speed,"
        "pstate",
        "--format=csv,noheader,nounits,",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, _ = await proc.communicate()

    if proc.returncode != 0:
        return

    return stdout


async def raw_cuda_version() -> bytes | None:
    proc = await asyncio.create_subprocess_exec(
        "nvidia-smi", stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    stdout, _ = await proc.communicate()

    if proc.returncode != 0:
        return

    return stdout


def format_cuda_version(raw: bytes | None) -> Mapping[str, Any]:
    if raw is not None and (
        match := re.search(r"CUDA Version:\s*(\d+\.\d+)", raw.decode("utf-8"))
    ):
        return {"version": match.group(1)}

    return {}


def format_gpu_info(raw: bytes | None) -> Mapping[str, Any]:
    if raw is None:
        return {}

    try:

        def parse_int(v: str) -> int | None:
            return int(v) if v not in ("[N/A]", "[Not Supported]", "") else None

        def parse_float(v: str) -> float | None:
            return float(v) if v not in ("[N/A]", "[Not Supported]", "") else None

        gpus = []
        for line in raw.decode("utf-8").strip().split("\n"):
            p = [x.strip() for x in line.split(",")]
            gpus.append(
                {
                    "index": parse_int(p[0]),
                    "name": p[1],
                    "driver_version": p[2],
                    "pci_bus_id": p[3],
                    "compute_capability": p[4],
                    "memory": {
                        "total_mb": parse_int(p[5]),
                        "free_mb": parse_int(p[6]),
                    },
                    "clocks_max": {
                        "sm_mhz": parse_int(p[7]),
                        "memory_mhz": parse_int(p[8]),
                    },
                    "power": {
                        "limit_w": parse_float(p[9]),
                        "max_limit_w": parse_float(p[10]),
                    },
                    "temperature_c": parse_int(p[11]),
                    "fan_speed_pct": parse_int(p[12]),
                    "performance_state": p[13],
                }
            )

        return {"count": len(gpus), "devices": gpus}
    except (IndexError, ValueError):
        return {}


async def gpu_info() -> Mapping[str, Any]:
    try:
        raw_info, raw_version = await asyncio.gather(raw_gpu_info(), raw_cuda_version())

        return {**format_gpu_info(raw_info), **format_cuda_version(raw_version)}
    except FileNotFoundError:
        return {}

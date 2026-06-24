#!/usr/bin/env python3

import pandas as pd
from pathlib import Path

RESULTS = Path(__file__).parent.parent / "results"
RAW_CSV = RESULTS / "raw.csv"
OUT_CSV = RESULTS / "summary.csv"


def main() -> None:
    files = sorted(RESULTS.glob("raw_*.csv"))
    if not files:
        raise SystemExit(f"No raw_*.csv files found in {RESULTS}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    print(f"Loaded {len(files)} file(s): {[f.name for f in files]}")

    numeric = [
        "load_time_s", "ttft_ms", "gen_tps_burst", "gen_tps_sustained", "total_time_s",
        "quality_score",
        "peak_cpu_pct", "avg_cpu_pct",
        "peak_gpu_pct", "avg_gpu_pct",
        "peak_mem_gb", "avg_mem_gb",
        "peak_cpu_temp_c", "avg_cpu_temp_c",
        "peak_gpu_temp_c", "avg_gpu_temp_c",
        "peak_power_w", "avg_power_w",
    ]
    for col in numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    summary = (
        df.groupby(["runner", "test_id", "task"])[numeric]
        .agg(["mean", "std", "min", "max"])
        .round(2)
    )
    summary.to_csv(OUT_CSV)
    print(summary.to_string())
    print(f"\nSaved to {OUT_CSV}")


if __name__ == "__main__":
    main()

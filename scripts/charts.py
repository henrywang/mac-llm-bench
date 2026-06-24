#!/usr/bin/env python3

import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import pandas as pd
from pathlib import Path

matplotlib.use("Agg")

RESULTS = Path(__file__).parent.parent / "results"
CHARTS = RESULTS / "charts"

COLORS = {"ollama": "#4C72B0", "omlx": "#DD8452"}


def _bar(ax: plt.Axes, df: pd.DataFrame, metric: str, title: str, ylabel: str) -> None:
    runners = df["runner"].unique()
    x = np.arange(len(df["task"].unique()))
    width = 0.35
    tasks = df["task"].unique()
    for i, runner in enumerate(runners):
        vals = [df[(df["runner"] == runner) & (df["task"] == t)][metric].mean() for t in tasks]
        ax.bar(x + i * width, vals, width, label=runner, color=COLORS.get(runner))
    ax.set_xticks(x + width / 2)
    ax.set_xticklabels(tasks, rotation=15)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.legend()


def _save(fig: plt.Figure, name: str) -> None:
    CHARTS.mkdir(parents=True, exist_ok=True)
    path = CHARTS / name
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved {path}")


def main() -> None:
    files = sorted(RESULTS.glob("raw_*.csv"))
    if not files:
        raise SystemExit(f"No raw_*.csv files found in {RESULTS}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    numeric = [
        "ttft_ms", "gen_tps_burst", "gen_tps_sustained", "total_time_s", "load_time_s",
        "peak_cpu_pct", "avg_cpu_pct", "peak_gpu_pct", "avg_gpu_pct",
        "peak_mem_gb", "avg_mem_gb",
        "peak_cpu_temp_c", "avg_cpu_temp_c", "peak_gpu_temp_c", "avg_gpu_temp_c",
        "peak_power_w", "avg_power_w",
    ]
    for col in numeric:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    runners = list(df["runner"].unique())
    width = 0.35

    # tps.png — burst vs sustained
    fig, ax = plt.subplots()
    t2 = df[df["test_id"] == "t2"]
    x = np.arange(len(runners))
    ax.bar(x - width / 2, [t2[t2["runner"] == r]["gen_tps_burst"].mean() for r in runners],
           width, label="burst", color="#4C72B0")
    ax.bar(x + width / 2, [t2[t2["runner"] == r]["gen_tps_sustained"].mean() for r in runners],
           width, label="sustained", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(runners)
    ax.set_title("Token/s: burst vs sustained")
    ax.set_ylabel("tok/s")
    ax.legend()
    _save(fig, "tps.png")

    # ttft.png — TTFT by task
    fig, ax = plt.subplots()
    ttft_df = df[df["ttft_ms"].notna()]
    _bar(ax, ttft_df, "ttft_ms", "Time-to-first-token by task", "ms")
    _save(fig, "ttft.png")

    # load_time.png
    fig, ax = plt.subplots()
    t1 = df[df["test_id"] == "t1"]
    x = np.arange(len(runners))
    vals = [t1[t1["runner"] == r]["load_time_s"].mean() for r in runners]
    bars = ax.bar(x, vals, color=[COLORS.get(r) for r in runners])
    ax.set_xticks(x)
    ax.set_xticklabels(runners)
    ax.set_title("Cold-start load time")
    ax.set_ylabel("seconds")
    _save(fig, "load_time.png")

    # cpu_util.png
    fig, ax = plt.subplots()
    x = np.arange(len(runners))
    ax.bar(x - width / 2, [df[df["runner"] == r]["peak_cpu_pct"].mean() for r in runners],
           width, label="peak", color="#4C72B0")
    ax.bar(x + width / 2, [df[df["runner"] == r]["avg_cpu_pct"].mean() for r in runners],
           width, label="avg", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(runners)
    ax.set_title("CPU utilization")
    ax.set_ylabel("%")
    ax.legend()
    _save(fig, "cpu_util.png")

    # gpu_util.png
    fig, ax = plt.subplots()
    ax.bar(x - width / 2, [df[df["runner"] == r]["peak_gpu_pct"].mean() for r in runners],
           width, label="peak", color="#4C72B0")
    ax.bar(x + width / 2, [df[df["runner"] == r]["avg_gpu_pct"].mean() for r in runners],
           width, label="avg", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(runners)
    ax.set_title("GPU utilization")
    ax.set_ylabel("%")
    ax.legend()
    _save(fig, "gpu_util.png")

    # memory.png
    fig, ax = plt.subplots()
    ax.bar(x - width / 2, [df[df["runner"] == r]["peak_mem_gb"].mean() for r in runners],
           width, label="peak", color="#4C72B0")
    ax.bar(x + width / 2, [df[df["runner"] == r]["avg_mem_gb"].mean() for r in runners],
           width, label="avg", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(runners)
    ax.set_title("RAM usage")
    ax.set_ylabel("GB")
    ax.legend()
    _save(fig, "memory.png")

    # temperature.png
    fig, ax = plt.subplots()
    x = np.arange(len(runners))
    n = len(runners)
    w = 0.2
    ax.bar(x - w * 1.5, [df[df["runner"] == r]["peak_cpu_temp_c"].mean() for r in runners],
           w, label="CPU peak °C", color="#4C72B0")
    ax.bar(x - w * 0.5, [df[df["runner"] == r]["avg_cpu_temp_c"].mean() for r in runners],
           w, label="CPU avg °C", color="#7BA3D4")
    ax.bar(x + w * 0.5, [df[df["runner"] == r]["peak_gpu_temp_c"].mean() for r in runners],
           w, label="GPU peak °C", color="#DD8452")
    ax.bar(x + w * 1.5, [df[df["runner"] == r]["avg_gpu_temp_c"].mean() for r in runners],
           w, label="GPU avg °C", color="#E8AA82")
    ax.set_xticks(x)
    ax.set_xticklabels(runners)
    ax.set_title("Temperature")
    ax.set_ylabel("°C")
    ax.legend()
    _save(fig, "temperature.png")

    # power.png — avg power over all runs per runner (bar, since we don't store time-series)
    fig, ax = plt.subplots()
    ax.bar(x - width / 2, [df[df["runner"] == r]["peak_power_w"].mean() for r in runners],
           width, label="peak", color="#4C72B0")
    ax.bar(x + width / 2, [df[df["runner"] == r]["avg_power_w"].mean() for r in runners],
           width, label="avg", color="#DD8452")
    ax.set_xticks(x)
    ax.set_xticklabels(runners)
    ax.set_title("Power draw")
    ax.set_ylabel("W")
    ax.legend()
    _save(fig, "power.png")

    # t4_quality.png + t4_time.png
    t4 = df[df["test_id"] == "t4"].copy()
    t4["quality_score"] = pd.to_numeric(t4["quality_score"], errors="coerce")
    t4["total_time_s"] = pd.to_numeric(t4["total_time_s"], errors="coerce")
    t4_tasks = sorted(t4["task"].unique())
    x4 = np.arange(len(t4_tasks))

    fig, ax = plt.subplots()
    for i, runner in enumerate(runners):
        vals = [t4[(t4["runner"] == runner) & (t4["task"] == t)]["quality_score"].mean()
                for t in t4_tasks]
        ax.bar(x4 + i * width, vals, width, label=runner, color=COLORS.get(runner))
    ax.set_xticks(x4 + width / 2)
    ax.set_xticklabels(t4_tasks, rotation=15)
    ax.set_ylim(0, 100)
    ax.set_title("T4 Coding quality score (0–100)")
    ax.set_ylabel("score")
    ax.legend()
    _save(fig, "t4_quality.png")

    fig, ax = plt.subplots()
    for i, runner in enumerate(runners):
        vals = [t4[(t4["runner"] == runner) & (t4["task"] == t)]["total_time_s"].mean()
                for t in t4_tasks]
        bars = ax.bar(x4 + i * width, vals, width, label=runner, color=COLORS.get(runner))
        for bar, v in zip(bars, vals):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                        f"{v:.0f}s", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x4 + width / 2)
    ax.set_xticklabels(t4_tasks, rotation=15)
    ax.set_title("T4 Coding task completion time")
    ax.set_ylabel("seconds")
    ax.legend()
    _save(fig, "t4_time.png")

    # radar.png
    categories = ["tok/s", "TTFT (inv)", "RAM (inv)", "Temp (inv)", "Power (inv)", "Quality"]
    n_cats = len(categories)
    angles = [n / n_cats * 2 * np.pi for n in range(n_cats)] + [0]

    def normalize(vals: list[float], invert: bool = False) -> list[float]:
        mn, mx = min(vals), max(vals)
        if mx == mn:
            return [0.5] * len(vals)
        normed = [(v - mn) / (mx - mn) for v in vals]
        return [1 - n for n in normed] if invert else normed

    tps_vals = [df[df["runner"] == r]["gen_tps_sustained"].mean() for r in runners]
    ttft_vals = [df[df["runner"] == r]["ttft_ms"].mean() for r in runners]
    mem_vals = [df[df["runner"] == r]["avg_mem_gb"].mean() for r in runners]
    temp_vals = [df[df["runner"] == r]["avg_cpu_temp_c"].mean() for r in runners]
    pwr_vals = [df[df["runner"] == r]["avg_power_w"].mean() for r in runners]
    qual_vals = [df[df["runner"] == r]["quality_score"].mean() for r in runners]

    norm_tps = normalize(tps_vals)
    norm_ttft = normalize(ttft_vals, invert=True)
    norm_mem = normalize(mem_vals, invert=True)
    norm_temp = normalize(temp_vals, invert=True)
    norm_pwr = normalize(pwr_vals, invert=True)
    norm_qual = normalize(qual_vals)

    fig, ax = plt.subplots(subplot_kw={"polar": True})
    for i, runner in enumerate(runners):
        vals = [norm_tps[i], norm_ttft[i], norm_mem[i], norm_temp[i], norm_pwr[i], norm_qual[i]]
        vals += vals[:1]
        ax.plot(angles, vals, label=runner, color=list(COLORS.values())[i])
        ax.fill(angles, vals, alpha=0.15, color=list(COLORS.values())[i])
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_title("Overall comparison (higher = better)")
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    _save(fig, "radar.png")


if __name__ == "__main__":
    main()

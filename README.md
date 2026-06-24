# llm-runner-benchmark

Benchmark comparing **Ollama** vs **oMLX** for running Qwen3.6:35B-A3B locally on Apple Silicon.
Tests cover token throughput, latency, Python coding quality (via Pi agent), and resource usage (CPU, GPU, RAM, power).

See **[REPORT.md](REPORT.md)** for full results and charts.

## Hardware target

MacBook Air M5, 32GB unified memory. Results will vary on other configurations.

## Models

| Runner | Model |
|--------|-------|
| Ollama 0.30.10 | `qwen3.6:35b-a3b-nvfp4` (NVFP4) |
| oMLX 0.4.4 | `mlx-community/Qwen3.6-35B-A3B-4bit` (4-bit MLX) |

## Prerequisites

- macOS 15+, Apple Silicon
- [Ollama](https://ollama.com) — `brew install ollama`
- [oMLX](https://omlx.ai) — download and install the macOS app
- [just](https://github.com/casey/just) — `brew install just`
- [uv](https://github.com/astral-sh/uv) — `brew install uv`
- [Pi coding agent](https://github.com/earendil-works/pi) — for T4 coding tasks
- `sudo` access — required for `powermetrics`

## Setup

```bash
# 1. Create venv and install Python dependencies
just setup

# 2. Pull Ollama model (~22GB)
just pull-ollama

# 3. Load mlx-community/Qwen3.6-35B-A3B-4bit in oMLX via the menu bar app

# 4. Smoke test both runners
just test-ollama
just test-omlx
```

## Running the benchmark

```bash
just bench-ollama       # run all tests for Ollama
just bench-omlx         # run all tests for oMLX
just bench-test t2      # single test (t1/t2/t3/t4/t5)
```

T4 uses the Pi coding agent via the custom providers in `pi-extensions/`.
Results are written to `results/raw_ollama.csv` and `results/raw_omlx.csv`.

## Generate report

```bash
just report     # analyze CSVs → summary.csv + regenerate all charts
```

## Tests

| ID | What |
|----|------|
| T1 | Cold-start model load time |
| T2 | Raw token/s — burst and sustained |
| T3 | Time-to-first-token on a 4K-token input |
| T4 | Python coding tasks via Pi agent (automated) |
| T5 | Daily tasks — email summary, news digest, PR review |

Resource metrics (CPU %, GPU %, RAM, power draw) are captured for every run
via `powermetrics` and `psutil`.

## Baseline versioning

Baseline CSVs are stored in `results/baseline/` and never overwritten.
After updating a runner or model, archive results with:

```bash
just save-baseline <ollama_ver> <omlx_ver>
```

## Project structure

```
llm-runner-benchmark/
├── Justfile
├── prompts/              # frozen prompt files
├── pi-extensions/        # Pi agent provider configs for each runner
├── scripts/
│   ├── run_benchmark.py
│   ├── analyze.py
│   └── charts.py
└── results/
    ├── baseline/         # frozen baseline CSVs per version
    └── charts/           # generated PNG charts (embedded in REPORT.md)
```

## License

MIT

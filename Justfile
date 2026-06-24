ollama_model := "qwen3.6:35b-a3b-nvfp4"
omlx_model   := "Qwen3.6-35B-A3B-4bit"
omlx_port    := "8000"
omlx_key     := env_var_or_default("OMLX_API_KEY", "")

default:
    @just --list

setup:
    uv venv
    uv pip install pandas matplotlib requests psutil asitop

pull-ollama:
    ollama pull {{ollama_model}}

test-ollama:
    ollama run {{ollama_model}} /no_think "Say hello in one sentence"

test-omlx:
    curl -s http://localhost:{{omlx_port}}/v1/chat/completions \
      -H "Content-Type: application/json" \
      -H "Authorization: Bearer {{omlx_key}}" \
      -d '{"model":"{{omlx_model}}","messages":[{"role":"user","content":"Say hello"}],"max_tokens":20}' \
      | python3 -m json.tool

start-ollama:
    ollama serve

stop-ollama:
    pkill ollama || true

start-omlx:
    omlx start

stop-omlx:
    omlx stop

bench:
    uv run scripts/run_benchmark.py

bench-ollama:
    uv run scripts/run_benchmark.py --runner ollama

bench-omlx:
    uv run scripts/run_benchmark.py --runner omlx

bench-test test:
    uv run scripts/run_benchmark.py --test {{test}}

analyze:
    uv run scripts/analyze.py

charts:
    uv run scripts/charts.py

report: analyze charts

check-metrics:
    sudo powermetrics --samplers gpu_power,cpu_power,thermal -n 1

check-memory:
    memory_pressure

bench-t4-ollama:
    uv run scripts/run_benchmark.py --runner ollama --test t4

bench-t4-omlx:
    uv run scripts/run_benchmark.py --runner omlx --test t4

save-baseline ollama_ver omlx_ver:
    mkdir -p results/baseline
    cp results/raw_ollama.csv results/baseline/raw_ollama_{{ollama_ver}}.csv
    cp results/raw_omlx.csv   results/baseline/raw_omlx_{{omlx_ver}}.csv
    @echo "Saved baseline: ollama={{ollama_ver}} omlx={{omlx_ver}}"

clean:
    rm -f results/raw_*.csv results/summary.csv results/metrics_*.txt results/charts/*.png

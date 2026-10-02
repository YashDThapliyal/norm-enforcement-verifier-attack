#!/usr/bin/env bash
# Dedicated Ollama server for the experiments: 4 parallel slots, 8k context, q8 KV cache.
set -euo pipefail
BIN="${OLLAMA_BIN:-/Applications/Ollama.app/Contents/Resources/ollama}"
mkdir -p logs
OLLAMA_HOST=127.0.0.1:11435 OLLAMA_NUM_PARALLEL=4 OLLAMA_KEEP_ALIVE=24h OLLAMA_MAX_LOADED_MODELS=3 \
OLLAMA_FLASH_ATTENTION=1 OLLAMA_KV_CACHE_TYPE=q8_0 OLLAMA_CONTEXT_LENGTH=8192 \
nohup "$BIN" serve > logs/ollama.log 2>&1 &
echo "ollama on :11435 (pid $!)"

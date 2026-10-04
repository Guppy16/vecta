#!/bin/bash
# The talker's model server: Nemotron 3 Nano Omni (Q4) on llama.cpp HIP, 127.0.0.1:18090.
#
# Runs in the llamahip container (/work = ~/git/vecta/.scratch) with the build from
# .scratch/omni (upstream llama.cpp + the dynamic-resolution and mtmd patches in
# .scratch/omni-pipeline); logs to .scratch/nemotron-server.log. Two slots: one decides
# each turn, one transcribes it (talker.py).
# Images are capped at 256 tokens: the knee of the accuracy/latency curve
# (.scratch/nemotron-hillclimb/RESULTS.md section 4).
#
#   scripts/nemotron_server.sh          # start, wait until healthy
#   scripts/nemotron_server.sh stop
set -euo pipefail
MODELS=/work/omni/models/nemotron-omni-gguf
LLM=$MODELS/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-UD-Q4_K_XL.gguf
MMPROJ=$MODELS/mmproj-omni-dyn-F16.gguf
BIN=/work/omni/llama.cpp/build-hip/bin/llama-server
PORT=18090

if [ "${1:-start}" = stop ]; then
    docker exec llamahip pkill -f "[l]lama-server.*--port $PORT" || true
    exit 0
fi
# CKPT_AFTER_MTMD: keep a cache checkpoint after audio/images, so the next turn reuses the prefix
docker exec -d -e LD_LIBRARY_PATH=/opt/rocm/lib -e LLAMA_SERVER_CKPT_AFTER_MTMD=1 -w /work/omni llamahip \
    bash -c "exec $BIN -m $LLM --mmproj $MMPROJ -ngl 999 -c 32768 -np 2 -b 2048 -ub 1024 -fa on \
        --image-max-tokens 256 --jinja --reasoning-format none --host 127.0.0.1 --port $PORT \
        > /work/nemotron-server.log 2>&1"
for _ in $(seq 1 300); do
    curl -sf "http://127.0.0.1:$PORT/health" > /dev/null && { echo "ready on :$PORT"; exit 0; }
    sleep 1
done
echo "server did not come up"
docker exec llamahip tail -20 /work/nemotron-server.log
exit 1

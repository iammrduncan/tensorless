#!/bin/bash
# G2-like smoke test on one card: prepare, start llama-server at -c 32768 with the judge's flags, stream 64 greedy
# tokens on chat prompts, record nvidia-smi memory every 0.5 s.
#   g2_smoke.sh <engine dir> <pack dir> <work dir> [ctx]
set -u
ulimit -c 0
ENG=$1; PACK=$2; WK=$3; CTX=${4:-32768}
mkdir -p $WK/prep
M=$(python3 -E -s $ENG/tools/inkling-prepare/prepare.py $PACK $WK/prep | tail -1)
( while true; do nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits >> $WK/mem.txt 2>/dev/null; sleep 0.5; done ) &
MP=$!
$ENG/build/bin/llama-server -m $M -c $CTX -ngl 99 -lm none -fit off -fa auto -b 512 -ub 512 -ctk f16 -ctv f16 -t 8 -tb 8 -np 1 --port 18093 > $WK/server.log 2>&1 &
SP=$!
python3 $ENG/tests/inkling2/g2_client.py 18093 $SP
kill $SP 2>/dev/null; sleep 1; kill $MP
echo "peak MiB $(sort -n $WK/mem.txt | tail -1)"
grep -E "inkling2|buffer size|compute buffer|error|failed|GGML_ASSERT|ABORT" $WK/server.log | head -40
tail -25 $WK/server.log | cut -c1-300

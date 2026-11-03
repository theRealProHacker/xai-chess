#!/bin/bash
# Sequential SFT over tasks.py tasks: stage k trains on task k with replay of tasks 1..k-1 (train.py --replay),
# warm-started from stage k-1's last.pt; cosine LR decay on AdamW per stage. A finished stage only re-checks
# its reports, a killed one resumes. Extra args go to every stage (data, layers, LoRA, epochs, replay, ...).
#   ORDER="phase control attacks" RUN=oracle/runs/sft INIT=oracle/runs/x/last.pt oracle/curriculum.sh --lora 16
# ORDER entries may be task:E to give that stage E epochs (overrides --epochs).
# INIT (optional) is the first stage's --init-ckpt; its LoRA config must match the flags.
set -e
cd "$(dirname "$0")/.."
: "${ORDER:?set ORDER, e.g. ORDER=\"phase control attacks\"}" "${RUN:?set RUN, the output dir}"
prev=$INIT; seen=()
for te in $ORDER; do
    t=${te%%:*}; ep=; [ "$t" != "$te" ] && ep=${te#*:}       # task:E runs that stage for E epochs
    seen+=("$t")
    out=$RUN/$(printf %02d ${#seen[@]})_$t
    ${PY:-python} oracle/train.py --out "$out" --tasks "${seen[@]}" --sched cosine ${prev:+--init-ckpt "$prev"} "$@" ${ep:+--epochs $ep}
    prev=$out/last.pt
done

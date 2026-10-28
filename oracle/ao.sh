#!/bin/bash
# AO-style injection (--inject add: vectors added to the residual after decoder layer 1, scaled to its norm)
# through the same lineage and flags as the embed-injection runs, so the two compare stage for stage:
#   q08b -> q08b_lora -> g50k_sft -> curriculum (8 tasks), and multipv from g50k_sft
# Every train.py call resumes or only re-checks a finished run. Extra args go to every call.
#   RUN=oracle/runs/ao oracle/ao.sh
set -e
cd "$(dirname "$0")/.."
R=${RUN:-oracle/runs/ao}; export PY=${PY:-python}; X=(--inject add "$@")
$PY oracle/train.py --out $R/q08b --grad-ckpt "${X[@]}"
$PY oracle/train.py --out $R/q08b_lora --lora 16 --init-proj $R/q08b/epoch9.pt "${X[@]}"
$PY oracle/train.py --data oracle/data_g50k --out $R/g50k_sft --lora 16 --init-ckpt $R/q08b_lora/epoch9.pt \
    --lr 3e-4 --epochs 5 --eval-every 1000 --test-n 1000 "${X[@]}"
C=(--lora 16 --lr 3e-4 --epochs 2 --replay 0.2 --test-n 1000 "${X[@]}")
ORDER="best pv control attacks phase move_facts move_attacks move_board" RUN=$R/curr INIT=$R/g50k_sft/epoch4.pt \
    oracle/curriculum.sh --data oracle/data_g50k "${C[@]}"
ORDER=multipv RUN=$R/mpv INIT=$R/g50k_sft/epoch4.pt oracle/curriculum.sh --data oracle/data_g50k_mpv "${C[@]}"

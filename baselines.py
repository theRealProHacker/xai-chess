#!/usr/bin/env python3
"""Baselines for the corpus meta-eval -- how well do cheap NON-interventional scores separate
faithful from red-herring pin-citations, vs the rule-level causal `eff`?

Honest scope. BonaFide's 8 metrics are defined for LLM chain-of-thought: token-level
perturbations (Filler Tokens, Paraphrasing, Adding Mistakes, Early Answering) and
model-internal quantities (CC-SHAP, SCM, FUR), plus Simulatability. In this no-LLM,
structured-factor + engine-oracle setting they have no faithful analogue without the deferred
ChessSim / LLM-commentator track -- so "porting all 8" is not meaningful. What ports cleanly
are the design doc's named weak baselines, `random` and `saliency`, which is what "beat the
baselines" requires here.

  random   : uniform score (chance floor).
  saliency : the naive heuristic "the pin matters because the move targets valuable pinned
             material" = value(pinned piece) if the move attacks it after being played, else
             0. Non-causal -- it never asks what removing the pin does.

Run on the SAME records as the eff meta-eval (aligned to its dumped rows.json) so the
comparison, including a paired bootstrap, is apples-to-apples.
"""
import json
import argparse
import random
import chess

from pin_verify import neutralize_pin, PIECE_VALUE
from corpus_eval import auroc, auroc_ci, med


def saliency(board, move, pinned_sq):
    """value(pinned piece) if the mover attacks the pinned square after `move`, else 0."""
    val = PIECE_VALUE[board.piece_at(pinned_sq).piece_type]
    mover = board.turn
    nb = board.copy(stack=False)
    nb.push(move)
    return float(val) if nb.is_attacked_by(mover, pinned_sq) else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("records", help="the SAME labeled JSONL the eff meta-eval ran on")
    ap.add_argument("--rows", required=True, help="<records>.rows.json dumped by corpus_eval.py")
    args = ap.parse_args()

    eff = json.load(open(args.rows))  # [label, eff_wp, eff_logit, eff_wp_rule, eff_logit_rule] x N
    rng = random.Random(0)

    base = []  # (label, random, saliency), in the SAME order/skip-logic as corpus_eval -> rows
    for line in open(args.records):
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        board = chess.Board(rec["fen"])
        move = chess.Move.from_uci(rec["justified_move_uci"])
        if move not in board.legal_moves:                 # same quarantine as the verifier
            continue
        pinned_sq = chess.parse_square(rec["cited_factors"][0]["args"]["pinned"])
        sp, _ = neutralize_pin(board, pinned_sq)
        if sp is None:                                    # board-level neutralization failed
            continue
        label = 1 if str(rec.get("source", "")).startswith("faithful") else 0
        base.append((label, rng.random(), saliency(board, move, pinned_sq)))

    assert len(base) == len(eff), f"alignment broken: {len(base)} baselines vs {len(eff)} eff rows"
    assert all(b[0] == e[0] for b, e in zip(base, eff)), "labels misaligned"

    # Assemble all scorers on identical records: baselines + the eff metric (from rows).
    cols = {
        "random":          [b[1] for b in base],
        "saliency":        [b[2] for b in base],
        "board eff_wp":    [e[1] for e in eff],
        "rule  eff_wp":    [e[3] for e in eff],
        "rule  eff_logit": [e[4] for e in eff],
    }
    labels = [b[0] for b in base]

    print(f"# baselines vs eff | N={len(base)} (faithful={sum(labels)} red-herring={len(labels)-sum(labels)})\n")
    print(f"{'scorer':<18}{'AUROC':>7}{'  95% CI':>16}{'faithful':>11}{'herring':>10}")
    print("-" * 62)
    for name, scores in cols.items():
        pos = [s for s, l in zip(scores, labels) if l == 1]
        neg = [s for s, l in zip(scores, labels) if l == 0]
        a, lo, hi = auroc_ci(pos, neg)
        print(f"{name:<18}{a:>7.3f}   [{lo:.3f}, {hi:.3f}]{med(pos):>+11.3f}{med(neg):>+10.3f}")

    # Paired bootstrap: does rule eff beat the best baseline on the SAME records?
    rule = cols["rule  eff_wp"]
    for bname in ("saliency", "board eff_wp"):
        b = cols[bname]
        rng2 = random.Random(0)
        n = len(labels)
        diffs = []
        for _ in range(2000):
            idx = [rng2.randrange(n) for _ in range(n)]
            rp = [rule[i] for i in idx]; bp = [b[i] for i in idx]; ll = [labels[i] for i in idx]
            da = (auroc([x for x, y in zip(rp, ll) if y == 1], [x for x, y in zip(rp, ll) if y == 0])
                  - auroc([x for x, y in zip(bp, ll) if y == 1], [x for x, y in zip(bp, ll) if y == 0]))
            diffs.append(da)
        diffs.sort()
        pt = (auroc([x for x, y in zip(rule, labels) if y == 1], [x for x, y in zip(rule, labels) if y == 0])
              - auroc([x for x, y in zip(b, labels) if y == 1], [x for x, y in zip(b, labels) if y == 0]))
        print(f"\npaired Δ AUROC rule_eff - {bname}: {pt:+.3f}  "
              f"95% CI [{diffs[50]:+.3f}, {diffs[1949]:+.3f}]  P(Δ>0)={sum(d>0 for d in diffs)/2000:.3f}")


if __name__ == "__main__":
    main()

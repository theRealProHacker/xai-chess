#!/usr/bin/env python3
"""Meta-eval seed: run the pin verifier over a labeled corpus and report how well each
`eff` scale separates faithful (theme=pin, pin exploited) from red-herring (incidental
pin) records -- i.e. AUROC of the metric against the corpus's own ground-truth labels.

Labels are read from the record `source` prefix ("faithful"/"unfaithful"), which is what
lichess_ingest.py writes from the Lichess theme tags. AUROC = 0.5 is chance; the headline
question is whether eff_logit (de-saturated) beats eff_wp, and whether either clears the
field's 0.70 reference bar (BonaFide)."""
import json
import random
import argparse
import logging
import statistics
import chess
import chess.engine

logging.getLogger("chess.engine").setLevel(logging.CRITICAL)
from pin_verify import verify_record


def auroc(pos, neg):
    """P(score(faithful) > score(red-herring)) with ties at 0.5 (Mann-Whitney U / AUROC)."""
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def auroc_ci(pos, neg, b=2000, seed=0):
    """Bootstrap 95% CI for AUROC (resample both classes with replacement). Deterministic
    via a fixed seed. Returns (point, lo, hi)."""
    if not pos or not neg:
        return float("nan"), float("nan"), float("nan")
    rng = random.Random(seed)
    boots = []
    for _ in range(b):
        rp = [rng.choice(pos) for _ in pos]
        rn = [rng.choice(neg) for _ in neg]
        boots.append(auroc(rp, rn))
    boots.sort()
    return auroc(pos, neg), boots[int(0.025 * b)], boots[int(0.975 * b)]


def med(xs):
    return statistics.median(xs) if xs else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("records", help="labeled JSONL (source starts with faithful/unfaithful)")
    ap.add_argument("--engine", required=True)
    ap.add_argument("--nodes", type=int, default=400000)
    ap.add_argument("--threshold", type=float, default=0.10)
    args = ap.parse_args()

    engine = chess.engine.SimpleEngine.popen_uci(args.engine)
    engine.configure({"Threads": 1, "Hash": 64})
    srl = "MaskPinner" in engine.options
    limit = chess.engine.Limit(nodes=args.nodes)

    rows = []  # (label, eff_wp, eff_logit, eff_wp_rule, eff_logit_rule)
    quarantined = 0
    with open(args.records) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            label = 1 if str(rec.get("source", "")).startswith("faithful") else 0
            r = verify_record(engine, rec, limit, args.threshold, srl)
            scored = [fo for fo in r.get("factors", []) if fo.get("status") == "scored"]
            if not scored:
                quarantined += 1
                continue
            fo = scored[0]
            rows.append((label, fo["eff_wp"], fo["eff_logit"],
                         fo.get("eff_wp_rule"), fo.get("eff_logit_rule")))
    engine.quit()

    nf = sum(r[0] == 1 for r in rows)
    nh = sum(r[0] == 0 for r in rows)
    print(f"# corpus meta-eval | engine={args.engine.split('/')[-1]} nodes={args.nodes} "
          f"| faithful={nf} red-herring={nh} quarantined={quarantined}\n")
    print(f"{'scale':<18}{'AUROC':>7}{'  95% CI':>16}{'faithful':>11}{'herring':>10}")
    print("-" * 62)
    for name, idx in [("board eff_wp", 1), ("board eff_logit", 2),
                      ("rule  eff_wp", 3), ("rule  eff_logit", 4)]:
        pos = [r[idx] for r in rows if r[0] == 1 and r[idx] is not None]
        neg = [r[idx] for r in rows if r[0] == 0 and r[idx] is not None]
        a, lo, hi = auroc_ci(pos, neg)
        print(f"{name:<18}{a:>7.3f}   [{lo:.3f}, {hi:.3f}]{med(pos):>+11.3f}{med(neg):>+10.3f}")

    # Paired test: board and rule eff are measured on the SAME positions, so the right
    # "does the surface choice matter" statistic is the difference in AUROC with a PAIRED
    # bootstrap (resample records, recompute both AUROCs on that resample). This removes the
    # shared position-to-position variance and is far more powerful than comparing the two
    # overlapping marginal CIs above.
    paired = [r for r in rows if r[1] is not None and r[3] is not None]

    def paired_diff(idx_a, idx_b, b=2000, seed=0):
        rng = random.Random(seed)
        n = len(paired)
        diffs = []
        for _ in range(b):
            s = [paired[rng.randrange(n)] for _ in range(n)]
            d = (auroc([r[idx_a] for r in s if r[0] == 1], [r[idx_a] for r in s if r[0] == 0])
                 - auroc([r[idx_b] for r in s if r[0] == 1], [r[idx_b] for r in s if r[0] == 0]))
            diffs.append(d)
        diffs.sort()
        point = (auroc([r[idx_a] for r in paired if r[0] == 1], [r[idx_a] for r in paired if r[0] == 0])
                 - auroc([r[idx_b] for r in paired if r[0] == 1], [r[idx_b] for r in paired if r[0] == 0]))
        return point, diffs[int(0.025 * b)], diffs[int(0.975 * b)], sum(d > 0 for d in diffs) / b

    print()
    for label, ia, ib in [("eff_wp", 3, 1), ("eff_logit", 4, 2)]:
        pt, lo, hi, fp = paired_diff(ia, ib)
        print(f"paired Δ AUROC rule-board ({label}): {pt:+.3f}  95% CI [{lo:+.3f}, {hi:+.3f}]  P(Δ>0)={fp:.3f}")

    # Persist per-record scores so the analysis can be redone without re-running the engine.
    with open(args.records + ".rows.json", "w") as f:
        json.dump(rows, f)


if __name__ == "__main__":
    main()

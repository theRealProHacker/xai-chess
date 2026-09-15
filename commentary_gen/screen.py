"""Screen candidates on a fixed dev minibatch, then pack judge batches for each.

    python screen.py proposals_r1.json --demos D0 --n 40
Creates one candidate per (instruction, demo set) and runs it on the same n dev ids.
"""
import argparse, json, random, subprocess, sys
from pathlib import Path

HERE = Path(__file__).parent
PY = sys.executable


def minibatch(n, seed=0):
    ids = [json.loads(l)["id"] for l in open(HERE / "data/dev.jsonl")]
    random.Random(seed).shuffle(ids)
    p = HERE / "runs" / f"minibatch_{n}_{seed}.txt"
    p.write_text("\n".join(sorted(ids[:n])))
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("proposals")
    ap.add_argument("--demos", nargs="*", default=[""], help="demo set names from candidates/demo_sets.json; '' = none")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch", type=int, default=20)
    a = ap.parse_args()
    sets = json.load(open(HERE / "candidates/demo_sets.json"))
    props = json.load(open(HERE / "candidates" / a.proposals))
    ids = minibatch(a.n, a.seed)
    names = []
    for p in props:
        for d in a.demos:
            cand = {"name": p["name"] + (f"_{d}" if d else ""), "instruction": p["instruction"],
                    "demos": sets[d] if d else [], "rationale": p.get("rationale", "")}
            path = HERE / "candidates" / f"{cand['name']}.json"
            json.dump(cand, open(path, "w"), indent=1)
            subprocess.run([PY, HERE / "program.py", path, "--split", "dev", "--ids", ids, "--workers", "6"], check=True)
            subprocess.run([PY, HERE / "metric.py", cand["name"], "--pack", "--batch", str(a.batch)], check=True,
                           stdout=subprocess.DEVNULL)
            names.append(cand["name"])
    print("\n".join(names))


if __name__ == "__main__":
    main()

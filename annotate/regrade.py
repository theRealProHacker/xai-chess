"""Corrects the grade of tries whose only error is a stale Rated grade, after the cases were regraded.

  python annotate/regrade.py CASE...

In each case's round-1 final try, Rated(old) becomes the case's grade; what was written stays in
`dsl_written`, and the try is checked again. Round-2 tries of these cases are removed: they only regraded.
"""
import json, re, subprocess, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
ATTEMPTS = HERE / "data/attempts.jsonl"
VERIFIER = HERE.parent / "dsl_verifier/target/release/dsl_verifier"
GRADE = {"Blunder": "Blunder", "Mistake": "Mistake", "Inaccuracy": "Dubious", "Good": "Good"}

cases = {c["id"]: c for c in map(json.loads, (HERE / "data/cases.jsonl").open())}
targets = set(sys.argv[1:])
A = [json.loads(l) for l in ATTEMPTS.open()]
last = {a["case"]: i for i, a in enumerate(A) if a.get("round", 1) == 1}
out = []
for i, a in enumerate(A):
    if a["case"] not in targets:
        out.append(a); continue
    if a.get("round", 1) > 1:
        continue
    if i == last[a["case"]]:
        c = cases[a["case"]]
        m = re.match(r"\s*Because\(\s*Rated\((\w+)\)", a["dsl"])
        dsl = a["dsl"][:m.start(1)] + GRADE[c["kind"]] + a["dsl"][m.end(1):]
        r = subprocess.run([str(VERIFIER), "check"], input=json.dumps({"fen": c["fen"], "mov": c["mov"], "dsl": dsl}),
                           capture_output=True, text=True, timeout=900)
        o = json.loads(r.stdout.strip().splitlines()[-1])
        a = {**a, "dsl_written": a.get("dsl_written", a["dsl"]), "dsl": dsl, "rendered": o.get("english"),
             "v": o.get("v", "error"), "why": o.get("why"), "claims": o.get("claims"), "regraded": True}
        print(f"{a['case']:<14} {m[1]} -> {GRADE[c['kind']]}: {a['v']}")
    out.append(a)
with ATTEMPTS.open("w") as f:
    for a in out:
        f.write(json.dumps(a) + "\n")
print(f"{len(A) - len(out)} round-2 tries removed")

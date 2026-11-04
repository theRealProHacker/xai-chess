"""The agent run's tool: shows a case and checks and records each try in annotate/data/attempts.jsonl.

  python3 annotate/attempt.py show CASE
  python3 annotate/attempt.py try CASE --english "what you mean, in words" --dsl 'Because(...)'

A case is done when a try holds or is unknown, or after 5 tries.
"""
import argparse, json, re, subprocess, sys, time
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / ".pylibs"))
import chess

CASES = HERE / "data/cases.jsonl"
ATTEMPTS = HERE / "data/attempts.jsonl"
VERIFIER = HERE.parent / "dsl_verifier/target/release/dsl_verifier"
TRIES = 5


def case(cid):
    for c in map(json.loads, CASES.open()):
        if c["id"] == cid:
            return c
    sys.exit(f"no case {cid}")


def tries(cid):
    return [a for a in map(json.loads, ATTEMPTS.open()) if a["case"] == cid] if ATTEMPTS.exists() else []


def num(ply):
    return f"{ply // 2 + 1}{'...' if ply % 2 else '.'}"


def line(pv, ply):
    return " ".join((num(ply + i) if i == 0 or (ply + i) % 2 == 0 else "") + m for i, m in enumerate(pv))


def ev(e):
    return f"mate in {e['mate']}" if e.get("mate") else "mated" if e.get("mate") == 0 else f"{e['cp'] / 100:+.2f}"


def hint(err, src):
    """The same plain-language hints as the annotator."""
    if '"' in src:
        return 'Write moves and squares without quotes: Qxf2+, not "Qxf2+"'
    if err == "unexpected end":
        stack = []
        for ch in src:
            if ch in "([{":
                stack.append({"(": ")", "[": "]", "{": "}"}[ch])
            elif ch in ")]}" and stack and stack[-1] == ch:
                stack.pop()
        if stack:
            return f"Missing {''.join(reversed(stack))} at the end"
    if m := re.fullmatch(r'invalid type: string "(.*)"', err):
        return f"{m[1]} goes in a list here: [{m[1]}]"
    return None


def done(ts):
    return bool(ts) and (ts[-1]["v"] in ("holds", "unknown") or len(ts) >= TRIES)


def show(c):
    b = chess.Board(c["fen"])
    side = "White" if b.turn else "Black"
    print(f"Case {c['id']}  ({c['white']} {c['white_elo']} - {c['black']} {c['black_elo']})")
    rows = str(b).split("\n")
    print("\n".join(f"{8 - i} {r}" for i, r in enumerate(rows)) + "\n  a b c d e f g h  (uppercase White)")
    print(f"FEN {c['fen']}")
    print(f"{side} to move plays {num(c['ply'])}{c['mov']}: a {c['kind'].lower()} (Lichess grading; "
          f"loses {c['loss']:.2f} winning chances).")
    print(f"Eval (White's view) before {ev(c['before'])}, after {ev(c['after'])}")
    print(f"Best  {line(c['before']['pv'], c['ply'])}")
    print(f"Then  {line(c['after']['pv'], c['ply'] + 1)}")
    ts = tries(c["id"])
    for t in ts:
        print(f"\nTry {t['n']}: {t['dsl']}\n  -> {t['v']}: {t.get('why') or t.get('error') or ''}")
    if done(ts):
        print("\nThis case is done.")
    else:
        print(f"\n{TRIES - len(ts)} tries left.")


def attempt(c, english, dsl, note):
    ts = tries(c["id"])
    if done(ts):
        sys.exit("This case is done; no more tries.")
    r = subprocess.run([str(VERIFIER), "check"], input=json.dumps({"fen": c["fen"], "mov": c["mov"], "dsl": dsl}),
                       capture_output=True, text=True, timeout=600)
    out = json.loads(r.stdout.strip().splitlines()[-1]) if r.stdout.strip() else {"error": r.stderr.strip()[-500:]}
    rec = {"case": c["id"], "n": len(ts) + 1, "english_intent": english, "dsl": dsl, "note": note,
           "rendered": out.get("english"), "v": out.get("v", "error"), "why": out.get("why"),
           "claims": out.get("claims"), "error": out.get("error"),
           "hint": hint(out["error"], dsl) if "error" in out else None,
           "at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    with ATTEMPTS.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"Try {rec['n']}/{TRIES}: {rec['v'].upper()}")
    if rec["error"]:
        print(f"Parse error: {rec['error']}" + (f"\nHint: {rec['hint']}" if rec["hint"] else ""))
    else:
        print(f"Renders as: {rec['rendered']}")
        if rec["why"]:
            print(f"Why: {rec['why']}")
        for cl in rec["claims"] or []:
            print(f"  {cl['name']:<12} {cl['v']:<8} {cl['why']}")
    ts.append(rec)
    print("Case done." if done(ts) else f"{TRIES - len(ts)} tries left.")


ap = argparse.ArgumentParser()
sub = ap.add_subparsers(dest="cmd", required=True)
s = sub.add_parser("show"); s.add_argument("case")
t = sub.add_parser("try"); t.add_argument("case")
t.add_argument("--english", required=True, help="what the reason means, in words")
t.add_argument("--dsl", required=True)
t.add_argument("--note", default="", help="what you changed from the last try and why")
a = ap.parse_args()
if a.cmd == "show":
    show(case(a.case))
else:
    attempt(case(a.case), a.english, a.dsl, a.note)

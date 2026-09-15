"""Bottom-up coding of the free-text judge notes from r4 (JUDGE2.md).

Categories were derived by reading all 300 notes first; the regexes only reproduce that reading so
the counts are reproducible. Multi-label. Output: judge_dev/coded.jsonl and a summary on stdout.
"""
import collections, json, re, statistics, sys
from pathlib import Path

RUN = Path(__file__).parent / "runs" / (sys.argv[1] if len(sys.argv) > 1 else "r4_best_lessons_think") / "judge_dev"

MISTAKES = {  # name: (regex over the mistakes text)
    "geometry: piece 'attacks/defends' a square it cannot reach, or through a blocker":
        r"not on (any|either|the)? ?(of )?(its |that |the )?(bishop's |knight's )?diagonal|doesn't (cover|reach|defend|attack|touch)|does not (cover|reach|defend|attack|touch)|don't (run|pass|touch|reach)|isn't a (legal )?knight|not a knight move|geometric|line of sight|blocked by|blocks the (rook|file|defen|diagonal)|sits (directly )?between|share no rank|not among them|not reachable|isn't on|not on the .*diagonal|f3-c4 isn't|diagonal.*blocked|line stops",
    "phantom / vanished / misplaced piece (wrong occupant, wrong origin square, captured piece misidentified)":
        r"no (black |white )?(rook|knight|bishop|queen|pawn)s? (left|on|there|to)|there is no|invents|fabricat|phantom|nonexistent|doesn't exist|was (already )?(traded|captured)|is empty|isn't even|(came|retreat)[a-z]* from|captures? (a|the) (rook|queen|pawn|knight)|captured.*(rook|knight|bishop)|misidentif|no such|already (left|advanced|gone)|has been empty|is on (f7|d1|c7|e8|c8|d3)|belongs? to (Black|White)|king (is|went|sits) on|on d3, not d4|sits on d3",
    "'open file' when a pawn still blocks it (half-open or closed)":
        r"open['’]? ?(e|c|h|b|d)?-?file|file.*(only )?half-open|half-open|file is (at most|only|fully|closed|blocked)|isn't actually open|no file is",
    "light/dark bishop swapped":
        r"light[- ]squared|dark[- ]squared|light square|dark square|square[- ]colou?r|black-squared",
    "rooks 'connected' / back rank 'cleared' when pieces still sit between":
        r"connect(s|ed|ing)? (the |my |their |his )?rooks|back[- ]rank|between the (two )?rooks|between the a8|rook coordination",
    "square called 'undefended/weak/vacant' that is covered or occupied":
        r"undefended|vacant|weakened|weakens|isn't actually a hole|still covers|remains? (covered|guarded)|defended twice|covered twice|three (black )?units|defenders?\b.*(shows?|would)|still holds|not open|d7 is open|supports? my knight",
    "'forced' or 'threat' claimed where other legal replies exist / capture illegal":
        r"forced|is not (even )?a legal|isn't a legal|illegal|not a threat|isn't really a threat|other legal|legal replies|four other|three legal|not forced|wasn't forced|could instead block",
    "material / piece count wrong (extra rook, bishop pair, up/down material, 'development complete')":
        r"material|bishop pair|only one (rook|bishop|white rook)|plural|(no|has) (rooks|knights) left|up a (full )?(rook|bishop|piece)|down a (full )?rook|two rooks|one rook|completed? develop|finished developing|still on (its|their) home|remains? undeveloped|doubles Black's rooks|count",
    "side / ownership / direction backwards (narrates wrong side, weakness assigned to wrong side, wrong wing)":
        r"backwards|reversed|wrong side|other side|perspective|narrat|whose|inverted|inverts|flips|opposite|own (weakness|liabilit|king)|beneficiar|belongs to|kingside makes no sense|queenside, not|not queenside|wrong wing|swings to the kingside",
    "pawn-structure mislabel (isolated vs doubled, backward, passed, weak squares)":
        r"(isn't|not|never|no longer) (actually )?(isolated|doubled|backward|passed)|(isolated|doubled).*(isn't|but|actually|by definition)|mislabel|calls? .* (isolated|backward|passed)|backward|already (passed|qualifies)|turns .* backward|is already passed",
    "true but generic: misses the concrete point (hanging piece, fork, mate, tempo attack, plan, theory)":
        r"never (notices?|identif|mentions?|states?|names?|registers?|engages?|explores?|calculat|gets concrete|touches|recover)|misses (the|that|both|two|any|concrete|entirely)|missing the|omits|neither (text|the thinking|thinking) .*(notic|identif|mention|name|engage|state|register|flag|calc)|stays? (at )?(a )?generic|generic|vague|filler|boilerplate|doesn't (note|identify|spell|engage|address|mention|flag|name)|never (mentions|explains|engages|calculat)|skips|only gestures|stops short|undersells|doesn't (capture|recognize)|nothing (stated )?is (false|factually)|no (clear |concrete |outright |chess |calculation )?(factual )?(error|claim is wrong)",
    "evaluation / judgement inverted (praises a bad move, wrong assessment of who is better, wrong plan)":
        r"praise|straightforwardly (good|strong)|unambiguously|glosses|engine (analysis|eval)|verdict|inverted|not the best plan|mistake|blunder|misconception|self-critique|overs(ells|tates)|frames? .* (as|purely|positively|simply)|drawish|likely a draw|space advantage.*backwards|treats the position as|underestimat",
    "lesson vocabulary leaked ('qualitative value', 'to do list', planning notes in output)":
        r"qualitative|to do list|to-do|task-management|planning notes|scratch reasoning|template|textbook phrase|jargon|engine-speak",
    "thinking-output gap: error confined to thinking (output dropped/softened it) or claim invented at generation":
        r"(doesn't|didn't|does not|did not) (survive|carry|leak|affect|reach)|not (present |even )?in the thinking|isn't (even )?present in the thinking|dropped|drops (the|this|it|that)|softened|softens|quietly|sidesteps|avoids repeating|avoids the error|not (in|part of) the thinking|invented (purely )?at generation|fabricated at generation|leaked|leak|correctly (notes|identif).*(but|then)|own (reasoning|analysis).*(then|but)|deliberately omits|drops this and",
}

TOOLS = {
    "per-piece attack/defence map with blockers (what does this piece reach; what attacks/defends this square)":
        r"attack(ed|er|s)?[- /]?(squares?|list|map|defend)|attacker[s/]*|defender[s/ ]*(count|list|check|lookup)|defended[- ]squares|line[- ]of[- ](sight|attack)|ray[- ]trace|diagonal.*(check|list|trace|lookup|scope)|squares?[- ]control|square[- ]defenders|mobility|scope|what (does|it) (this piece|the .* )?(newly )?attack|multi-target|legal-attacks|which (squares|pieces|enemy)|what .* (bears on|hits)|attacks-of|controlled",
    "hanging-piece / attacker-vs-defender count scan (incl. pins)":
        r"hanging|attacker(s)?[- /]?(vs\.?|versus|/)[- ]?defender|attackers?.and.defenders?|en prise|defender count on|zero (black )?defenders|undefended.*(check|scan)|capturable|move-safety|safe(ty)? check|for free",
    "piece inventory / occupancy / material count (FEN, what sits on square X, king location, capture log)":
        r"inventory|piece[- ]list|piece[- ]census|census (on|after)|occupan|occupant|what (piece )?(sits|is) (there|on)|material[- /]?(count|piece|tally)|piece[- ]count|FEN|board-state|king[- ]location|capture log|king's exact square|where, by color|piece[- ]history|square before the move|pawn-location|remaining (pieces|rooks|minor)|surviving pieces|whose piece|by colou?r and square|confirming.*sits",
    "file status: pawn census per file (open / half-open / closed)":
        r"file[- ](status|occupancy|census|defense|pawn)|per-file|files? (have|with) (zero|no) pawns|pawns? (remaining|present) on|pawn-occupancy|pawn census|which files",
    "square colour / bishop identity lookup":
        r"colou?r|light|dark|bishop[- ]identity|per-bishop|both (black|white)? ?bishops",
    "legal-move list / move legality / check-reply enumerator":
        r"legal[- ]move|legal reply|legal replies|legality|attempting (the move|Qxf7)|check-response|enumerat|replies|is illegal|appears in the legal|recapture(-| )options|two legal recaptures|comparison of the two",
    "short forcing-line / tactics search (checks, forks, mates, discovered attacks, candidate-move safety)":
        r"forcing|tactic|fork|mate|discovered|look-ahead|plies|ply|sequence|follow-up|candidate[- ]move|threat[- ]scan|threats? (list|check|search|scan)|next move|what .* would (attack|accomplish|threaten|do|open)|hypothetical|one-ply|two-ply|what (the|its) advance|promotion|zugzwang|king-vs-pawn|race|checks and threats|threat search|safety check on",
    "engine evaluation / move comparison":
        r"engine|eval|stockfish|move[- ]comparison|top alternatives|played move vs|comparison at this node|tempo/development counter|alternative immediate",
    "opening book / named plans / structure-typical plans":
        r"opening|book|theory|ECO|named[- ]plan|plans? lookup|typical|structure.*(plan|lookup|skeleton)|endgame-type|knight-maneuver|outpost finder|plan generator|candidate-plan|plans check|reroute",
    "pawn-structure classifier (isolated / doubled / backward / passed, weak squares, tension)":
        r"pawn[- ]structure|isolated|doubled|backward|passed|weak[- ](pawn|square)|pawn[- ]weakness|pawn[- ]tension|pawn-blocking|pawn-count|which squares .* pawns can",
    "board-state predicates: back-rank connectivity, development checklist, castling squares, king safety":
        r"back[- ]rank|between the (two )?rooks|development (checklist|counter)|home[- ]square|castling[- ]square|which side.*castl|flight[- ]square|king-safety|rook/file census|which files each side's rooks|pin[- ](detector|checker)|pinned-piece|alignment|battery",
    "none (voice, relevance, side-of-narration, judgement)":
        r"^none\b|^\s*none\b|not (something|a defect|a chess|essential)|no(thing)? (tool|lookup)|isn't a chess-content|framing choice|content choice|stylistic choice|language/judgment|wording imprecision",
}


def code(text, table):
    t = text or ""
    return [k for k, rx in table.items() if re.search(rx, t, re.I)]


def main():
    rows = [json.loads(l) for l in open(RUN / "scores.jsonl")]
    rows.sort(key=lambda r: r["id"])
    coded = []
    for r in rows:
        m = code(r.get("mistakes", ""), MISTAKES) if r.get("mistakes", "").strip() else []
        t = code(r.get("tools", ""), TOOLS) if r.get("tools", "").strip() else []
        if not t and r.get("tools", "").strip():
            t = ["(uncoded)"]
        if not m and r.get("mistakes", "").strip():
            m = ["(uncoded)"]
        coded.append({**r, "mistake_cats": m, "tool_cats": t})
    with open(RUN / "coded.jsonl", "w") as fh:
        for r in coded:
            fh.write(json.dumps(r) + "\n")

    n = len(coded)
    clean = [r for r in coded if not r.get("mistakes", "").strip()]
    print(f"n={n}  no-mistake items: {len(clean)}  (mean overall of those {statistics.mean(r['overall'] for r in clean):.2f})")

    def table(key, title):
        c = collections.Counter(x for r in coded for x in r[key])
        print(f"\n## {title}")
        for k, v in c.most_common():
            ids = [r for r in coded if k in r[key]]
            mo = statistics.mean(r["overall"] for r in ids)
            mf = statistics.mean(r["faithful"] for r in ids)
            print(f"{v:4d}  o={mo:.2f} f={mf:.2f}  {k}")

    wrong=[r for r in coded if r["faithful"]<=3]; gen=[r for r in coded if r["faithful"]>=4 and r["overall"]<=3]; ok=[r for r in coded if r["faithful"]>=4 and r["overall"]>=4]
    print(f"top-level: output has a false claim (faithful<=3): {len(wrong)} | true but generic/misses point (faithful>=4, overall<=3): {len(gen)} | accepted (faithful>=4, overall>=4): {len(ok)}")
    table("mistake_cats", "Mistakes (multi-label, of items with a note)")
    table("tool_cats", "Tools the judges asked for (multi-label)")

    # where did the error live?
    tf_lt_f = [r for r in coded if r["thinking_faithful"] < r["faithful"]]
    tf_gt_f = [r for r in coded if r["thinking_faithful"] > r["faithful"]]
    print(f"\nthinking worse than output (error caught before output): {len(tf_lt_f)}  ids={[r['id'] for r in tf_lt_f]}")
    print(f"output worse than thinking (error introduced at generation): {len(tf_gt_f)}  ids={[r['id'] for r in tf_gt_f]}")
    print("uncoded mistakes:", [r["id"] for r in coded if "(uncoded)" in r["mistake_cats"]])
    print("uncoded tools:", [r["id"] for r in coded if "(uncoded)" in r["tool_cats"]])


if __name__ == "__main__":
    main()

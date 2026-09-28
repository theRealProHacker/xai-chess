# Phase 1 scoping — inducing a chess-explanation DSL from natural-language commentary

*deep-research, full mode. 2026-09-06 (rev. 2 — reframed from top-down design to corpus-driven
induction). Grounded in `paper/landscape.md` §4c–5, `OPERATORS.md`, `pin_verify.py` FACTOR_TYPES,
and the 735,597-comment corpus in `commentary/`.*

## Research Question Brief

### Topic area
How to *derive* a domain-specific language for chess move-explanations from a large corpus of
human commentary — using language models to read what annotators actually say, and symbolic
machinery to keep only what the board can decide.

### Primary research question

> **How can a compositional, board-verifiable DSL for chess move-explanations be induced from a
> corpus of natural-language commentary, and what division of labour between language models
> and symbolic verification makes the induced vocabulary sound rather than merely plausible?**

The design commitment is bottom-up: the type inventory is discovered from what annotators write,
not invented from a textbook motif list. The symbolic side is not the author of the vocabulary —
it is the gate that decides which proposed term survives.

### FINER assessment

| Criterion | Score | Justification |
|---|---|---|
| Feasible | 5/5 | The corpus exists, is parsed, and each comment is already bound to a FEN, a move and a source. Induction runs on a sample; nothing new needs collecting. |
| Interesting | 5/5 | The two halves fail in opposite directions and each is the other's fix: an LLM reading 735k comments produces a rich vocabulary with no guarantee any term is real; the rule-decidable core guarantees reality but, hand-authored, has stalled at a handful of motifs for forty years. |
| Novel | 4/5 | `landscape.md` finds no compositional motif language, and every symbolic vocabulary in the record is hand-authored. Jhamtani et al. (ACL 2018) induced a six-way *category* scheme over this same commentary, which is the nearest precedent and is far coarser than a language. The risk to novelty is in the method half — corpus-driven DSL and taxonomy induction is an active field outside chess. |
| Ethical | 5/5 | No human subjects. The corpus is unlicensed upstream: usable under §60d UrhG for research, not redistributable (`commentary/LICENSING.md`). An induced *vocabulary* is not a redistribution of the comments; quoted examples are attributed. |
| Relevant | 5/5 | Whether the vocabulary is derived or invented decides whether the benchmark grades explanations people actually give. |
| **Average** | **4.8/5** | |

### Scope boundaries

**In scope**
- Induction methods: taxonomy/ontology induction from text, frame and predicate induction, library learning and DSL synthesis, semantic parsing with weak or latent supervision, LLM-driven schema discovery.
- The symbolic gate: which candidate predicates are decidable from a board, from a line, or only from an oracle; witness structure; how a term's extension is checked.
- The corpus as evidence: what kinds of claims annotators make, their distribution, and how to sample without inheriting the corpus's source skew.
- Coverage and residual: how to measure what fraction of real commentary the induced language expresses, and how to report what it cannot.
- Chess-specific precedent: Jhamtani's categories, Bizjak & Guid's static/dynamic terms, Caïssa's witness predicates, lichess `cook.py`, CQL, NAGs.

**Out of scope**
- Generation quality of prose from the DSL (a downstream question).
- Engine internals except as the decidability oracle.
- Re-inventorying `landscape.md`; it is an input.
- Implementation. This phase produces an induction *design* backed by evidence, not the pipeline.

**Key assumptions**
1. Commentary contains a recoverable stratum of claims about *why* a move is good, mixed with description, chat and evaluation — and that stratum can be separated. Jhamtani's category distribution is the first evidence for or against.
2. Board-decidability is the right filter for admitting a term. Terms that fail it are not discarded silently; they are reported as the residual.
3. A language model can propose vocabulary far better than it can guarantee it, so proposal and verification must be separate stages with separate failure modes.

### Sub-questions

1. **What is in the corpus?** What claim types do annotators actually make, in what proportion, and how do those distributions differ across the four sources — and what does the existing chess-commentary literature already establish about that?
2. **How is a language induced from such a corpus?** Which methods — clustering, frame induction, library learning, LLM schema proposal, semantic parsing — transfer to a domain where every candidate term is externally checkable, and what does each fail at?
3. **What makes the result sound?** How does the symbolic gate admit or reject a proposed term, what witness must a term return, and how are coverage and residual measured without the LLM grading its own proposals?

### Candidates considered

| # | Candidate | FINER avg | Why not selected |
|---|---|---|---|
| 1 | The induction question above | 4.8 | **Selected.** |
| 2 | "How should labour be divided between symbolic core and LLM in a chess-explanation DSL?" (rev. 1's RQ) | 4.6 | Top-down. Treated the corpus as a test set; the division-of-labour question survives as sub-question 3. |
| 3 | "What fraction of human commentary is expressible in a board-verifiable DSL?" | 4.2 | The measurement, not the method. Becomes the follow-on study once a grammar exists. |
| 4 | "Can an LLM reliably parse commentary into DSL expressions?" | 3.8 | Presupposes the DSL. Downstream. |
| 5 | "What formal representations exist for chess explanation?" | 2.6 | `landscape.md` §4c answers it. |

---

## Methodology Blueprint

### Research paradigm
**Pragmatist / design science.** The deliverable is a defensible induction design plus the
evidence constraining it. Judged by whether the resulting language works, not by a hypothesis test.

### Method
**Structured comparative synthesis** across three literatures that do not cite each other —
chess commentary and explanation, corpus-driven vocabulary/DSL induction, and neuro-symbolic
verification — read through one question: *does the method produce terms whose truth can be
checked outside the model that proposed them?* Unit of analysis is a method, coded on a fixed
axis set. PRISMA is rejected: the corpus is heterogeneous and mostly non-empirical, so a pooled
flow would be ceremony.

### Data strategy
**Secondary literature**, plus the repo's own corpus as the object the design must fit.
Sources: ACL Anthology, arXiv, ACM DL, IEEE Xplore, Springer LNCS/LNAI (ACG, KI, CG),
NeurIPS/ICLR/ICML/AAAI/IJCAI, ICGA Journal, *Artificial Intelligence*.
**Sampling:** purposive, seeded from `landscape.md` §4c–5 and `PAPERS.md`, expanded by forward
citation on Jhamtani, Bizjak & Guid, Caïssa and the library-learning line.
**Time frame:** 1976–2026; the induction-method sweep restricted to 2018 onward.

### Analytical framework
1. **Corpus characterisation from the literature** — what is already known about the claim-type distribution in chess commentary (Jhamtani's six categories and anything since), and what that predicts for the recoverable stratum.
2. **Method matrix** — each induction method coded on: what it proposes (term / type / program / frame), supervision required, whether proposals are externally checkable, how it handles the long tail, reported evaluation.
3. **Gate design** — cross-tabulate proposals against decidability class: board-decidable, line-decidable (needs search), oracle-only, not decidable. The last class is the residual and is reported, not hidden.
4. **Witness requirement** — for each admitted class, what a term must return beyond a boolean, differenced against Caïssa (returns witnesses, unevaluated) and lichess `cook.py` (discards them).
5. **Falsification** — state what corpus evidence would show the induced-vocabulary approach failing, e.g. a claim-type distribution where the "why" stratum is too thin or too idiosyncratic to compress.

### Validity criteria

| Criterion | Strategy |
|---|---|
| Source verifiability | Every claim traces to a fetched source; unconfirmed items marked **UNVERIFIED** inline, per `landscape.md`'s existing discipline (PARADISE's syntax, Caïssa's evaluation and CQL's built-in filters are already known-unreachable and stay marked). |
| Non-circularity | The design must keep proposal and verification in separate hands. A method where the LLM both proposes a term and adjudicates its truth is coded as failing this criterion, not as a variant. |
| Corpus skew | 48% of comments are GameKnot, 44% lichess studies. Any induction design must stratify by source and report per-source vocabulary, or it induces GameKnot's idiom and calls it chess. |
| Confirmation bias | Phase 2 opens with a disconfirming sweep for an existing motif grammar or chess ontology, rather than re-confirming `landscape.md`'s gap claim. |
| Lexical discipline | *Faithfulness* names explainer-internal fidelity only; this work concerns decision-level causal load-bearing. |
| Transfer validity | Non-chess induction results are labelled as transfer, never asserted as chess results. |

### Limitations by design
- Primary texts for PARADISE and the Advice Language clause syntax are unreachable; those holes get reported as holes.
- The coverage number itself needs the pipeline run — this report designs the measurement, it does not produce the figure.
- Commentary records what annotators *said*, which is not what caused the move. The induced vocabulary inherits human error; only the symbolic gate and the operator algebra address that, and neither is part of induction.
- The induction-method literature moves fast; the report is a snapshot with a stated cut-off.

### Ethics
AI-assisted research disclosed. Corpus used under the §60d UrhG research lane, not redistributed;
quoted comments attributed where the corpus preserves the annotator.

### Reporting standard
None mandated. Structured comparative synthesis with an explicit search log.

### Preregistration
Not applicable.

---

## Devil's Advocate Report — Checkpoint 1 (rev. 2)

### Verdict: PASS (three major issues carried into Phase 2)

### Critical issues
No critical issues identified.

### Major issues

1. **The corpus may not contain the thing being induced.**
   - *Type*: Evidence / assumption
   - *Location*: Assumption 1, sub-question 1
   - *Problem*: Jhamtani's own scheme puts a large share of commentary in description, quality and contextual chat rather than rationale. If the "why" stratum is a small and idiosyncratic minority, an induced vocabulary is a vocabulary of the *sayable*, not of reasons — and the project's premise weakens before any method question is reached.
   - *Recommendation*: Sub-question 1 runs first and is treated as a go/no-go. Recover the actual category proportions from the literature, and state the threshold below which the induction design is not worth building.

2. **Induction from commentary launders human error into the type system.**
   - *Type*: Method
   - *Location*: Assumption 3, gate design
   - *Problem*: Annotators are often wrong, and the frequent terms will be the popular ones, not the load-bearing ones. Frequency in the corpus is evidence about *language*, not about causation — and the benchmark's whole point is that those come apart.
   - *Recommendation*: Keep two separate quantities and never let them merge — how often a term is *said*, and whether the factor it names is causally load-bearing under the operator algebra. The DSL's job is to make the second measurable, not to inherit the first as ground truth.

3. **Corpus skew is a vocabulary skew.**
   - *Type*: Scope
   - *Location*: Validity, corpus skew
   - *Problem*: Nearly half the comments come from one amateur site. Induction over the pooled corpus will produce GameKnot's register with a lichess accent, and the per-source word-count differences already observed suggest the registers genuinely differ.
   - *Recommendation*: Stratified induction with per-source vocabularies compared, and a term admitted only if it appears across sources — an implicit cross-validation the design should make explicit.

### Minor issues
- "Symbolic AI" needs pinning to a tradition (production rules / logic programming / typed terms); they have different induction stories.
- NAGs are a standing, standardised output vocabulary nobody targets — a cheap comparison point for the induced inventory, easy to forget.
- Bizjak & Guid's dynamic-over-static result (0.418 vs 0.252 top-1) says the reason is not in the position alone. Induction will surface line-referring claims early; the gate's decidability classes must be ready for them rather than discarding them as undecidable.

### Strongest counter-argument
> "You are inducing a language from a corpus of what amateurs happened to type, then filtering it
> through what a rule engine happens to decide. The intersection is neither what people mean nor
> what matters — it is the accident of those two filters overlapping, and calling it a language
> for chess explanation overstates it considerably."

The defensible reply is to report that intersection *as* the finding, with the residual on both
sides quantified, rather than presenting the surviving vocabulary as a complete account.

### What's missing
- The reverse direction: nothing here tests whether the induced language can be *read back* by a player. No system in `landscape.md` §4c evaluates its representation with a user.
- Cost. LLM extraction over a 735k-comment corpus is a real budget; the design should say what sample size the evidence supports.

### Stress test

| Test | Result |
|---|---|
| Remove the strongest source — does the argument hold? | Yes — drop Jhamtani and the induction case still stands on the general taxonomy-induction literature. |
| Flip the question — is hand-authoring the vocabulary credible? | **Partly.** Forty years of hand-authored vocabularies exist and none is compositional or evaluated; but a hybrid — hand-authored core, induced periphery — is a live alternative the report must argue against, not assume away. |
| Does it generalise? | Yes — any domain with commentary plus a decidable substrate (Go, shogi, poker). Which also means prior art likely exists outside chess. |
| "So what?" | Yes — a derived vocabulary is what makes the benchmark grade explanations people actually give, rather than the ten motifs a designer thought of. |

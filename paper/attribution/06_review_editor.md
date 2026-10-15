# Editorial Review — Phase 5, `05_report.md`

## Verdict: MINOR REVISION — weighted score 3.8 / 5

| Dimension | Weight | Score | Note |
|---|---|---|---|
| Originality & contribution | 20% | 3.5 | Single-decomposition framing and residual-as-coverage are useful; method parts are existing. |
| Methodological rigor | 25% | 3.5 | Math mostly right; one aggregation bug in $\rho_L$ (C-1), one false linearity claim (M7), one untested claim stated as shown (§5.2). |
| Evidence sufficiency | 25% | 4 | 16 citations spot-checked: 12 OK, 3 minor imprecisions, 1 misattribution. |
| Argument coherence | 15% | 4 | RQ → front/back curve → protocol is clean; DA corrections integrated. |
| Writing quality | 15% | 4 | Terse, scoped honestly. Some table/text inconsistencies. |

Strengths: (1) §8's coverage/error split is correct and protects the project's main value. (2) §5.1 Table 2 arithmetic checks out (partial $R^2$ = 0.99/0.68/0.76/0.76/0.53/0.24; square-identity shares 84%/10%). (3) M6, M9, M3, M4 formulas are correct as written; completeness identities hold. (4) Limitations (§11) are specific and do not hedge.

## Math check (per method)

| Method | Status |
|---|---|
| M1 point form | Correct; ⟨∇y, q⟩ is left unassigned (another source of incompleteness, unstated). |
| M1 board-path IG | Completeness holds (gradient theorem on $\Delta h=\sum_G W^G\Delta\tilde f^G+\Delta\varepsilon$, $q$ cancels). $\sum_s$ missing in $\mathrm{IG}_G$. |
| M2 | Correct. Cost undercounts (see Minor). |
| M6 φ | Correct two-player Shapley; completeness exact. |
| M6 $\rho_L$ | **Wrong aggregation** (C-1). |
| M9 | Correct; $\sum\varphi=y(x)-\mathbb E y(x_0)$ holds; cost arithmetic (31×20×8≈5k; 200×3×30 s≈5 h) checks. |
| M10 | Split identity correct ($\sum_c\Delta c_j=\Delta h_j$). Text mixes AttnLRP (zero-ref) with DeepLIFT rescale (baseline-ref) in one pass (Minor). |
| M3 | LEACE formula matches Belrose et al. |
| M4 | $(J^\top)^+\nabla y$ equals Bai et al.'s $\nabla y^\top J^+$; surjectivity argument at bit level correct. |
| M5 | Standard DAS interchange; graded IIA fine. |
| M7 | "network is linear in the embedding" false (Major-1). |
| M8, M12 | Fine. |
| §8 err | $=\mathbb E|A_\varepsilon|/\mathbb E|\Delta|$ for complete methods: correct. |

## Required revisions

### Critical

- [ ] **C-1. $\rho_L$ cancels across pairs (§6.1 M6, l.223; inherited by §8, H2).** Numerator $\sum\phi_n$ is signed, denominator $\sum|\Delta^{\rm true}|$ is not. With a perfect vocabulary ($\phi_n=\Delta^{\rm true}$ every pair) and edits of mixed sign, $\rho_L=\sum\Delta/\sum|\Delta|\approx0$. The headline curve would read "nothing named" for a complete vocabulary. Fix: $\rho_L=\sum_{\rm pairs}\mathrm{sign}(\Delta^{\rm true})\,\phi^{(L)}_n/\sum_{\rm pairs}|\Delta^{\rm true}|$, equivalently orient every pair so $\Delta^{\rm true}\ge0$ (the two-player φ is antisymmetric under $x\leftrightarrow x'$). State it. Same fix for M11's per-pair $A_\varepsilon/\Delta^{\rm true}$ (l.287), which is undefined at $\Delta\approx0$; use the aggregated, oriented ratio.

### Major

- [ ] **M-1. M7 linearity (l.275).** "With attention and LayerNorm frozen, the network is linear in the embedding" is false: $\sigma(P_L(u))$ remains. Ameisen et al.'s graph is linear *between nodes given frozen feature activations*, attention patterns and LN denominators. Fix the sentence to that. Also "smolgen is unmodelled" (l.275): smolgen only biases attention logits, so freezing patterns freezes it; say its routing is hidden like QK, not unmodelled.
- [ ] **M-2. Overclaim in §5.2 (l.154) and §10 (l.386).** "A random-initialised embedding … a decoder would fit it almost as well, which shows directly that the embedding FVU says nothing about learning" is an untested prediction stated as shown, and likely wrong in part: a random embedding weights all 8 history frames and the broadcast planes, which the occupant vocabulary cannot fit; the trained 0.002 FVU means lc0 *learned* to nearly ignore them at $e$. Fix: "predicts", make it a C1 check (add to H1), and replace "says nothing about learning" with "says little beyond locality and low weight on history planes".
- [ ] **M-3. Castling claim (l.154, l.205).** "History, castling and rule-50 content fall into $\varepsilon_e$": the extractor now puts castling rights on the king square (commit 473b3ce). Check and drop castling from the list if covered; rule-50 (broadcast, position-level) stays.
- [ ] **M-4. Two coverage measures that are not comparable (§8, l.325; §9.2).** $\rho_L$ is a signed share; M9 coverage $1-\varphi_\varepsilon/\sum|\varphi|$ is an absolute share. Either use one definition for both or state they differ and never put them on one axis.
- [ ] **M-5. H4 localisation for distributional M9 (l.351, l.365).** M9 with a pool baseline explains $y(x)-\mathbb E y(x_0)$, not the edit. Ranking the edited family among $|A_G(x)|$ tests the wrong thing. Fix: localise on $A_G(x')-A_G(x)$ with a shared baseline pool.
- [ ] **M-6. Identifiability column inconsistent (Table 4, l.299, l.303).** M2 "unique per joint patch" and M6 "exact" sit beside M9 "ridge-dependent at family level", but M2 single-family patches use the same $W^G_L$ split, and M6's named/residual split depends on decoder fit. Fix: M2 "joint patch unique; per-family ridge-dependent"; M6 "exact given $D_L$; depends on fit (C3)".

### Minor

- [ ] l.92 vs l.106 — **Kulkarni et al. (2026)** misattributed: biblio T5.13 says the hybrid adds a concept bottleneck beside retained SAE latents; *matching residual directions to candidate concepts* is Shang et al. only. Drop Kulkarni from that sentence.
- [ ] l.104 — Yeh et al. completeness is accuracy *normalised between random and full*, not "fraction retained" (T4.12). Reword.
- [ ] l.94 — Bilodeau et al. *prove* near-random performance for rich model classes and *argue* for end-task tying (T1.17); split proof from argument.
- [ ] l.203 — add $\sum_s$ to $\mathrm{IG}_G$.
- [ ] l.82 — "Square identity can absorb credit only in methods that evaluate a gradient at a single point": in M1 point form $q$ receives nothing; ⟨∇y,q⟩ is simply dropped. Say that instead.
- [ ] l.215 — M2 cost: 30 families × 11 taps counts single patches only; joint + leave-one-out ≈ 61 per tap (~670/position). Real minimal edits change few families, so give the count per changed family.
- [ ] l.233 — M10: "one backward pass with AttnLRP rules" then "DeepLIFT rescale against the same baseline" contradicts the one-reference rule two sentences later. Pick DeepLIFT/G-DeepSHAP; cite AttnLRP only for softmax handling.
- [ ] l.144 — "declines monotonically after" rests on L5 and L9 only; L3–L4, L6–L8 are fitted (commit 95c3419) but not in Table 1/2. Add them or say "falls from L2 to L9".
- [ ] l.170 — (g) fit capacity: training rows are 16,000 × 64 ≈ 1M tokens, not 16,000; and capacity is constant across taps, so (g) explains decay only via rarer bits. Reword.
- [ ] Table 4 — M10 is "all taps, one pass" but "Combines: no", while M6 (also a sweep) is "yes (sweep)". Mark both the same.
- [ ] l.323 — "dummy axiom" is a Shapley notion; for M2 joint and M1 board-path IG say "unchanged families have zero delta".
- [ ] l.72 — define value target precisely ($\ell_W-\ell_L$ pre-softmax), since §9.2 fixes the metric as a logit.
- [ ] l.40 — "about 122 entries": theme counts include duplicates (Ameisen in T5 and T6); give the deduplicated count (the reference list has 88).

### Suggestions

- §1 l.24 corrects the user's note ("mostly square identity" is the front). Quote which fit the note referred to before contradicting it.
- Add a one-line worked example of M6 on one real edit (pin removal) with the four suffix values; the user wants concrete examples.

## Citation spot-check (16)

| Report claim | Biblio annotation | Verdict |
|---|---|---|
| Covert: 26 methods by removal (l.92) | T1.12 same | OK |
| Bilodeau: proven uninformative unless tied to query (l.94) | T1.17 proves near-random; *argues* end-task | Minor |
| Zhou: 65.9% unexplained, ResNet-50 (l.104) | T2.12 same | OK |
| Engels: ~half predictable; earlier layers reduce most (l.104) | T5.9 same | OK |
| Yeh: completeness = accuracy retained (l.104) | T4.12 normalised random→full | Minor |
| Havasi: missing vocab vs weak map (l.106) | T4.9 same | OK |
| Kulkarni: residual matched to candidate concepts (l.106) | T5.13 hybrid dictionary | Misattributed |
| Leask: no canonical size (l.106) | T5.10 same | OK |
| Makelov 2025: supervised dictionary yardstick; newer SAEs approach (l.110) | T5.11 same | OK |
| MIB: DAS strongest, SAE ≈ neurons (l.110) | T7 annotation same (T6.15 omits DAS) | OK |
| Lin 2026: BT4, EV >0.6, hand-labelled + detector-checked (l.114) | T8.7 same | OK |
| Sutter: near-perfect IIA on random nets (l.118) | T7.12 same | OK |
| Park: coincide under causal inner product (l.118) | T3.16 same | OK |
| Ameisen: completeness 0.80 / replacement 0.61 (l.275) | T5.6 same | OK |
| Lomasov: 85%→50–65%, 6 concepts, other model (l.126) | T8.14 same | OK (DA fix applied) |
| Sandmann: late override (l.126) | T8.6 same | OK |

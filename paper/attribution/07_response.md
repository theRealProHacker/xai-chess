# Response to reviewers — Phase 6 revision

`05_report.md` is unchanged; the revision is `07_report_final.md`. Sources: Ed = `06_review_editor.md`, Eth = `06_review_ethics.md`, DA = `06_review_da3.md`. Locations refer to `07_report_final.md`.

## Critical and major

| # | Source | Issue | Action | Location |
|---|---|---|---|---|
| 1 | Ed C-1 | $\rho_L$ signed numerator over unsigned denominator cancels across pairs | All pairs oriented so $\Delta^{\rm true}\ge0$ (stated in notation); $\rho_L=\sum\phi_n/\sum\Delta^{\rm true}=\sum\mathrm{sign}(\Delta)\phi_n/\sum|\Delta|$; antisymmetry of $\phi_n$ stated; the mixed-sign failure stated | §3.2, M6 (§6.1) |
| 2 | Ed C-1 | M11 per-pair $A_\varepsilon/\Delta^{\rm true}$ undefined at $\Delta\approx0$ | M11 uses the aggregated, oriented coverage of §8 | §6.7 |
| 3 | Ed M-1 | M7 "network linear in the embedding" false | Rewritten: edges linear between nodes given frozen attention patterns, LN denominators and feature activations; network stays nonlinear through $\sigma$ | §6.5 |
| 4 | Ed M-1 | "smolgen is unmodelled" | Smolgen biases attention logits before softmax (checked in ONNX graph), so freezing patterns freezes it; its routing is hidden like QK routing | §4.7, §6.5 |
| 5 | Ed M-2, DA minor | Random-init claim stated as shown | Now a prediction and a C1 check, added to H1; "says nothing about learning" replaced by "says little beyond locality and occupancy"; notes that fen-only history duplicates occupancy, so history planes cannot separate the cases here | §5.2 item 5, §9.3 H1, §10 |
| 6 | Ed M-3 | Castling listed as falling into $\varepsilon_e$ | Extractor puts castling rights on the king square (`king.castle`, 473b3ce; fits postdate it). Castling planes are broadcast to all 64 squares (`lc0_planes.py`), so castling content stays in $\varepsilon_e$ on non-king squares only; rule-50 stays | §3.1, §5.2 item 2, C2 |
| 7 | Ed M-4 | Two incomparable coverage measures | One definition, $\mathrm{Cov}=\sum A_{\rm named}/\sum\Delta^{\rm true}$ on oriented pairs, instantiated for M6, M2 joint, pairwise M9, M0; distributional M9's absolute residual share kept on its own axis | §8, M9 (§6.1), §6.7 |
| 8 | Ed M-5 | H4 localisation on $|A_G(x)|$ tests the wrong thing | Localisation on $A_G(x')-A_G(x)$ with a shared baseline or pool, for all non-pairwise methods; H4 rewritten, M0 added as floor | §9.2, §9.3 H4 |
| 9 | Ed M-6 | Table 4 identifiability column inconsistent | M2 "joint patch unique; per-family ridge-dependent"; M6 "exact given $D_L$; depends on fit (C3)"; M1, M9, M4, M8 aligned | Table 4 |
| 10 | DA MJ-1 | No first recommendation | New "What to try first": M2 + M6 on pin edits (pinner off line, interposition), taps $e$ (C2), L2, L5, L9, ~1,000 pairs, C1 + C3, M0 floor; cost; decision rules naming when to promote M8 and when M5 | §9.5, Table 6 (stage A0), Abstract |
| 11 | DA MJ-2 | $\rho_L$ framed as the attribution | Reframed as coverage and a tap-selection guide; per-family attribution from M2/M9 where $\rho_L$ is high and above C1; $\rho_e\approx1$ under C2 by construction stated and excluded from H2 | Abstract, §1, M6, §5.2 item 4, §9.3 H2, §10 |
| 12 | DA MJ-3 | No identifiability statement | New subsection: identifiable ($\Delta^{\rm true}$, joint named patch, $\gamma_L$, $\phi_n$, $\phi_\varepsilon$, $\rho_L$, LEACE effect of $G$ with its correlates) vs not (per-family split among collinear families); remedy: attribute to the edit, merge unstable families | §7.1 |
| 13 | DA MJ-4a | Missing features→$y$ surrogate baseline | Added M0: surrogate on features or probe outputs, family Shapley, explicit residual (PCBM-h style); correlational floor | §6.0, Table 4, §8, §9.3 H4, §9.5 |
| 14 | DA MJ-4b | No policy treatment | New §6.8: policy logit is a bilinear form of the from- and to-square tokens at `/encoder9/ln2` (read from ONNX); exact four-player split there; needs an ln2 decoder; legality constraint; value-vs-policy $\rho_L$ tests cause (f) | §6.8, §9.6, §11 |
| 15 | Eth E1 | Table "square level only" contradicted by Sandmann and Lin | Reworded: squares and heads (Jenner), steering directions (Sandmann), unsupervised sparse features (Lin); none targets a fixed human feature family | Table 10 |

## Minor and suggestions

| # | Source | Issue | Action | Location |
|---|---|---|---|---|
| 16 | Ed | Kulkarni misattributed (residual matched to candidates) | Dropped from that sentence; kept for the hybrid dictionary in M12 | §4.3, M12 |
| 17 | Ed | Yeh completeness wording | "accuracy of the head on concept-reconstructed activations, normalised between random and full accuracy" | §4.3 |
| 18 | Ed | Bilodeau proof vs argument | Split: proves near-random for rich model classes; argues for end-task tying | §4.1 |
| 19 | Ed | $\sum_s$ missing in $\mathrm{IG}_G$ | Added | M1 |
| 20 | Ed | M1 point form: $q$ dropped, not absorbing credit | Stated in notation and M1 | §3.2, M1 |
| 21 | Ed | M2 cost undercounted | $2k+1$ suffix passes per tap per changed-family count; ~12 families for pin edits | M2, §9.5 |
| 22 | Ed | M10 mixes AttnLRP and DeepLIFT references | DeepLIFT rescale / G-DeepSHAP against one baseline; AttnLRP only for softmax handling | M10 |
| 23 | Ed | "declines monotonically" rests on L5, L9 only | Table 2 now has all eleven taps from `results_square_decoder.json` (incl. features-only row); partial $R^2$ falls at every tap L2→L9 | Table 2, §5.1 |
| 24 | Ed | (g) fit capacity misstated | ~1M training tokens (16,000 × 64), 14,565 kept bits; capacity constant across taps, so (g) matters only via rare bits | §5.4 |
| 25 | Ed, DA | Table 4: M6 "yes (sweep)" vs M10 "no" | Both "sweep" | Table 4 |
| 26 | Ed | "Dummy axiom" for non-Shapley methods | "Zero delta" for M2 joint and M1 board-path IG; dummy axiom only for M6 and pairwise M9 | §8 |
| 27 | Ed | Value target undefined | $y=\ell_W-\ell_L$, pre-softmax WDL logits (`/value/dense2`), side to move's view; policy as pre-softmax logit | §3.1 |
| 28 | Ed | "about 122 entries" includes duplicates | "about 122 entries with overlaps; 89 distinct works cited" | §2, Abstract |
| 29 | Ed sugg., DA minor | §1 corrected the user's note | Correction removed. Note quoted; measured square-only and features+square FVUs stated without comment; "Square identity is the dominant term at the front, not the back" deleted | §1, §5.3 |
| 30 | Ed sugg., DA minor | No worked example | Worked example on a concrete pin (FEN after 3.Bb5 d6) with two edits; real extractor bit diffs (Table 7); step list; output templates with "—" (Tables 8, 9); policy variant (Bxc6) | §9.6 |
| 31 | DA minor | FEN inputs carry no history | Stated with source (`lc0_planes.py` fen-only fill); edit protocol consequence (re-encode, all frames carry edit, no artefact) and scope limit (synthetic history only) | §3.1, §5.2 item 1, §9.1, §11 |
| 32 | DA minor | M8∘M2 composition omitted | Added as feature-path composition (path patching in decoder coordinates, attention live), placed under erasure/substitution families | §6.2, Table 4, §10 |
| 33 | DA minor | M11 omitted from "run without port" | List now M0, M2, M3, M6, M8, M8∘M2, M9, M11 | Table 4 note |
| 34 | Eth E2 | Jenner "locate look-ahead in specific heads" | "use attention-head ablation in their evidence for look-ahead" | §4.7, §6.5, §10 |
| 35 | Eth E3 | AxBench without follow-up caveat | Clause added: 2026 follow-up, not verified, contests the SAE result | §4.4 |
| 36 | Eth E4 | Lin EV exception | "lower for attention in middle layers" added | §4.5 |
| 37 | Eth E5 | Disclosure scope | Names Claude (Anthropic) via Claude Code; states claims were checked against AI-written annotations from abstracts and landing pages, not full texts read by a human; lists non-AI data sources | AI-Use Disclosure |
| 38 | Eth E6 | Inconsistent in-text tier flags | In-text flags removed; every sub-Tier-1 source marked in the reference list | §2, References |
| 39 | — | New source for M8∘M2 | Goldowsky-Dill et al. (2023), from `02_biblio_T6.md`, Tier 2 | References |
| 40 | — | Taps mis-described as "post-LN block outputs" | Taps are `/encoder{i}/ln1` (post-attention, pre-FFN); heads read `/encoder9/ln2`; L9 suffix includes FFN 9 (ONNX graph) | §3.1, §5.3 |

## Acknowledged limitations (not resolved by text)

| Issue | Status |
|---|---|
| Per-family credit among collinear families (DA MJ-3) | Not identifiable from the decoder; reported jointly or merged; stated as a limit of the method, §7.1 |
| AxBench 2026 follow-up (Eth E3) | Not verified; flagged as unverified in text |
| Jenner head-localisation claim (Eth E2) | Annotation not extended; wording softened instead |
| Policy decoder at `/encoder9/ln2` | Not fitted; §6.8 and §11 say so |
| Real-history inputs | Not available in the puzzle data; scope limited in §9.1 and §11 |
| Prose length | ≈9,100 words of prose excluding tables and display math |

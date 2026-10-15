# Phase 5: Ethics review of 05_report.md

## Verdict: CONDITIONAL

There are no fabricated references and no systematic misrepresentation. One claim is attributed to sources that do not support it (E1). Fix E1 before delivery. E2 to E6 are recommended.

## Dimension assessment

| Dimension | Status | Notes |
|---|---|---|
| AI disclosure | pass (minor warn) | §AI-Use Disclosure names AI scope (search, verification, synthesis, DA, drafting), human-sourced numbers, and author control. It does not name the model/tool, and it does not say that claims were checked against AI-written annotations rather than full texts (E5). |
| Attribution integrity | warn | 88/88 references resolve by DOI/arXiv/URL to a `02_biblio_T*.md` entry. Every in-text author-year has a reference entry, and every reference is cited. 45 claims spot-checked, 40 clean, 5 with issues (E1 to E4). |
| Fabrication risk | pass | No ghost citations. Mueller et al. and Ameisen et al. use APA ellipsis truncation, which is correct. Makelov et al. (2024) has 4 authors in the reference list and 3 on arXiv; the biblio notes the discrepancy. |
| Dual-use | pass | Risk level None (chess-network interpretability). |
| Fair representation | pass (minor warn) | Counter-evidence is presented (Makelov 2025 counter-signal, Wu 2024 reply to Makelov 2024, the Pálsson vs Sutter tension, the §10 strongest objection). The report corrects its own biblio's overclaim: T8 calls Lomasov "independent evidence for the same decay", and §4.8 correctly says it shows probe decay on a different model. See E3 for an omission. |
| Data ethics | pass | Lichess puzzle FENs (public, CC0) and model weights only. No PII. |
| Human subjects | n/a | None. No IRB needed. |
| Conflict of interest | pass | No self-citation. The project's own prior work (noise-floor bias, MaskPinner) is stated as project work. T8 notes a same-lab overlap (Lin/He, Fudan) and DeepMind authors on DeepMind's own model (Schut, McGrath). The report does not repeat these; that is low-stakes. |

## Issues

**E1 (fix before delivery): Table 7, row "Interventional validation of named families on lc0".** The cell reads "lc0 interventions exist at square level only (Jenner et al., 2024; Sandmann et al., 2025; Lin et al., 2026)". Two of the three sources contradict "square level only":
- Sandmann annotation (T8 #6): a logit lens plus *steering along a direction* (61.7% recovery). That is a direction intervention, not a square-level one.
- Lin annotation (T8 #7): pathways validated by *ablation and copy-patching of sparse features*. That is a feature-level intervention, though on unsupervised features.
- Only Jenner (per-square patching, head ablation) fits.

Suggested wording: "lc0 interventions target squares, heads, steering directions or unsupervised sparse features; none target a fixed human feature family." The gap claim survives. The evidence sentence must change.

**E2 (recommended): §4.7 and §6.5 say Jenner et al. "locate look-ahead in specific attention heads".** The T8 annotation records "attention-head ablation" and a bilinear probe, but not that look-ahead is located in heads. The paper does report this, so the claim is probably true. Either extend the annotation, or soften the wording to "find attention heads necessary for look-ahead".

**E3 (recommended): §4.4 cites AxBench (Wu et al., 2025) for "supervised directions match or beat SAE latents" without the caveat in the T7 annotation.** The annotation says a 2026 follow-up (arXiv 2605.31183, unverified) contests the SAE result. Add one clause, or verify and cite the follow-up.

**E4 (minor): §4.5 says Lin et al. "reach explained variance above 0.6 per module".** The annotation qualifies this: "lower for attention in middle layers". Add the exception.

**E5 (minor): disclosure scope.** Name the AI system used. State that cited claims were checked against Phase-2 annotations written by AI agents from abstracts and landing pages, not against full texts read by a human. The current sentence ("Cited claims were checked against the annotated bibliographies") reads as stronger verification than took place.

**E6 (minor): in-text tier flags are inconsistent.** Tier 2 and 3 are flagged in-text for Ameisen, Lindsey, Paulo, Lin and Sandmann, and Tier 4 for Chan. They are not flagged for the other Tier 2–3 sources: Heimersheim & Nanda (2024), Kramár et al. (2024), Chen et al. (2020), Wu et al. (2024), Poupart (2024), Mahinpei et al. (2021) and Margeloiu et al. (2021). Some of these carry load (Chen 2020 grounds the GT tiers; Heimersheim & Nanda grounds C9). Flag them, or drop all in-text tier flags and rely on the reference list.

Also noted, no change needed: §4.8 says Tomlin's late-layer profile is "possibly because their labels are commentary mentions". This is the report's own hedged inference from the annotation's stated weakness, not a claim the source makes. The hedge is adequate.

## Spot-check log (45 claims, against annotation text)

Clean (40): Aas; Abraham/CEBaB; Achtibat 2023, 2024; Adebayo; Ali; Ameisen (completeness 0.80 / replacement 0.61, error nodes, frozen attention); Bai; Beckers & Halpern (constructive abstraction); Bilodeau; Chan (Tier 4); Chen 2020; Covert (26 methods); Engels (half of error linearly predictable; earlier layers reduce it most); Farnik; Frye; Geiger 2020; Hammersborg & Strümke; Havasi (side channel; missing vocab vs weak map); He/GCAV (only multi-layer concept map); Heimersheim & Nanda (OR gates); Janzing; Kramár (two false-negative modes); Kulkarni; Kumar I.E.; Lawson; Lomasov (85% to 50–65%, 270M, six concepts); Makelov 2025 (supervised dictionary yardstick; newer SAEs approach it); Margeloiu; MIB (DAS best, SAE ≈ neurons); Nanda 2023 (mine/theirs linearity); Pálsson & Björnsson (nonlinear probes + erasure); Park (causal inner product); Ramaswamy (~32 concepts); Schut (behavioural validation); Sutter (near-perfect IIA on random nets); Sundararajan & Najmi; Wu 2024 reply; Zhang & Nanda (logit metric); Zhou (65.9% residual).

Issues (5): Table 7 Sandmann and Lin (E1), Jenner (E2), AxBench (E3), Lin EV (E4).

Retraction Watch: no known retractions among the cited works. Not checked against the database.

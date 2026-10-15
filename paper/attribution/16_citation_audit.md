# Phase 5a — Citation compliance audit

Date 2026-10-09. Style IEEE numeric (10_config.md). Inputs: src/part1-4.md, src/refs.py, 02_biblio_T1-T8.md. paper.md rebuilt with `python3 src/build.py paper.md` (89 cited, 0 unused).

## 1. Orphans

- In-text -> list: 0 (build.py exits on any unknown key).
- List -> in-text: 0 (89 entries in refs.py, 89 cited).
- Numbering: by first appearance; ranges rendered "[n]–[m]" (IEEE).

## 2. IEEE format

All 89 entries checked: initials-first authors, "et al." only for >6 authors (18 entries, all >6), quoted sentence-case titles, abbreviated italic venues, vol./no./pp./Art. no. where known, "doi:" or "[Online]. Available:" URL, arXiv preprints as "Year, arXiv:ID". Tier tags (2–4) are deliberate (Appendix E, References note) and match the biblio tiers.

Fixes applied in src/refs.py:

| Key | Change | Source |
|---|---|---|
| karvonen2024 | added pp. 83091–83118 | Crossref 10.52202/079017-2644 |
| beckers2019 | added no. 1, pp. 2678–2685, doi 10.1609/aaai.v33i01.33012678 (replaces arXiv URL) | Crossref |
| espinosa2023 | added no. 10, pp. 11791–11799, doi 10.1609/aaai.v37i10.26392 | Crossref |
| geiger2020 | added pp. 163–173, doi 10.18653/v1/2020.blackboxnlp-1.16 | Crossref |
| ferrando2024 | added doi 10.18653/v1/2024.emnlp-main.965 | Crossref |
| shin2023 | added PMLR vol. 202 (consistency with other ICML entries) | ICML 2023 = PMLR 202 |
| poupart2024 | added workshop venue (InterpPol @ RLC 2024); stays Tier 3 | arXiv comment |

Not fixed (no verified data in hand): PMLR pages missing for kim2018, park2024, ravfogel2022, shin2023; he2025 (ICCV 2025) has no DOI in the biblio. palsson2024 pages come from IOS Press via T8 (Crossref lists none).

## 3. DOIs

Every DOI in 02_biblio_T*.md is present in refs.py (20/20). Four more found and added (above).

Online spot-check, 30 identifiers:
- Crossref, 20 DOIs, all resolve with matching title, first author, author count, venue, volume/pages: aas2021, achtibat2023, belinkov2022, bilodeau2024, chen2022, elazar2021, hammersborg2024, hewitt2019, huang2024, karvonen2024, mcgrath2022, nanda2023, palsson2024, ravfogel2020, ravichander2021, schut2025, shang2024, tomlin2022, tucker2021, zhou2018.
- Crossref search, 4 new DOIs: beckers2019, espinosa2023, geiger2020, ferrando2024.
- arXiv abstract pages, 6 IDs, title/authors/year match: lin2026 (2604.10158), sandmann2025 (2508.21380), paulo2025 (2501.18823), engels2025 (2410.14670, "Published in TMLR"), lomasov2025 (2510.26025), poupart2024 (2406.04028).
- arXiv export API returned "Rate exceeded" for batch queries; sutter2025 fetch blocked by a session limit. Not counted.

## 4. Claims in §4 against biblio annotations (41 checked)

| § | Claim | Key | Verdict |
|---|---|---|---|
| 4.1 | 26 methods classified by removal rule | covert2021 | OK |
| 4.1 | conditional/interventional/baseline Shapley differ under same axioms | sundararajan2020 | OK |
| 4.1 | argues interventional removal | janzing2020 | OK |
| 4.1 | off-manifold removal misleads | frye2021 | OK |
| 4.1 | true to model vs true to data | chen2020 | OK |
| 4.1 | both Shapley variants fail as importance | kumarIE2020 | OK |
| 4.1 | grouped Shapley for dependent features | aas2021 | OK |
| 4.1 | distributional baselines, internal-neuron IG | lundstrom2022 | OK |
| 4.1 | complete+linear no better than random; tie to end-task | bilodeau2024 | OK |
| 4.2 | irrelevant properties stay readable | ravichander2021 | OK |
| 4.2 | control tasks | hewitt2019 | OK |
| 4.2 | control tasks | belinkov2022 | OK (survey covers baselines/complexity) |
| 4.2 | high probe accuracy, small behavioural effect | elazar2021 | OK |
| 4.2 | interchange: readable not used | geiger2020 | OK |
| 4.2 | CBM gap | margeloiu2021 | OK |
| 4.2 | linear-probe accuracy unreliable; nonlinear probes + erasure | palsson2024 | OK |
| 4.2 | lc0 probe validated by causal effect | jenner2024 | OK |
| 4.2 | many correlational methods fail | abraham2022, huang2024, mueller2025 | OK |
| 4.3 | 65.9% unexplained, ResNet-50 | zhou2018 | OK |
| 4.3 | residual predictor (PCBM-h) | yuksekgonul2023 | OK |
| 4.3 | side channel | havasi2022 | OK |
| 4.3 | one error node per site | marks2025 | OK |
| 4.3 | error nodes per layer/position | ameisen2025 | OK |
| 4.3 | ~half SAE error linearly predictable; earlier-layer maps reduce it most | engels2025 | FIXED: the earlier-layer result concerns the nonlinear remainder, not total error (abstract confirmed). Now "linear maps from earlier-layer SAE outputs reduce the nonlinear remainder most." |
| 4.3 | residual directions matched to larger bank | shang2024 | OK |
| 4.3 | no canonical dictionary size | leask2025 | OK |
| 4.3 | names what vocabulary misses, behavioural validation | schut2025 | OK |
| 4.3 | missing vocabulary vs weak map | havasi2022 | OK |
| 4.4 | supervised attribute dictionaries as SAE yardstick | makelov2025 | OK |
| 4.4 | supervised directions match/beat SAEs (detection, steering) | kantamneni2025, wu2025 | OK (Kantamneni: detection only; AxBench: both) |
| 4.4 | DAS strongest, SAEs ≈ neurons | mueller2025 | OK |
| 4.5 | transcoders; skip variant | dunefsky2024, paulo2025 | OK |
| 4.5 | crosscoders; CLTs with error nodes | lindsey2024, ameisen2025 | OK |
| 4.5 | multi-layer SAEs, Jacobian SAEs | lawson2025, farnik2025 | OK |
| 4.5 | only concept-map method fusing layers | he2025 | OK |
| 4.5 | every MLP+attention replaced; EV >0.6, lower attention mid-layers; hand labels vs rule detectors | lin2026 | OK |
| 4.6 | any network abstracts any algorithm; near-perfect IIA on random nets | sutter2025 | OK |
| 4.6 | constructive abstraction not met | beckers2019 | OK |
| 4.6 | causal inner product | park2024 | OK |
| 4.6 | CAVs inconsistent across layers, entangled | nicolson2025 | OK |
| 4.6 | leakage diagnostics | mahinpei2021, espinosa2023, parisini2026, huang2024 | OK; mahinpei demonstrates leakage rather than defining a metric (minor, left) |
| 4.6 | ~32 concepts | ramaswamy2023 | OK |
| 4.7 | G×I conservation fix | ali2022 | OK |
| 4.7 | softmax rules; CRP | achtibat2024, achtibat2023 | OK |
| 4.7 | AtP* false negatives | kramar2024 | OK |
| 4.7 | attention frozen | ameisen2025 | OK |
| 4.7 | attention-head ablation | jenner2024 | OK |
| 4.7 | forward-only routes | ferrando2024 | OK |
| 4.8 | per-layer concepts, AlphaZero/lc0 | mcgrath2022, hammersborg2024 | OK |
| 4.8 | 85% → 50–65%, 270M, six concepts | lomasov2025 | OK per T8; flag: abstract ties "6 concepts" to the 240-position Chess960 set, so the layer-probe concept count may differ. Left for author. |
| 4.8 | late layers override mid-layer solutions | sandmann2025 | OK |

Uncited claim: "A 2026 follow-up, not verified here, contests the AxBench SAE result" (§4.4) matches T7 #15's note; correctly hedged.

## 5. PAPERS.md [53] — report only

PAPERS.md [53], Jiao, Eilender, Tang & Anderson, "Understanding Skill Adaptation in Transformers Using Sparse Autoencoders: Chess as a Model System" (OpenReview Wxl0JMgDoU), failed verification in T8: OpenReview blocked automated access, no search hit for the exact title, only a GitHub repo (CSSLab/maia2-skill-adaptation) with a different title. Not cited in this paper. PAPERS.md not edited; needs a manual check.

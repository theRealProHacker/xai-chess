# Context brief (shared by every agent in this pipeline)

## Research goal (user's words)
"How to do feature attribution through the decoder on the embed layer, through an encoder on any
layer, and focus on methods that combine several approaches (e.g. using several decoders on several
layers). Is to not just find the single one approach but several interesting ones. Put this in a
single paper."

## The system
- Model: Leela Chess Zero transformer `t1-256x10-distilled-swa-2432500`: 64 square tokens, d=256,
  10 post-LN encoder blocks (8 heads, smolgen), policy + value heads. ONNX via leela2onnx, run with
  onnxruntime on CPU (~180 pos/s). Taps: `/attn_body/rehape` (embedding), `/encoder{i}/ln1|ln2`,
  FFN internals, value head `/value/dense1`, `/value/dense2`.
- Human features: `features/square/` (Rust) emits a binary vector per square, 30,837 bits/square
  (occupant, pawn structure, x-rays, threats, routes, roles, mate/material lines from an engine,
  etc.); ~36.5 bits on per square; bits with <20 train positives dropped.
- Data: 20k positions (Lichess puzzle FENs), 80/20 split by position.

## Results so far (held-out)
1. Encoder/probe (token at layer L -> its per-square features): linear probe per layer;
   most families near-perfectly readable already at the embedding; some (pawn_cover, strong
   squares, mate, passed pawn) gain 0.04-0.14 by layer 5.
2. Decoder (features + square one-hot -> token at layer L), ridge, FVU:
   | tap | square only | features+square | features add |
   | embedding | 0.164 | 0.002 | 0.16 |
   | L0 | 0.575 | 0.183 | 0.39 |
   | L1 | 0.818 | 0.198 | 0.62 |
   | L2 | 0.828 | 0.195 | 0.63 |
   | L5 | 0.841 | 0.392 | 0.45 |
   | L9 | 0.897 | 0.680 | 0.22 |
   MLP decoder ~similar trend. So the embedding is (almost) a linear function of the features,
   and the features' explanatory power peaks around L1-L2 and decays with depth.
3. Board-level decoder from all 64 squares' features to value/dense1, dense2, value output:
   overfits; test FVU 0.52-0.62 at 151k positions, still falling with data.
   Value head applied to decoded layer-9 tokens: also weak.
4. Earlier: MLP probes for "pin is relevant" peak AUROC 0.945 (enc8) vs Stockfish NNUE 0.839.
   Causal intervention operators exist (patched Stockfish MaskPinner / LineBlocker), i.e. the
   project values interventional ground truth over correlational readouts.

## The user's open question (journey.md)
"How do we get the relevance/feature attribution from our decoders?
1. Start at the back: near-perfect reconstruction, but mostly square identity; attribution must
   fight through 'the deep jungle of attention' -> weak and trivial.
2. Start at the front (embedding): reconstruction worse deeper, but attribution clearer.
3. Therefore a hybrid that combines the results in some way?"
Concretely: the embedding E ≈ D(f) (decoder D from features f). Then e.g. gradient of the
value/policy w.r.t. E, chained through D, gives a per-feature attribution
(grad_E · ∂D/∂f, or integrated gradients / occlusion over feature bits). At deeper layers,
decoders explain less variance; the residual (unexplained part) is the "error node".

## Project values (honor these)
- The user is research-first, has strong domain expertise, pushes back on over-claiming;
  wants concrete examples, honest scoping, terse prose (no AI slop).
- VERY IMPORTANT to the user: human feature vocabularies miss concepts; measure what the
  vocabulary misses (the residual / error term), not only whether named features are faithful.
- Interventional (causal) validation is preferred over correlational probe accuracy.

## Output location
All pipeline artifacts go in `paper/attribution/` in the repo
(/home/rharvey/Dokumente/Programmieren/xai-chess/paper/attribution/). Existing bibliography for the
wider project: `PAPERS.md`, `paper/references.bib`, `RESEARCH_SUMMARIES.md` (reuse verified entries).

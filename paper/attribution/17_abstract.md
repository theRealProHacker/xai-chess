# Abstract (academic-paper Phase 5b, abstract_bilingual_agent)

Deviation from the agent spec, per 10_config.md: English abstract only; no zh-TW abstract or keywords. Composed independently of the abstract in paper.md.

## English Abstract

Explaining a network in chess terms needs a map from concepts to activations and a removal rule. For the Leela Chess Zero (lc0) transformer, ridge decoders from 30,837 rule-computed per-square features to each square token leave 0.002 of token variance unexplained at the embedding and 0.680 at the last block; linear probes read the features back. The question is which methods turn these maps into attributions of lc0's value and policy outputs, and what the vocabulary leaves unnamed. Only these fits, feature bit counts and the exported graph's structure were measured; methods, costs and outcomes are proposals from a survey of 89 works. (a) The embedding is local to each square, so decoder attribution there is input saliency in feature names, admissible only with a local vocabulary. (b) At later layers, patching the decoded change of a real board edit attributes that edit, a named/residual split measures coverage, and probe-side erasure and interchange test whether named content is used. (c) Layers combine as a per-layer sweep, an erasure cascade, feature paths with attention live, or a replacement graph with attention frozen. Only an edit's joint effect is identifiable; its split among collinear families is not. For complete pairwise methods, faithfulness error equals the residual share, so the residual is reported as coverage. The recommended first experiment runs decoded substitution and the named/residual split on about 1,000 pin edits at four layers, with random-initialisation and ridge-sweep controls, costing about one CPU hour; preset rules pick the next method from its outcome.

**Keywords**: concept-based explanation, activation patching, causal abstraction, Shapley values, identifiability, mechanistic interpretability, Leela Chess Zero

## Quality report

| Metric | Value |
|---|---|
| Word count | 250 words (target 150-250) |
| Components | 5/5: background (sentences 1-2), purpose (3), method with measured/proposed split (4), findings (5-9), implication / first experiment (10) |
| Keywords | 7; none repeats a title word ("attribution", "decoder", "probe", "vocabulary", "chess transformer") except the model name |
| Citations | none |
| Abbreviations | lc0 defined; FVU described, not abbreviated |
| Numbers | all match paper.md: 30,837 bits, FVU 0.002 / 0.680 (Table 2), 89 works (App. E), ~1,000 pin edits, 4 taps, ~1 CPU hour (Sec. 10.5) |

## Comparison with the abstract in paper.md

- Structure: paper.md uses bold run-in labels (Background/Purpose/Method/Answer/Recommendation). This version is one unstructured paragraph, the usual form for TMLR and NeurIPS workshops; the labels can be kept if the venue allows them.
- Measured vs proposed: paper.md states it in the final sentence ("Nothing beyond the existing fits has been run"). This version puts the split before the answer, so a reader knows the status of (a)-(c) while reading them, and lists what was measured (fits, bit counts, graph read).
- Opening: paper.md opens on the setup (vocabulary mapped to tokens). This version opens on the problem (a map alone is not an attribution; a removal rule is needed), which motivates the question.
- Method line: paper.md lists "nine-axis taxonomy" and "checks against the ONNX graph"; this version drops the taxonomy (not a result) and keeps the graph read as a measurement.
- (b): paper.md does not name the named/residual split as a coverage measure in (b); this version does, matching Section 2.2 (rho_L is coverage, not attribution).
- First experiment: this version adds the cost (one CPU hour) and that decision rules are preset; paper.md omits both.
- Keywords: paper.md includes "feature attribution" and "chess", which repeat title words, and "residual coverage", a term the paper coins. This version replaces them with "activation patching", "Shapley values" and "mechanistic interpretability", which are established indexing terms.
- Length: paper.md's abstract is 240 words including labels; this one is 250.

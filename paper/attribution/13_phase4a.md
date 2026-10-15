## Acceptance Criteria Paraphrase

Contract: `academic_paper/writer_full/v1` (mode `writer_full`, stage `writer_full_drafting`, baseline v3.6.6). Paper: "Attributing a chess transformer's decisions to human features: decoders, encoders, and their combination across layers"; field: ML interpretability; target length ~9,500 words main text + ~4,400 words appendix. All seven dimensions are paraphrased below (minimum_dimensions = "all").

### D1: section_completeness
(Mandatory.) I will write every section the Structure Architect's approved outline lists, for both the main text and the appendix, in the outline's order. No section stays a stub or heading only. No `[MATERIAL GAP]` placeholder may remain when the draft goes to citation_compliance. If material for a section is missing, I write the section from what is available, narrow its claims to fit, and record the narrowing explicitly. I do not leave a placeholder.

### D2: citation_density
(Mandatory.) Every factual claim carries at least one citation. This covers prior interpretability methods (probing, decoders/encoders, attribution), chess-engine and transformer facts, and reported results from other work. Claims about this paper's own experiments point to the relevant section, table, figure or appendix. When I cannot find a source for a claim, I either add a source from the assigned bibliography or rewrite the claim with appropriate hedging. No unsupported factual assertion is left.

### D3: argument_blueprint_fidelity
(Mandatory.) Each section follows the Claim–Evidence–Reasoning chains in the Argument Blueprint from argument_builder_agent. The Blueprint is the source of these chains; the contract is not. A section's topic sentences state the claim the Blueprint assigns to it. Its evidence cites the sources the Blueprint assigns. Its explanation ties the evidence back to the section's stated purpose. I do not add claims the Blueprint does not support, and I do not quietly drop chains it requires.

### D4: total_word_count
(High priority.) The whole draft must land within ±10% of the target in the Paper Configuration Record. Using the metadata here (~9,500 main text + ~4,400 appendix), main text should fall at about 8,550–10,450 words and the appendix at about 3,960–4,840 words. If the Configuration Record defines the target differently, the Record wins. I will count words before handoff and trim or expand to stay inside the band.

### D5: per_section_word_count
(High priority.) Each section must land within ±15% of its word allocation in the Paper Configuration Record. I will check every section against its allocation and fix any section that falls outside. I will not offset an overlong section by shortening a different one.

### D6: paragraph_structure
(Normal priority.) At least 80% of body paragraphs follow TEEL: a topic sentence, then evidence (a citation or one of this paper's own results), then an explanation of what the evidence shows, then a link to the section's argument or to the next paragraph. The opening paragraph of the introduction and the closing paragraph of the conclusion are exempt. Every section boundary gets an explicit transition that tells the reader how the next section follows from the previous one.

### D7: register_consistency
(Normal priority.) The prose stays in an academic register throughout, with no colloquial or informal wording. Phrasing should suit ML interpretability as the Paper Configuration Record specifies. Terms such as probe, decoder, encoder, layer, attribution and feature are defined once and used the same way every time after that. Chess vocabulary is used precisely and introduced where a non-chess ML reader needs it.

[PRE-COMMITMENT-ACKNOWLEDGED]

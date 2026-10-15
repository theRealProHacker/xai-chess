## Contract Paraphrase

Contract: `academic_paper/evaluator_full/v1` (stage `evaluator_full_evaluation`, baseline v3.6.6). Paper: "Attributing a chess transformer's decisions to human features: decoders, encoders, and their combination across layers"; field ML interpretability; ~9,500 words main + ~4,400 appendix; type methods survey + position paper with an unrun preregistered protocol. All five dimensions are paraphrased (paraphrase_minimum_dimensions = "all").

Writer pre-commitment check (`check_writer_artifact: pre_commitment_artifacts`): the writer's `<phase4a_output>` paraphrases all seven `writer_full/v1` dimensions (D1 section_completeness through D7 register_consistency) and ends with `[PRE-COMMITMENT-ACKNOWLEDGED]`. Two commitments bear on my scoring. Writer D2 promises that claims about the paper's own results point to a section, table, figure or appendix, and that unsourced claims are hedged rather than left bare. Writer D3 promises that each section follows the Argument Blueprint without adding or dropping claims. Writer D7 promises that probe, decoder, encoder, layer, attribution and feature are defined once and used consistently. In Phase 6b I will check the draft against these promises under my D3, D2 and D5/D4 respectively. Writer D4 gives word bands of 8,550–10,450 (main) and 3,960–4,840 (appendix); I will note a breach but will not score length as a separate dimension. Writer D6's TEEL and transition commitment feeds my D4. One deviation: the artifact's top heading reads "Acceptance Criteria Paraphrase", not "Contract Paraphrase". This is a naming difference only. Every dimension is covered, so it is not structural drift.

### D1: originality
(High priority.) The paper has to add something that existing interpretability work does not already provide. For a survey and position paper, a set of method summaries does not count. What counts is a framework that organises decoders, encoders and their combination across layers in a way prior surveys do not; a clearly argued position; or a protocol that would test something no published work has tested. The introduction must name the gap. The discussion must engage closely with the nearest prior work (chess-model probing, concept and attribution methods), not cite it in a list.

### D2: methodological_rigor
(Mandatory.) The design has to fit the questions asked. For the survey part, that means a stated basis for which methods were included and how they were compared. For the preregistered protocol, it means hypotheses, model and layers, feature sets, metrics, controls, baselines and decision rules, specified closely enough that a third party could run it unchanged. Because the protocol has not been run, the paper must say so clearly and must not describe it as validated. Limitations must be stated and discussed, including confounds specific to decoder/encoder attribution such as probe capacity, selectivity and correlated features.

### D3: evidence_sufficiency
(Mandatory.) Every factual claim needs support. Claims about prior methods and models need citations. Claims about the paper's own existing results need a pointer to a table, figure or appendix. Predictions about the unrun protocol must be labelled as predictions. The cited evidence must be appropriate and must not come mostly from one source. The paper must not cherry-pick: if it cites results that favour its position, it must also acknowledge known contrary or null findings. Because no new experiment backs the position, the main risk here is inferences that go beyond what the cited work shows.

### D4: argument_coherence
(High priority.) The argument must run consistently from the abstract through the introduction to the conclusion. The thesis should be stated the same way in each. The research questions or position posed in the introduction must be answered or explicitly carried forward in the conclusion. The survey sections must lead logically into the position and then into the protocol, with each section boundary explaining how one part follows from the last. No two passages may contradict each other, for example by hedging a claim in one section and asserting it in another.

### D5: writing_quality
(Normal priority.) The prose must be clear, precise and academic in register, at a level suited to an ML interpretability venue. Grammar errors should be rare and citation style consistent. Every figure and table must be referenced in the text. Technical terms must be used consistently once defined, and chess terms must be explained enough for an ML reader without chess background. Paragraphs must not be so long that the argument becomes hard to follow.

## Scoring Plan

### D1: originality
dimension_id: D1
what_to_look_for: The introduction names a specific gap, such as the absence of a joint decoder+encoder attribution account across layers of a chess transformer, and the discussion explicitly contrasts the paper's framework, position or protocol with the closest prior chess-probing and attribution work.
what_triggers_block: There is no or only a vague contribution statement, the survey restates known methods without a new organising framework or argued position, or prior work is listed without being engaged.
what_triggers_warn: A contribution is stated, but it is only weakly distinguished from the closest cited work, or engagement is deep for some prior sources and only a list citation for others.

### D2: methodological_rigor
dimension_id: D2
what_to_look_for: The survey states its inclusion and comparison basis, and the preregistered protocol specifies hypotheses, model and layers, feature sets, metrics, controls and baselines, and decision rules in enough detail to run it unchanged; limitations, including probe-capacity and correlated-feature confounds and the protocol's unrun status, are stated and discussed.
what_triggers_block: The protocol cannot be replicated from the text, its design does not test the stated hypotheses, the unrun protocol is presented as if it had produced results, or limitations are missing.
what_triggers_warn: The design is sound but some protocol parameters or survey-selection choices are underspecified or unjustified, or limitations are listed without discussion of how they affect the conclusions.

### D3: evidence_sufficiency
dimension_id: D3
what_to_look_for: Every claim about prior work carries a citation, every claim about the paper's own results points to a section, table, figure or appendix, and protocol predictions are labelled as predictions rather than findings.
what_triggers_block: Several core claims lack any citation or internal pointer, the argument relies heavily on a single source, position claims go beyond the cited evidence, or contrary and null findings are left out in a way that favours the thesis.
what_triggers_warn: Core claims are supported, but some secondary claims lack support, or citation density is clearly uneven between the survey, position and protocol sections.

### D4: argument_coherence
dimension_id: D4
what_to_look_for: The thesis is worded consistently across the abstract, introduction and conclusion, the conclusion answers or explicitly defers each question posed in the introduction, and each section boundary carries an explicit transition from survey to position to protocol.
what_triggers_block: The conclusion does not address the introduction's questions, there are several unexplained jumps between sections, or two passages contradict each other on the same claim.
what_triggers_warn: The overall logic holds, but some transitions are abrupt or forced, or a claim's strength or hedging changes slightly between sections.

### D5: writing_quality
dimension_id: D5
what_to_look_for: The register is consistently academic, grammar errors are rare, citation format is the same throughout, every figure and table is referenced in the text, and defined terms (probe, decoder, encoder, attribution, feature) and chess vocabulary are used consistently.
what_triggers_block: The prose is hard to follow, grammar errors are numerous, the register is colloquial, or citation format varies across the manuscript.
what_triggers_warn: The prose is readable but has several grammar or word-choice problems, some paragraphs are too long, there are minor citation-format or term-usage inconsistencies, or a figure or table is not referenced.

[PRE-COMMITMENT-ACKNOWLEDGED]

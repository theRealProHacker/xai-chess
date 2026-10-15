# Copy-edit pass after evaluator round 2

Source edited: `src/part1-4.md`; `paper.md` rebuilt (89 refs, 0 unused).

- N1: unit is "pin-edit pairs" throughout (abstract, §2.5, §10.5, Table 4, §13, App. A). §10.5: about 500 pins, two edits each, about 1,000 pairs. Cost unchanged (already assumed 1,000 pairs). App. A: Gram matrix is tap- and network-independent, about seven factorisations, the hour is an upper bound. §10.5 solve dimension 14,565 → 14,629 to match App. A.
- N2: §10.5 defines a1 = 0, h8 = 63 (python-chess). Both edits use "a move that would be legal for its side if it were to move". Order: minor first, then nearest, then lower index. Checked with python-chess on the worked example: Bb5 off-line candidates c4 (26) and a6 (40) at distance 1, so Bc4. Black pseudo-legal moves to d7: Ke8, Qd8, Bc8, so minor-first gives …Bd7.
- N3: rank stability = per-tap ranks of mean |Δ_G| over pairs, min pairwise Spearman over five α.
- N4: round-trip calibration added in §8.3 and App. A M8∘M2.
- N5: H1 drops M1c from the zeroed-delta clause and adds sign/rank agreement with M6 under C2.
- N6: Table 5 row `square.control` → `square`.
- N7: §11.1 L2 total R² 0.81. Table 1 value-head row: 3,960 held-out of 20,000.
- N8: M1c "zeroing" → "mean occlusion" (§6.1, App. A); §6.2 "zeroing deltas". M1c residual defined as g_e(q_e) − g_e(q_e + ε_e) in Table A, Table C1 and App. A.
- N9: M4 whitening uses (Σ^{1/2})^+, with a reason.
- §11.3: dangling "third reason" rewritten as a separate point.
- §12: one paragraph with the acknowledged limitations (sample unit and cost scaling, edit conventions, aggregated rank stability, uncalibrated remainder as upper bound, M1c by agreement only, pseudo-inverse). Label and rounding nits fixed in text, so they are not listed there.

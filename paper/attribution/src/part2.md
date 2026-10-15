## 6. (a) Attribution Through the Embedding Decoder

### 6.1 Three forms of M1

The working note proposes chaining the output gradient through the decoder, $\nabla_E y\cdot\partial D/\partial f$, and names integrated gradients and occlusion over feature bits as alternatives. These give three sub-methods with different removal rules.

*M1a, point gradient.* Multiplying the chained gradient by $\tilde f$ gives

$$A^{(e)}_G(x)=\sum_s\big\langle\nabla_{h_{e,s}}y,\ n^G_{e,s}\big\rangle,\qquad A^{(e)}_\varepsilon=\sum_s\big\langle\nabla_{h_{e,s}}y,\ \varepsilon_{e,s}\big\rangle.$$

It is first-order on binary features and leaves $\langle\nabla y,q\rangle$ unassigned, so it is a sensitivity map, not a complete attribution.

*M1b, board-path IG.* IG [@sundararajan2017] runs in activation space between two real boards, with a real edit of $x$ or an average over real positions as baseline [@lundstrom2022]. Because $D_e$ is linear, the integrand splits over families, and $\sum_G\mathrm{IG}_G+\mathrm{IG}_\varepsilon=y(x)-y(x_0)$ holds exactly (Appendix A, M1).

*M1c, feature occlusion.* Family $G$'s decoded share on each square is replaced by a baseline and the network is run forward:

$$h^{\rm occ}_{e,s}=h_{e,s}+W^G_e\big(b^G_s-\tilde f^G_s\big),\qquad O_G(x)=g_e(h^{\rm occ}_e)-y(x).$$

Mean occlusion sets $b^G_s=0$, the per-square mean. Bit flipping sets $b^G_s$ to $\tilde f^G_s$ with chosen bits inverted. This is M9's value function with a synthetic baseline, and it runs under ONNX at one forward pass per family. Both baselines produce feature vectors no board produces, often ones that break the extractor's logical constraints, such as a route bit without the piece that makes the route. M1c is therefore off-manifold [@frye2021], [@kumarIE2020]. Its on-manifold counterpart is a real board edit, which changes only bits a board can change but changes several families at once. At $e$ that counterpart is M2 and M6 under the local vocabulary (Section 7.2).

All three forms reduce to the same object, because $h_{e,s}$ is a function of square $s$'s planes alone (Section 5.2) and $D_e$ re-expresses those planes in the vocabulary. M1a and M1b are input saliency on each square's planes, and M1c is occlusion of those planes along a decoder column, renamed into features. Occluding a non-local family moves $h_{e,s}$ along a direction the embedding produces only through a change in the square's own planes, so it asks how lc0 responds to a plane change that ridge associates with the family. The renaming is useful for reading the result. It does not change what is measured.

### 6.2 Admissibility

The vocabulary must be local (C2), or decoded deltas must be zeroed on squares whose planes did not change. Otherwise non-local families collect spurious credit that $\varepsilon_e$ cancels (Section 5.2, consequence 3). M1c has no second board, so zeroing deltas does not apply. Occluding a non-local family is a valid intervention, read as in Section 6.1 rather than as removing that family. M1a and M1b also need the gradient to survive ten post-LN blocks. Ali et al. [@ali2022] show that Gradient×Input violates conservation through attention and LayerNorm, so M1a is a sensitivity map at best. M1b is complete, but its path crosses the whole network, and completeness is a property of the integral, not evidence of faithfulness. Both need a PyTorch port, verified against the ONNX graph, and cost one backward pass per position (M1a) or about 32 (M1b). M1c avoids the gradient and the port, at the price of off-manifold inputs. It costs $m+1$ forward passes per position for family occlusion.

### 6.3 What it can and cannot say

M1 answers which of this square's inputs, named in the vocabulary, move the output. It cannot say what lc0 computed from them, because at the embedding nothing has been computed across squares yet. Its residual collects broadcast-plane content and the 0.2% misfit. Under C2 the named share is predicted to be near 1 by construction (Section 5.2, consequence 4), so a high front-end coverage would not be a finding about lc0. Saliency maps can look plausible on random networks [@adebayo2018], so C1 is mandatory here too.

H1 (Section 10.3) states the testable part. The finding is negative for the front as an attribution site. The working note's expectation that attribution is "clearer" there holds only in that it is local. Attribution that says something about lc0's computation needs the later taps of Section 7.

## 7. (b) Attribution Through Decoders and Encoders at Any Layer

Methods at a single tap $L$ run in two directions. The decoder direction asks how much of an output change the named content carries and through which families. The encoder direction asks whether readable content is used. Both are needed at the same tap, because readable is not used (Section 4.2). Table A summarises them, and Appendix A gives full definitions.

### 7.1 The correlational floor: M0

M0 fits a surrogate $s(\phi(x))\approx y(x)$ on all 64 squares' features, or $s(P_L(h_L))$ on probe outputs with a residual predictor as in post-hoc concept bottlenecks [@yuksekgonul2023]. It attributes $s$ with Shapley values over families [@aas2021]. M0 answers which families *predict* $y$, true to the data rather than to lc0 [@chen2020]. Its first stage exists: the board-level decoder of Table 1 (test FVU 0.52–0.62). Engine-derived families predict value for trivial reasons [@kumarA2022], [@parisini2026], so C7 (with and without them) is mandatory. M0 is the floor: an interventional method that localises edits no better than M0 adds nothing.

### 7.2 Decoder direction: M2, M6, M9

**M2, decoded substitution.** For a real edit and family $G$, M2 adds $W^G_L(\tilde f^G_s(x')-\tilde f^G_s(x))$ to $h_{L,s}(x)$ on the squares where $f^G$ changed and runs the suffix $g_L$. Patching all changed families jointly gives $\Delta^{(L)}_{\rm named}$, and the consistency gap

$$\gamma_L=\Delta^{\rm true}-\Delta^{(L)}_{\rm named}$$

measures whether the decoded edit commutes with the real one, a causal-consistency check [@rubenstein2017]. The design follows supervised dictionary edits [@makelov2025], concept interventions [@koh2020], [@shin2023] and in-distribution corruption [@zhang2024]. Decoded edits are off-distribution and can open dormant pathways [@makelov2024], and single-family patches do not add when families interact like OR gates [@heimersheim2024]. The joint patch is identifiable, but single-family patches are not (Section 9.1). M2 needs $2k+1$ suffix passes per tap for $k$ changed families.

**M6, named/residual split.** M6 treats $n_L$ and $\varepsilon_L$ as two players on an oriented pair. The exact two-player Shapley value needs four suffix evaluations, two of which are $y(x)$ and $y(x')$:

$$\phi^{(L)}_n=\tfrac12\Big[g_L(q+n'+\varepsilon)-g_L(q+n+\varepsilon)+g_L(q+n'+\varepsilon')-g_L(q+n+\varepsilon')\Big],\qquad \phi^{(L)}_\varepsilon=\Delta^{\rm true}-\phi^{(L)}_n.$$

The first difference is M2's joint patch, so M6 is M2 symmetrised over the two residual contexts. Completeness is exact because $g_L(q+n'+\varepsilon')=y(x')$. Swapping $x$ and $x'$ negates $\phi_n$, so on oriented pairs the named share

$$\rho_L=\frac{\sum_{\rm pairs}\phi^{(L)}_n}{\sum_{\rm pairs}\Delta^{\rm true}}$$

is well defined. Without orientation the denominator $\sum\Delta^{\rm true}$ can cancel to near zero. Orientation makes it $\sum|\Delta^{\rm true}|$ (Appendix A, M6), and pairs with $|\Delta^{\rm true}|\le\tau$ are dropped (Section 10.5). This is baseline Shapley [@sundararajan2020] with the error node as a player [@marks2025]. $\rho_L$ has two players and never says which family moved the output. It is a coverage measure and a guide: per-family attribution (M2, M9) is worth running where $\rho_L$ is high, above C1 and stable under C3. Its uncertainty is a bootstrap interval over pairs. M6 costs one suffix pass per tap per pair beyond M2.

**M9, family Shapley with an error player.** Players are the families and $\varepsilon$. *Pairwise* M9 explains $x'$ against the single baseline $x$, so its values sum to $\Delta^{\rm true}$ and unchanged families receive zero by the dummy axiom. *Distributional* M9 uses a pool matched on material and phase and explains $y(x)-\mathbb E\,y(x_0)$, not an edit. Grouping follows Aas et al. [@aas2021], the value function baseline Shapley [@sundararajan2020] with distributional baselines [@lundstrom2022], and 28 family players plus $\varepsilon$ respect the cap of Ramaswamy et al. [@ramaswamy2023]. Mixed coalitions combine families from different boards, so M9 is true to lc0 composed with $D_L$ [@chen2020] and leaves the data manifold [@kumarIE2020]. Permutation sampling costs about 30 s per position per tap under ONNX. A relevance chain (M10) gives the same split from one backward pass with DeepLIFT rules [@shrikumar2017] chained as in G-DeepSHAP [@chen2022], untested through smolgen and post-LN.

### 7.3 Encoder direction: M3, M4, M5

**M3, LEACE erasure.** LEACE [@belrose2023] gives the least-squares-optimal linear erasure of family $G$ at $L$ in closed form. Applying it on all 64 squares and running the suffix gives $\Delta^{\rm erase}_{G,L}$, reported beside the probe AUROC of $G$ at $L$. The pair forms a use-versus-readability curve, after amnesic probing [@elazar2021], INLP [@ravfogel2020] and R-LACE [@ravfogel2022], and the chess case for erasure [@palsson2024]. M3 measures the use of $G$ together with its linear correlates, which makes it identifiable. Recomputation from other squares hides use, and M8 measures it (Section 8.2).

**M4, concept gradient.** With a family-score probe and its Jacobian $J_s$, the concept gradient [@bai2023] projects $\nabla_{h_{L,s}}y$ onto family directions and leaves an unexplained part $\rho^\perp$. TCAV is the linear special case [@kim2018]. M4 must work at family level ($m=28<256$). At bit level $J_s^\top$ is surjective, $\rho^\perp\equiv0$ by construction, and the minimum-norm split is arbitrary. The inner product should be whitened, with gradient and probe Jacobian transformed by the same matrix (Appendix A) [@park2024], and the pseudo-inverse is unstable for collinear families [@nicolson2025]. Probe directions need not be the model's [@tucker2021].

**M5, interchange alignment.** DAS [@geiger2024], in its boundless form [@wu2023], learns a rotation $R_L$ and subspace size $k_G$ such that swapping the subspace from a source board reproduces the source's output. Graded IIA scores the swap, and RAVEL's Isolate score requires that pairs differing only in a correlated family leave the output unchanged [@huang2024]. The theory is causal abstraction [@geiger2021], [@geiger2025]. The decisive control is C1, given near-perfect IIA on random networks [@sutter2025]. Subspace patches can produce dormant-pathway illusions [@makelov2024], though the criterion also flags valid solutions [@wu2024], so C8 is a diagnostic only.

### 7.4 Policy

The policy head reads `/encoder9/ln2`. For a non-promotion move $a\to b$, the ONNX graph computes

$$\ell_{a\to b}=c\,\big\langle Q\,m(h_a)+b_Q,\ K\,m(h_b)+b_K\big\rangle,\qquad m(h)=\mathrm{mish}(W_1h+b_1),$$

with $m$ applied per token. The logit depends on two tokens only, so a decoder split at this tap gives an exact four-player attribution over $n_a,\varepsilon_a,n_b,\varepsilon_b$ from 16 head evaluations. That needs a decoder at `/encoder9/ln2`. The existing fits are at ln1. At earlier taps a policy logit is a target like value for M2, M6, M8∘M2 and M9, with credit on families at $a$ or $b$ separated from credit elsewhere. Moves must be legal on both boards. $\rho_L$ for value and policy on the same pairs tests cause (f) of Section 5.4.

Each of these methods reads one tap. Section 8 combines taps.

## 8. (c) Combining Decoders and Encoders Across Layers

Four compositions answer the working note's request for a hybrid (Figure 2). They differ in what they hold fixed and what they cost, and none needs a supervised cross-layer dictionary, which no source provides (Section 4.5).

![Figure 2](fig_compositions.png)

**Figure 2.** The four cross-layer compositions on the tap axis, with an intervention at L2 as example. Filled diamonds mark interventions, open circles re-reads, squares decoder reads or graph nodes. Schematic without data, script `fig_compositions.py`.

### 8.1 Sweep: M6 at every tap

M6 run at every tap on the same oriented pairs gives the coverage curve $\rho_L$ and its differences $\kappa_L=\rho_L-\rho_{L+1}$. $\kappa_L$ is not a flow through block $L$: the decoders are fitted independently, attention mixes squares, LayerNorm renormalises, and $\kappa_L$ moves with decoder quality. A tied decoder $W_L=W+\Delta_L$ with a penalty on $\Delta_L$ would make the decoders share structure, a supervised analogue of crosscoders [@lindsey2024], [@lawson2025]. The sweep is the cheapest composition, and its value lies in the shape of $\rho_L$ against the C1 curve.

### 8.2 Intervene and read: M8

M8 erases (M3) or substitutes (M2) family $G$ at $L$ and reads every later tap $L'>L$ through probes or decoders. The transfer tensor $T^{L\to L'}_{G\to G'}$ (Appendix A) records how the probe score of $G'$ at $L'$ changes. Its diagonal measures recomputation, and the change in $\phi^{(L')}_\varepsilon$ measures whether $G$ feeds unnamed computation. It builds on amnesic probing [@elazar2021], LEACE at every layer [@belrose2023], earlier-layer maps of later error [@engels2025] and backup behaviour [@heimersheim2024], with a block-Jacobian variant [@farnik2025]. The extractor's own dependency graph confounds transfer and is reported beside it. One patched forward pass per $(G,L)$ gives all later taps under ONNX.

### 8.3 Feature paths with attention live: M8∘M2

For a real edit, M8∘M2 applies M2's patch of $G$ at $L$ and runs the unfrozen network to $L'$. The change at $L'$ is split through features: patched and unpatched tokens are read by the probe, $\hat f=P_{L'}(h_{L'})$, and the decoded difference $W^{G'}_{L'}(\hat f^{G'}_{\rm patch}-\hat f^{G'})$ is family $G'$'s named change. The remainder $r_{L'}$ is the part of the change that the probe-decoder round trip does not reproduce. Regressing the change on $D_{L'}$'s columns instead would be degenerate. The 14,565 columns span $\mathbb R^{256}$, so $r_{L'}$ would vanish as the ridge penalty shrinks. Because the round trip is not the identity, $r_{L'}$ also holds round-trip error. As calibration, the same round trip is applied to the real edit's change $h_{L'}(x')-h_{L'}(x)$ and its remainder reported beside the exact split $\varepsilon'_{L'}-\varepsilon_{L'}$ from true bits. Only the excess of $\pi_{G\to r}$ over this floor is read as content leaving the vocabulary. Each part is then pushed to the output, giving path scores $\pi_{G\to G'}$ and a residual branch $\pi_{G\to r}$. This is path patching [@goldowsky2023] with feature coordinates at both ends. It answers through which named families at $L'$ the effect of $G$ at $L$ reaches $y$, and how much of it leaves the vocabulary between $L$ and $L'$. Path scores are not additive, so their sum is reported against $\Delta^{(L)}_G$. The split inherits the collinearity of Section 9.1. A pin pair with $L\in\{2,5\}$ and $L'\in\{5,9\}$ costs a few hundred passes under ONNX.

### 8.4 Replacement graph with attention frozen: M7

M7 freezes attention patterns and LayerNorm denominators per position [@ali2022], [@ameisen2025] and replaces each feed-forward block with a supervised skip transcoder that reads features through the probe, $\mathrm{FFN}_L(u)\approx S_Lu+V_L\sigma(P_L(u))+b_L$, with an error node per block and square. Given frozen attention, LayerNorm and feature activations, edges between nodes are linear [@ameisen2025]. The network itself stays nonlinear through $\sigma$. The parts come from transcoders [@dunefsky2024], [@paulo2025] and sparse feature circuits [@marks2025], and Lin et al. [@lin2026] are the unsupervised precedent on lc0. Reading features through $P_L$ matters, because a transcoder fed extractor bits can fit even if lc0 never computes them [@mahinpei2021]. M7's weakness is attention. Frozen query-key products and smolgen biases hide routing, which Jenner et al. [@jenner2024] show matters for look-ahead, so a frozen graph may credit routing to whichever node wrote the values. AtP*-style recomputation bounds what freezing hides [@kramar2024], and M8∘M2 is the live-attention alternative.

### 8.5 The residual across layers: M12 and M11

M12 regresses $\varepsilon_{L,s}$ on nested predictors: a nonlinear function of $f_s$ (cause b), features of attended squares (cause a), and $\varepsilon_{L-1,s}$ (carried forward), with Shapley increments over the three blocks. This follows Engels et al.'s [@engels2025] decomposition of SAE error. What remains, $\eta_L$, gets a size-swept sparse code whose latents join M9 or M7 as players [@kulkarni2026]. Naming latents against held-back candidate features follows Karvonen et al. [@karvonen2024], with a contrastive variant on board pairs [@poupart2024]. Latents depend on dictionary size [@leask2025] and naming works only if the candidate bank contains the concept [@shang2024]. M11 compares family attributions across methods and taps by rank, sign and coverage. A learned ensemble that matches $\sum_GA_G$ to $\Delta^{\rm true}$ is unidentified: with $\varepsilon$ included, every complete method has zero loss. GCAV's learned fusion [@he2025] shares the problem, so M11 compares and does not fuse.

The working note's hybrid is therefore four objects, two of which (M8 and M8∘M2) run under ONNX. All of them inherit the decoder's collinearity, which Section 9 addresses.

## 9. Identifiability and the Residual as Coverage

### 9.1 What is identifiable

Every real edit that changes a pin also changes occupancy-derived families: interposing a bishop in the worked example changes 331 bits on 21 squares, 180 of them route bits (Table 5). Ridge splits $W_L$ among collinear families by its penalty, so single-family credit from M1, M2, M4, M9 and M8∘M2 is partly an artefact of $\alpha$. C3 detects this but cannot remove it.

*Identifiable* given lc0 and a fixed edit: $\Delta^{\rm true}$. Given $D_L$ as well: the joint named patch $\Delta^{(L)}_{\rm named}$, $\gamma_L$, $\phi^{(L)}_n$, $\phi^{(L)}_\varepsilon$ and $\rho_L$. These depend on the decoder's prediction $W_L\Delta\tilde f$, which is far less sensitive to $\alpha$ than its per-family coefficients but still moves with decoder fit. C3 and C4 report how much. The LEACE effect of erasing $G$ at $L$ (M3) is also identifiable, read as the effect of $G$ together with its linear correlates.

*Not identifiable*: the split of a joint effect among collinear families. That covers single-family and leave-one-out patches, per-family Shapley values, concept-gradient coordinates and per-family path scores. Theory points the same way. With unrestricted maps any network abstracts any algorithm [@sutter2025], constructive abstraction requires conditions this vocabulary does not meet [@beckers2019], and probe and decoder directions agree only under a causal inner product [@park2024], [@nicolson2025].

Route bits dominate both edits (Table 5), so the extractor's encoding of geometry sets the scale of the problem. The remedy is to attribute to the edit and describe the edit by the families it changes. Per-family credit is reported only where C3 ranks are stable. Otherwise unstable families are merged into one player [@aas2021] until ranks stabilise, and the merge is stated. "Pin credit" means the credit of the families a pin edit changes, since the vocabulary has no pin bit.

### 9.2 Faithfulness error and residual share coincide

A standard faithfulness score compares the attribution summed over changed families $\delta$ with the true effect, in the style of CEBaB's ICaCE error [@abraham2022]:

$$\mathrm{err}=\frac{\mathbb E\big|\sum_{G\in\delta}A_G-\Delta^{\rm true}\big|}{\mathbb E|\Delta^{\rm true}|}.$$

If family values and residual sum to $\Delta^{\rm true}$ and unchanged families get zero, then $\sum_{G\in\delta}A_G-\Delta^{\rm true}=-A_\varepsilon$ on every pair, so $\mathrm{err}=\mathbb E|A_\varepsilon|/\mathbb E|\Delta^{\rm true}|$. This one-line identity covers M6, pairwise M9 and M1b with a real-edit baseline. Ranking these methods or layers by err would favour whichever pushes most credit into named families, the opposite of measuring what the vocabulary misses.

### 9.3 One coverage definition

For any method that splits a pair's effect into a named part $A_{\rm named}$ and a residual, over oriented pairs,

$$\mathrm{Cov}=\frac{\sum_{\rm pairs}A_{\rm named}}{\sum_{\rm pairs}\Delta^{\rm true}}.$$

M6 gives $A_{\rm named}=\phi^{(L)}_n$ and $\mathrm{Cov}=\rho_L$. M2's joint patch gives $\mathrm{Cov}=1-\sum\gamma_L/\sum\Delta^{\rm true}$. Pairwise M9 gives $\sum_G\varphi_G$, and M0 gives $s(x')-s(x)$. M11 compares these on one axis. Distributional M9 explains $y(x)-\mathbb E\,y(x_0)$, not an edit, so its residual share answers a different question and is never plotted with Cov. Coverage is a finding about lc0 and the vocabulary, reported per layer, per family where identifiable, and with and without engine-derived families. Complete pairwise methods are graded instead by $\gamma_L$, by control gaps against C1, C2 and C5, and by stability under C3. Error applies where an attribution can be wrong without the residual absorbing the difference: M0, M1a, M1c, M4, distributional M9, M10, and per-family M2. Single M2 patches need not sum to the joint patch when families interact [@heimersheim2024], so per-family M2 also reports the additivity gap $\sum_G\Delta^{(L)}_G-\Delta^{(L)}_{\rm named}$.

### 9.4 Three residual measures

$\phi^{(L)}_\varepsilon$ and $\varphi_\varepsilon$ say how much of a real output change runs through unnamed content at $L$. The M12 budget says why: non-locality, map class, carry-over or remainder. The M12 dictionary step asks whether the remainder holds stable, used units that a new extractor bit could name. Havasi et al.'s [@havasi2022] distinction between missing vocabulary and weak maps is the reason the budget comes first. Residuals are routinely reported as error nodes [@marks2025], [@ameisen2025] or residual predictors [@yuksekgonul2023], [@zhou2018]. Concepts lc0 has that humans have no word for would sit in $\eta_L$ at layers where coverage is low and stable. Schut et al. [@schut2025] are the closest precedent for naming such content, and naming depends on the dictionary and candidate bank [@leask2025], [@shang2024]. The protocol must therefore admit "nothing nameable" and coverage near zero at depth as outcomes.

With the grading rules fixed, Section 10 gives the protocol and the first run.

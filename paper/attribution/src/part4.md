## Appendix A. Method Cards

Each card gives the definition, question answered, failure modes, controls (Appendix B), cost and residual. Notation follows Section 3.2: $n^G_L$ is family $G$'s decoded share of the token at tap $L$, $\varepsilon_L$ the residual, $g_L$ the suffix network, $\Delta^{\rm true}$ the output change of an oriented real edit, and $P_L$ a probe.

Costs assume 180 forward passes per second under ONNX, a suffix from tap $L$ costing about $(10-L)/10$ of a pass, and a backward pass in a PyTorch port costing two to three forward passes. All costs are estimates.

**M0. Feature surrogate with family Shapley and a residual.** Fit $s(\phi(x))\approx y(x)$ on all 64 squares' features, or $s(P_L(h_L))$ on probe outputs at tap $L$. In the second case add a residual predictor $r(h_L)$ fitted to $y-s$, as in post-hoc concept bottlenecks with a residual (PCBM-h) [@yuksekgonul2023]. Attribute $s$ with Shapley values over families [@aas2021]. On a pair, report $A^{\rm M0}_G(x')-A^{\rm M0}_G(x)$ beside $\Delta^{\rm true}$. *Question:* which families predict $y$, true to the data rather than to lc0 [@chen2020]. *Failure modes:* engine-derived families predict value for trivial reasons; collinear families share credit by the surrogate's regulariser. *Implementation:* the surrogate class is fixed in advance, a ridge model for exact linear Shapley values or a gradient-boosted model with sampled group Shapley values, and the same class is used at every tap so that floors are comparable. *Controls:* C7 (mandatory), C3. *Cost:* minutes, no lc0 passes beyond $y$. *Residual:* $\mathbb E(y-s)^2/\mathrm{Var}(y)$, and on pairs the coverage of Section 9.3 with $A_{\rm named}=s(x')-s(x)$.

**M1. Decoder attribution at the embedding, three forms.** M1a, point gradient:

$$A^{(e)}_G(x)=\sum_s\big\langle\nabla_{h_{e,s}}y,\ n^G_{e,s}\big\rangle,\qquad A^{(e)}_\varepsilon=\sum_s\big\langle\nabla_{h_{e,s}}y,\ \varepsilon_{e,s}\big\rangle.$$

M1b, board-path IG, with $\gamma(\alpha)=h_e(x_0)+\alpha\,(h_e(x)-h_e(x_0))$ and $D_e$ linear [@sundararajan2017]:

$$\mathrm{IG}_G=\int_0^1\sum_s\big\langle\nabla_{h_{e,s}}y(\gamma(\alpha)),\ W^G_e(\tilde f^G_s-\tilde f^G_{0,s})\big\rangle\,d\alpha,\qquad \sum_G\mathrm{IG}_G+\mathrm{IG}_\varepsilon=y(x)-y(x_0).$$

The baseline is a real edit of $x$ or an average over real positions [@lundstrom2022]. M1c, feature occlusion, as defined in Section 6.1: $O_G(x)=g_e\big(h_e+W^G_e(b^G-\tilde f^G)\big)-y(x)$ with $b^G=0$ (mean occlusion, the per-square mean) or $\tilde f^G$ with chosen bits inverted (flipping), occluded on all 64 squares at once or per square. *Implementation of zeroing deltas:* with $U$ the set of squares whose 112 input values differ between $x$ and $x_0$, M1b uses $W^G_e(\tilde f^G_s-\tilde f^G_{0,s})\,\mathbb 1[s\in U]$ and adds the dropped part to $\mathrm{IG}_\varepsilon$. Under C2 the indicator is redundant, since local bits change only where planes change. *Question:* which of each square's inputs, named, move $y$. *Failure modes:* input saliency only (Section 5.2); spurious non-local credit unless C2 or zeroed deltas; conservation violated through post-LN blocks (M1a, M1b) [@ali2022]; off-manifold occlusion vectors that break the extractor's logical constraints (M1c) [@frye2021]. *Controls:* C2 decides, C3, C1, and for M1c agreement with M2 and M6 at $e$ on real edits. *Cost:* one backward pass per position (M1a), about 32 (M1b), after a port; $m+1$ forward passes per position under ONNX (M1c, family occlusion). *Residual:* $A^{(e)}_\varepsilon$ or $\mathrm{IG}_\varepsilon$, which collect broadcast-plane content and the 0.2% misfit. M1c has no complete residual: $g_e(q_e)-g_e(q_e+\varepsilon_e)$, the further change from also removing $\varepsilon_e$ once all families are occluded, is reported instead.

**M2. Decoded-substitution patching at layer $L$.** For family $G$ and the squares $S$ where $f^G$ changed:

$$h^{\rm patch}_{L,s}=h_{L,s}(x)+W^G_L\big(\tilde f^G_s(x')-\tilde f^G_s(x)\big),\ \ s\in S;\qquad \Delta^{(L)}_G=g_L(h^{\rm patch})-y(x).$$

The joint patch over all changed families gives $\Delta^{(L)}_{\rm named}=g_L(q+n'+\varepsilon)-y(x)$, leave-one-out patches give $\Delta^{(L)}_{-G}$, and $\gamma_L=\Delta^{\rm true}-\Delta^{(L)}_{\rm named}$ [@rubenstein2017]. *Implementation:* under ONNX the graph is split at the tap into a prefix and a suffix. The prefix runs once per board, the patched tensor is fed to the suffix, and only squares in $S$ are changed. Patches are made in the tap's own coordinates, after LayerNorm, so no renormalisation is applied to the patched token. *Question:* if lc0's layer-$L$ representation moved as the decoder says this edit moves it, how far would $y$ move, and how much of the true effect is left over? *Failure modes:* off-distribution patches and dormant pathways [@makelov2024]; non-additive single-family patches [@heimersheim2024]; per-family values ridge-dependent. *Controls:* C6, C8, C3, C1, C9. *Cost:* $2k+1$ suffix passes per tap for $k$ changed families; 12 or 13 for the pin edits of Table 5. *Residual:* $\gamma_L$.

**M3. Family erasure per layer.** With whitening $W=(\Sigma_{hh}^{1/2})^{+}$ and $P$ the orthogonal projector onto $\mathrm{col}(W\Sigma_{hz_G})$ [@belrose2023],

$$r^G_L(h)=h-W^{+}PW(h-\mu),\qquad \Delta^{\rm erase}_{G,L}(x)=g_L\big(r^G_L(h_L(x))\big)-y(x),$$

applied on all 64 squares. Report $\mathbb E|\Delta^{\rm erase}|$ and signed means beside the probe AUROC of $G$ at $L$. *Implementation:* the eraser is fitted on training positions with the family's bits as the concept $z_G$ and applied to held-out positions. LEACE is preferred to iterative nullspace projection because it removes exactly the linearly available information with the least change to $h$ [@belrose2023], [@ravfogel2020]. *Question:* is $G$'s linearly available information at $L$ used downstream [@elazar2021], [@palsson2024]? *Failure modes:* linear guarantee only; large families approach rank 256, so erased rank and variance removed are reported; concept–target correlation inflates engine-derived families [@kumarA2022]; recomputation hides use. *Controls:* C6 in whitened coordinates, C7, C1, C2. *Cost:* one affine patch under ONNX. *Residual:* output that survives erasing all families jointly.

**M4. Concept gradient through the probe.** With a family-score probe $P^{\mathcal G}_L:h_{L,s}\mapsto c_s\in\mathbb R^m$ and $J_s=\partial c_s/\partial h_{L,s}$ [@bai2023],

$$\mathrm{CG}_{L,s}=(J_s^\top)^{+}\,\nabla_{h_{L,s}}y,\qquad \rho^\perp_{L,s}=\nabla_{h_{L,s}}y-J_s^\top\mathrm{CG}_{L,s}.$$

TCAV is the linear case [@kim2018]. *Implementation:* the projection is computed in whitened coordinates $\tilde h=(\Sigma_{hh}^{1/2})^{+}(h-\mu)$, a pseudo-inverse because post-LN covariance is singular (rank at most 255), so that family directions are compared under the activation covariance rather than the Euclidean inner product [@park2024]. Both covectors transform alike, $\nabla_{\tilde h}y=\Sigma_{hh}^{1/2}\nabla_hy$ and $\tilde J_s^\top=\Sigma_{hh}^{1/2}J_s^\top$. Hence $\mathrm{CG}_{L,s}=(\Sigma_{hh}^{1/2}J_s^\top)^{+}\Sigma_{hh}^{1/2}\nabla_{h_{L,s}}y$ and the whitened residual is $\Sigma_{hh}^{1/2}\rho^\perp_{L,s}$, with $\rho^\perp$ as above. *Question:* how sensitive $y$ is to each family's probe coordinates, and what share of the gradient no family direction spans. *Failure modes:* family level only; probe directions need not be the model's [@park2024], [@tucker2021]; pseudo-inverse unstable for collinear families [@nicolson2025]. *Controls:* C1, C3 on probe regularisation, C2, control tasks [@hewitt2019], sign agreement with M3. *Cost:* one backward pass per position, after a port. *Residual:* $\|\Sigma_{hh}^{1/2}\rho^\perp_L\|^2/\|\Sigma_{hh}^{1/2}\nabla_{h_L}y\|^2$.

**M5. Interchange alignment per family per layer.** DAS [@geiger2024] in boundless form [@wu2023] learns $R_L$ and $k_G$. For base $x$ and source $x'$,

$$h^{\rm int}_{L,s}=h_{L,s}(x)+R_L^\top\Pi_{k_G}R_L\big(h_{L,s}(x')-h_{L,s}(x)\big),\quad s\in S.$$

The high-level counterfactual is $y(x')$ for minimal edits concentrated in $G$, otherwise a matched board [@gat2024]. Graded IIA is $1-\mathbb E\big(g_L(h^{\rm int})-y^*\big)^2/\mathbb E\big(y(x)-y^*\big)^2$. Isolate requires pairs that differ only in a correlated family to leave the $G$-interchange output unchanged [@huang2024]. $R_L$ is initialised from $W^G_L$ and, separately, from probe directions. MIB's causal-variable track is the closest benchmark [@mueller2025]. *Implementation:* $k_G$ is swept and the smallest size whose IIA is within a fixed tolerance of the best is reported, so that a family is not credited with a subspace larger than it needs. *Question:* does a subspace at $L$ mediate $G$'s effect, how many dimensions does it need, and does it separate from correlates? *Failure modes:* near-perfect IIA on random networks [@sutter2025]; dormant-pathway illusions [@makelov2024], contested by [@wu2024]. *Controls:* C1 (decisive), C8 as diagnostic. *Cost:* minutes per family and layer on CPU, after a port. *Residual:* $1-\mathrm{IIA}$, plus interchange of the orthogonal complement of all family subspaces, as in causal scrubbing [@chan2022].

**M6. Named/residual split.** Defined in Section 7.2. Three properties follow from the definition. *Completeness:* $\phi^{(L)}_n+\phi^{(L)}_\varepsilon=g_L(q+n'+\varepsilon')-g_L(q+n+\varepsilon)=y(x')-y(x)$, since the decoder split reconstructs each token exactly. *Antisymmetry:* exchanging $x$ and $x'$ exchanges $n\leftrightarrow n'$ and $\varepsilon\leftrightarrow\varepsilon'$, which negates both brackets and hence $\phi_n$. *Orientation:* by antisymmetry, $\sum_{\rm pairs}\phi_n/\sum_{\rm pairs}\Delta^{\rm true}$ over oriented pairs equals $\sum\mathrm{sign}(\Delta^{\rm true})\phi_n/\sum|\Delta^{\rm true}|$ over unoriented ones, so $\rho_L$ does not depend on the order in which a pair was recorded, and its denominator cannot cancel. Pairs with $|\Delta^{\rm true}|\le\tau$ are dropped (Section 10.5), and a bootstrap over pairs gives its interval. The layer differences are $\kappa_L=\rho_L-\rho_{L+1}$. *Question:* how much of a real output change is carried by what the vocabulary names at each tap. *Failure modes:* two players only, so no per-family answer; $\kappa_L$ is not a flow. Variants: conductance through the span of $W_L$ against its complement [@dhamdhere2019], and a tied decoder $W_L=W+\Delta_L$ [@lindsey2024], [@lawson2025]. $\rho_L$ can exceed 1 or turn negative when the players cancel, so both $\phi$ values are reported. Mixed coalitions such as $n'$ with $\varepsilon$ are off-manifold, and information moved between squares counts as residual. *Controls:* C1, C2 at $e$, C3, C9, and a neighbour-augmented decoder $D_L(f_s,\sum_t\bar a^L_{st}f_t)$ with lc0's mean attention $\bar a^L$ as a locality control. *Cost:* one suffix pass per tap per pair beyond M2's joint patch. *Residual:* $\phi^{(L)}_\varepsilon$ and $1-\rho_L$.

**M7. Supervised replacement graph with error nodes.** Per position, freeze attention patterns and LayerNorm denominators [@ali2022], [@ameisen2025]. Replace each feed-forward block (squared ReLU in this network) with

$$\mathrm{FFN}_L(u)\approx S_Lu+V_L\,\sigma\big(P_L(u)\big)+b_L,\qquad r_{L,s}=\mathrm{FFN}_L(u_s)-\big[S_Lu_s+V_L\sigma(P_L(u_s))+b_L\big].$$

Nodes are feature writes $V^G_L\sigma(P^G_L(u_s))$ per block and square, plus error nodes $r_{L,s}$. The edge weight from node $(L,s,G)$ to $y$ is activation times virtual weight through frozen attention. Ameisen et al. [@ameisen2025] report completeness 0.80 and replacement score 0.61 on a language model. Completeness is the share of absolute edge attribution that reaches $y$ through non-error nodes. The replacement score measures how often the replacement graph, with error nodes removed, reproduces the model's output. *Implementation:* transcoders are fitted per block by ridge on the probe outputs at the block input, so every node carries a family name by construction. *Question:* which family, written at which block on which square, reaches $y$ along which route. *Failure modes:* frozen query-key products and smolgen hide routing [@jenner2024]; leakage if fed extractor bits [@mahinpei2021]. *Controls:* C1, C3, C4, ablation of pruned graphs against M2 and M6, AtP*-style recomputation [@kramar2024]. *Cost:* ridge fits in minutes, one graph per position in seconds, after a port. *Residual:* error-node share per block and square.

**M8. Cross-layer erasure and substitution cascade.** Erase or substitute $G$ at $L$ and read every $L'>L$:

$$T^{L\to L'}_{G\to G'}=\mathbb E_x\Big[\mathrm{score}_{G'}\big(P_{L'}(h_{L'})\big)-\mathrm{score}_{G'}\big(P_{L'}(h^{\,\text{erase }G@L}_{L'})\big)\Big].$$

In the decoder-read variant, $n^G_L(x)$ is replaced by $n^G_L(x')$ and the change at $L'$ is read through $D_{L'}$'s split. *Implementation:* the extractor's own dependency graph (which bit families are computed from which) is written out once, and every off-diagonal entry of $T$ that the dependency graph predicts is flagged, so that transfer lc0 adds can be separated from transfer the vocabulary imposes. *Question:* which named features lc0 builds from which, where it recomputes an erased feature, and which named features turn into residual. *Failure modes:* probes read off-distribution activations; recomputation from unerased squares; confounding by the extractor's dependency graph. *Controls:* C6, C1, C7. *Cost:* one patched pass per $(G,L)$. *Residual:* $\Delta\phi_\varepsilon$ per $(G,L)$.

**M8∘M2. Feature-path composition.** With $h^{\rm patch}_{L'}$ the token at $L'$ after M2's patch of $G$ at $L$, $\hat f_s(h)=P_{L'}(h_s)$ the probe's output, and $\Delta n^{G'}_{L',s}=W^{G'}_{L'}\big(\hat f^{G'}_s(h^{\rm patch}_{L'})-\hat f^{G'}_s(h_{L'}(x))\big)$,

$$r_{L'}=h^{\rm patch}_{L'}-h_{L'}(x)-\sum_{G'}\Delta n^{G'}_{L'},\qquad \pi_{G\to G'}=g_{L'}\big(h_{L'}(x)+\Delta n^{G'}_{L'}\big)-y(x),\qquad \pi_{G\to r}=g_{L'}\big(h_{L'}(x)+r_{L'}\big)-y(x).$$

*Implementation:* $\hat f$ is the probe's probability vector, and the per-square centring of $D_{L'}$ cancels in the difference. $P_{L'}$ and $D_{L'}$ are fitted separately, so the round trip $D_{L'}\circ P_{L'}$ is not the identity and $r_{L'}$ keeps what the vocabulary cannot carry. A regression on the 14,565 columns of $W_{L'}$ would not, since they span $\mathbb R^{256}$. Families whose named change has norm below 1% of the joint named change, summed over squares, are skipped (proposed tolerance). *Calibration:* the same round trip on the real edit's change $h_{L'}(x')-h_{L'}(x)$ gives a remainder floor, reported beside $\varepsilon'_{L'}-\varepsilon_{L'}$; only the excess of $\pi_{G\to r}$ over that floor counts as leaving the vocabulary. *Question:* through which named families at $L'$ does $G$'s effect at $L$ reach $y$, and how much leaves the vocabulary between $L$ and $L'$ [@goldowsky2023]? *Failure modes:* non-additive path scores; collinearity at both ends; the probe reads patched tokens off-distribution, and the decoder receives soft probe outputs where it was fitted on bits. *Controls:* C1, C3, C6. *Cost:* one patched pass to $L'$ per $(G,L)$, then one suffix pass per family above tolerance. *Residual:* $\pi_{G\to r}$.

**M9. Family Shapley with an error player.** For coalition $S\subseteq\{G_1,\dots,G_m,\varepsilon\}$ and baseline $x_0$,

$$h^{(S)}_{L,s}=q_{L,s}+\sum_{G\in S}n^G_{L,s}(x)+\sum_{G\notin S}n^G_{L,s}(x_0)+\mathbb 1[\varepsilon\in S]\,\varepsilon_{L,s}(x)+\mathbb 1[\varepsilon\notin S]\,\varepsilon_{L,s}(x_0),$$

with $v(S)=\mathbb E_{x_0}[g_L(h^{(S)})]$. *Use:* pairwise M9 is the per-family attribution for an edit and is graded like M2. Distributional M9 attributes a single position against a pool and is graded by localisation (Section 10.2). Site players $(G,s)$ or $(G,L)$ split the same value by square or by depth at proportionally higher cost. A global variant gives ConceptSHAP's completeness [@yeh2020]. Players $(G,L)$ ask at which depth $G$ is used. *Failure modes:* mixed coalitions off-manifold [@kumarIE2020]; ridge-dependent family values. *Controls:* C1, C3, C7, C2, sign and rank agreement with M2. *Cost:* $(m+1)\times M\times|\mathcal B|$ suffix passes by permutation sampling, for example $29\times20\times8\approx4{,}600$. *Residual:* $\varphi_\varepsilon$.

**M10. Conservative relevance chain.** One modified backward pass with DeepLIFT rescale rules [@shrikumar2017], chained as in G-DeepSHAP [@chen2022], attention and LayerNorm detached [@ali2022], AttnLRP only for softmax handling [@achtibat2024]. Relevance at each tap is split over components by baseline-differenced ratios:

$$R^c_{L,s}=\sum_j R_{L,s,j}\,\frac{\Delta c_j}{\Delta h_{L,s,j}},\quad c\in\{n^{G_1},\dots,n^{G_m},\varepsilon\}.$$

The baseline is the same real edit or pool used by M6, so that M10 and M6 can be compared pair by pair. *Failure modes:* conservation is a design property, not faithfulness; the ratio is unstable where $\Delta h_j\approx0$; rules through smolgen and post-LN are untested. Concept relevance propagation [@achtibat2023] is the concept-conditioned relative. *Controls:* C1 [@adebayo2018], agreement with M6. *Residual:* $R^\varepsilon_L$.

**M11. Disagreement map.** For family attributions $A^{(k,L)}_G$ from methods $k$ at taps $L$ on one set of oriented pairs, report Kendall rank agreement over families, sign agreement, and coverage as defined in Section 9.3, aggregated over pairs rather than per pair, since a per-pair ratio $A_\varepsilon/\Delta^{\rm true}$ is undefined where $\Delta^{\rm true}\approx0$. *Question:* where do methods and taps agree on which families carry an edit, and where does agreement fall to the random-network level? *Failure modes:* agreement is not faithfulness, and the decoder-split methods (M1, M2, M6, M9, M10) share ridge artefacts and agree for that reason alone. Agreement is therefore computed within and across method families separately, against the C1 floor. A learned ensemble matching $\sum_GA_G$ to $\Delta^{\rm true}$ is rejected: it rewards small $\varepsilon$, and with $\varepsilon$ included every complete method has zero loss, so the weights are unidentified [@he2025]. *Cost:* negligible.

**M12. Residual budget and hybrid dictionary.**

$$\varepsilon_{L,s}\approx B^{\rm nl}_L\,\psi(f_s)+B^{\rm nb}_L\textstyle\sum_t\bar a^L_{st}f_t+B^{\rm cf}_L\,\varepsilon_{L-1,s}+\eta_{L,s},$$

with increments averaged over orders, a Shapley value over the three predictor blocks, following Engels et al.'s [@engels2025] decomposition of SAE error. The method has three steps. First, the budget assigns the variance of $\varepsilon_L$ to (b) nonlinearity in $f_s$, (a) non-locality, carry-over from $L-1$, and the remainder $\eta_L$. Second, a size-swept sparse code on $\eta_L$ adds its latents as players in M9 or nodes in M7, as in concept-bottleneck SAEs [@kulkarni2026]. Third, latents are matched to held-back candidate features or new extractor bits, scored as in Karvonen et al. [@karvonen2024], with a contrastive variant on board pairs [@poupart2024]. *Question:* how much of the decay in Section 5.4 is non-locality, map class, carry-over or remainder, and does the remainder hold stable, used, nameable units? *Failure modes:* latents depend on dictionary size [@leask2025]; naming works only if the candidate bank contains the concept [@shang2024]; predictable is not nameable. *Controls:* C1, size sweep, seed stability. *Cost:* regressions in minutes, SAEs in CPU hours per layer. *Residual:* the budget, then the share of $\varphi_\varepsilon$ recovered by named latents.

**Cost arithmetic for the first experiment (estimate).** M2's joint patch, $k$ single patches and $k$ leave-one-out patches take $2k+1$ suffix passes per tap, and M6 adds one, giving $2k+2\approx26$ for $k\approx12$. Suffixes from $e$, L2, L5, L8 and L9 cost about 1.0, 0.8, 0.5, 0.2 and 0.1 of a full pass, 2.6 in total. With 1,000 pin-edit pairs (500 pins, two edits each) and ten settings (five $\alpha$ values, two further splits, three random networks), that is $26\times2.6\times10^4\approx6.8\times10^5$ pass-equivalents, about 63 minutes at 180 passes per second. Two base passes per pair and network and M0 add minutes. The refits are 50 ridge solves (five taps, ten settings), each a Cholesky solve of a Gram matrix of dimension 14,629 (bits plus square one-hot), about $10^{12}$ floating-point operations or roughly a minute on a multi-core CPU. The Gram matrix does not depend on tap or network, so the 50 solves share about seven factorisations (five $\alpha$ values, two further splits) and the hour is an upper bound. Tapping 20,000 positions through three random networks takes about 6 minutes. The total is about two CPU hours. Batching, graph-splitting overhead and solve times have not been measured.

## Appendix B. Shared Controls

**Table B1**
*Controls C1 to C9*

| Code | Control | Basis |
|---|---|---|
| C1 | Random-initialised lc0, at least three seeds: refit maps, rerun the method; the C1 range is the min–max over seeds | [@adebayo2018], [@sutter2025] |
| C2 | Local vocabulary (occupant + square, plus rule-50, side-to-move and per-square castling bits if added); the only admissible vocabulary at $e$ | Section 5.2; [@hewitt2019] |
| C3 | Ridge-$\alpha$ sweep ($10^{-2}$ to $10^{2}$), further train/test splits; family-rank stability (Spearman) | Section 4.6 |
| C4 | Linear against MLP map; MLP reported only next to C1 | [@sutter2025] |
| C5 | Shuffled-label refit of $D_L$ or $P_L$ | [@adebayo2018] |
| C6 | Random direction of equal rank or norm | [@elazar2021] |
| C7 | With and without engine-derived families (`mate`, `material_win`, threat lines) | [@kumarA2022], [@parisini2026] |
| C8 | Row-space against nullspace split of a patch direction, as a diagnostic | [@makelov2024], [@wu2024] |
| C9 | Noising and denoising directions; fixed ablation type | [@heimersheim2024], [@miller2024] |

Each control isolates one alternative explanation: C1 the architecture and vocabulary without training, C3 the decoder's regulariser, C5 marginal statistics, C6 intervention size, C7 value-predictive engine features. C1 and C3 are the two controls the first experiment runs. C1 separates what lc0 learned from what the architecture and vocabulary impose. C3 separates lc0's structure from the decoder's regulariser. C2 is not a control in the usual sense at the embedding: it is the only vocabulary admissible there.

## Appendix C. Taxonomy

**Table C1**
*Methods on the nine taxonomy axes*

| # | Entry | Direction | Layer span | Map | Operator | Residual | Identifiability | Faithfulness test | Cost | Combines |
|---|---|---|---|---|---|---|---|---|---|---|
| M0 | input | features→$y$ | none or one tap | linear/GBM | Shapley on surrogate | $y-s$ | regulariser-dependent | localisation vs GT-M | minutes | no |
| M1a/b | input | decode | $e$; path crosses 10 blocks | linear | gradient, board-path IG | $\mathrm{IG}_\varepsilon$ | ridge-dependent per family | vs edits | port; cheap | no |
| M1c | input | decode | $e$; full forward pass | linear | occlusion, synthetic baseline | $g_e(q_e)-g_e(q_e+\varepsilon_e)$ | ridge-dependent per family | vs M2/M6 at $e$ | ONNX; $m{+}1$ passes | no |
| M2 | input | decode | per layer | linear | substitution patching | $\gamma_L$ | joint unique; per family ridge-dependent | interventional | ONNX; $2k{+}1$ passes/tap | no |
| M3 | readout | encode | per layer | linear | erasure | output after erasing all | unique for $G$ with linear correlates | interventional + random rank | ONNX; cheap | no |
| M4 | readout | encode | per layer | linear/MLP | Jacobian pseudo-inverse | $\|\rho^\perp\|$ | min-norm; unstable if collinear | vs edits; sign vs M3 | port; 1 bwd | no |
| M5 | readout | both | per layer | linear rotation | interchange | $1-$IIA | up to rotation | IIA, Isolate | port; hours | no |
| M6 | input | decode | all taps | linear | 2-player Shapley | $\phi_\varepsilon$, $1-\rho_L$ | exact given $D_L$ | completeness exact; C3 | ONNX; 1 pass/tap | sweep |
| M7 | both | both | cross-layer, path | sparse-linear | linear edges at frozen activations | error nodes | regulariser-dependent | ablation vs M2/M6 | port; s/pos | yes |
| M8 | readout or decoder | encode→encode | cross-layer | linear | erasure/substitution, re-read | $\Delta\phi_\varepsilon$ | as M3 or M2 | interventional | ONNX; 1 pass per $(G,L)$ | yes |
| M8∘M2 | input | decode→probe→decode | $L\to L'\to y$ | linear + probe | path patching | $\pi_{G\to r}$ | ridge-dependent at both ends | vs $\Delta^{(L)}_G$ | ONNX; hundreds/pair | yes |
| M9 | input | decode | per layer | linear | Shapley, pairwise or distributional | $\varphi_\varepsilon$ | ridge-dependent per family | localisation | ONNX; ~30 s/pos/layer | with site players |
| M10 | input | decode | all taps, one pass | linear | DeepLIFT rescale | $R^\varepsilon_L$ | rule-dependent | agreement with M6 | port, custom bwd | sweep |
| M11 | mixed | mixed | all | none | comparison | coverage spread | nothing fitted | random-net floor | negligible | yes |
| M12 | dictionary-matched | both | per layer, uses $L-1$ | linear + sparse | regression, SAE | budget | non-unique (SAE size) | random init, size sweep | CPU hours | yes |

*Note.* "Entry": decoder input, probe readout, or dictionary match. Five methods are operators on one decoder split (M1, M2, M6, M9, M10). M0, M1c, M2, M3, M6, M8, M8∘M2, M9 and M11 run without a PyTorch port.

## Appendix D. Worked Example: Procedure and Output Templates

The base position and the two edits are given in Section 10.6. The procedure below is for the pair $(x,x'_b)$, value target, at each tap $L\in\{e\ (\text{C2}),\ \text{L2},\ \text{L5},\ \text{L8},\ \text{L9}\}$.

1. Run lc0 on both boards to obtain $y(x)$, $y(x'_b)$ and $\Delta^{\rm true}$. Swap the boards if $\Delta^{\rm true}<0$.
2. Tap both boards and decode, giving $n_L$, $\varepsilon_L$, $n'_L$ and $\varepsilon'_L$.
3. Patch $n'-n$ on the 21 changed squares into $h_L(x)$ and run the suffix, giving $\Delta^{(L)}_{\rm named}$ and $\gamma_L$.
4. Patch $n-n'$ into $h_L(x'_b)$, giving $g_L(q+n+\varepsilon')$ and hence $\phi^{(L)}_n$ and $\phi^{(L)}_\varepsilon$.
5. For each of the 13 changed families, run a single patch and a leave-one-out patch, giving $\Delta^{(L)}_G$ and $\Delta^{(L)}_{-G}$.
6. Repeat steps 3 to 5 for each C3 setting and on each of the three random-initialised networks (C1).

**Table D1**
*Output for one pair, value target (template, nothing computed)*

| Tap | $\Delta^{\rm true}$ | $\Delta^{(L)}_{\rm named}$ | $\gamma_L$ | $\phi^{(L)}_n$ | $\phi^{(L)}_\varepsilon$ | $\phi^{(L)}_n$ range under C1 | $\phi^{(L)}_n$ range under C3 |
|---|---|---|---|---|---|---|---|
| $e$ (C2) | – | – | – | – | – | – | – |
| L2 | – | – | – | – | – | – | – |
| L5 | – | – | – | – | – | – | – |
| L8 | – | – | – | – | – | – | – |
| L9 | – | – | – | – | – | – | – |

*Note.* $\Delta^{\rm true}$ is the same in every row. Over all pairs, column $\phi^{(L)}_n$ sums to the numerator of $\rho_L$.

**Table D2**
*Per-family readout for the same pair at L5 (template, nothing computed)*

| Family | $\Delta^{(L)}_G$ | $\Delta^{(L)}_{-G}$ | Rank, $\alpha=10^{-2}$ | Rank, $\alpha=1$ | Rank, $\alpha=10^{2}$ | Reported as |
|---|---|---|---|---|---|---|
| `xray` | – | – | – | – | – | own row if ranks stable, else merged |
| `mobility` | – | – | – | – | – | as above |
| `route` | – | – | – | – | – | as above |
| `control` | – | – | – | – | – | as above |
| … (9 more) | – | – | – | – | – | as above |
| All changed, jointly | $\Delta^{(L)}_{\rm named}$ | – | – | – | – | always |

**Policy variant.** The same pair supports the move Bxc6 (b5 to c6), legal on both boards. At `/encoder9/ln2` its logit splits exactly over $n_{b5},\varepsilon_{b5},n_{c6},\varepsilon_{c6}$: four Shapley values from 16 head evaluations, summing to $\ell(x'_b)-\ell(x)$ (Section 7.4). The pair $(x,x'_a)$ needs a move legal on both boards, such as d2–d4, since the bishop has left b5. Both require the ln2 decoder, which has not been fitted. At L2 and L5 the same moves serve as policy targets for M2 and M6 without it.

**Policy procedure.** For a move $a\to b$ legal on both boards: (1) compute $\ell_{a\to b}(x)$ and $\ell_{a\to b}(x')$ and orient the pair by their difference; (2) at an ln1 tap, run steps 2 to 6 of the value procedure with the logit as target, reporting credit on families at $a$ and $b$ separately from credit elsewhere; (3) once an ln2 decoder exists, evaluate the head on the 16 combinations of $\{n_a,n'_a\}\times\{\varepsilon_a,\varepsilon'_a\}\times\{n_b,n'_b\}\times\{\varepsilon_b,\varepsilon'_b\}$ and compute the four Shapley values exactly. Comparing $\rho_L$ for value and for policy on the same pairs tests cause (f) of Section 5.4.

**Reading the templates.** Table D1 is read per row, against its C1 and C3 columns. A $\phi^{(L)}_n$ inside the C1 range or the C3 range carries no lc0-specific information at that tap. Table D2 is read only for families whose ranks agree across the three $\alpha$ values. In this pair, `route` changes 180 bits and `xray` 8. A large `route` value and a small `xray` value would therefore not show that lc0 ignores the x-ray. It may only show where ridge put the shared credit.

## Appendix E. Survey Method

The design is a structured methods survey, not a PRISMA systematic review, because the literature is mostly arXiv and lab reports with unstable terminology.

**Themes.** Eight themes were searched in parallel: attribution operators and their axioms, concept attribution through learned concept maps, probing and erasure, concept bottlenecks and completeness, sparse dictionaries across layers, circuit discovery with error nodes, causal abstraction, and interpretability of game-playing networks. Forum posts were admitted only as gray literature for methods without another source.

**Procedure and inclusion.** Each theme used keyword search, seed papers and one round of citation chasing per seed. Search stopped after two consecutive empty chases. The window was 2016 to 2026, plus foundational axiomatic work. Included were works that attribute outputs through a map to or from a concept vocabulary or dictionary, that define a faithfulness, completeness or identifiability criterion for such attributions, or that apply such methods to a game-playing network.

**Verification and tiers.** Every entry's title, authors, year, venue and identifier were checked against the arXiv record and the publisher or proceedings page. Sources were tiered: 1 peer-reviewed, 2 lab technical report, 3 preprint or workshop paper, 4 gray literature. The eight theme bibliographies hold about 122 entries with overlaps between themes, and 89 distinct works are cited. The reference list marks every source below Tier 1.

**Analysis and review.** Each method received a row on the nine axes of Appendix C. Two devil's-advocate reviews challenged the scoping and the synthesis. An editorial review, an ethics review and a third devil's-advocate review challenged the draft report, and each resulting change is logged in the project's response file (`paper/attribution/07_response.md`). The reviews added the correlational baseline M0, the policy treatment, the identifiability statement, the first-experiment recommendation and the worked example, and removed claims stated as shown that were only predicted.

**Themes and sections.** The themes map onto the paper as follows. Attribution operators supply the removal rules of Section 4.1 and the Shapley, gradient and occlusion machinery of M1, M6, M9 and M10. Concept attribution and probing supply Sections 4.2 and 4.6 and methods M3 and M4. Concept bottlenecks supply the residual precedents of Section 4.3 and M0. Sparse dictionaries across layers supply Section 4.5, M7 and M12. Circuit discovery supplies error nodes, path patching and patching conventions (M2, M8∘M2, C9). Causal abstraction supplies M5, the identifiability results of Section 9.1 and the benchmarks used to argue for interventional ground truth. Game-network interpretability supplies Section 4.8 and the lc0-specific constraints on frozen attention.

**Exclusions.** Saliency without a concept map and concept discovery without output attribution were excluded unless used here.

**Annotation.** Annotations were written by AI agents from abstracts and landing pages. Claims attributed to sources in this paper were checked against those annotations, not against full texts read by a human.

## Data Availability

The decoder and probe fits are in `lc0-probe/results_square_decoder.json` and related result files in the xai-chess repository (github.com/theRealProHacker/xai-chess). The feature extractor is `features/square/`, the input encoder `lc0-probe/lc0_planes.py`. Figure 1 is generated by `paper/attribution/fig_decoder.py` from the result file, and Figure 2 by `fig_compositions.py`. The lc0 network `t1-256x10-distilled-swa-2432500` is publicly distributed by the Leela Chess Zero project. Positions come from the public Lichess puzzle database.

## Ethics Statement

The work uses public chess positions and a public chess network. No human participants or personal data are involved.

## CRediT Author Statement

xai-chess project: conceptualisation, methodology, software, investigation, data curation, writing (review and editing), supervision. Literature search, synthesis and drafting were AI-assisted (see the AI-use disclosure).

## Conflict of Interest

None declared.

## Funding

This work received no external funding.

## AI-Use Disclosure

This paper was produced with Claude (Anthropic), run through Claude Code as a multi-agent pipeline: literature search, verification of bibliographic records against arXiv and publisher pages, evidence synthesis, three devil's-advocate reviews, editorial and ethics review, and drafting. Bibliographic annotations were written by AI agents from abstracts and landing pages. Cited claims were checked against those annotations, not against full texts read by a human. The numbers in Tables 1 and 2 come from the project's own code and result files. The embedding, head and input structure were read from the exported ONNX graph and `lc0-probe/lc0_planes.py`. The bit counts in Table 5 come from running the project's extractor. The project author directs the research and decides which conclusions to adopt.

## References

Sources below Tier 1 are marked: Tier 2 lab technical report, Tier 3 preprint or workshop paper, Tier 4 gray literature.

<!--REFERENCES-->

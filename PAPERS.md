# PAPERS — xai-chess / ChessFaith research bibliography

IEEE references for every work touched by this project — the original research-proposal citations, the office-hours/landscape corpus, and the completeness-verifier additions. **Titles are clickable** (canonical URL). Sorted by relevance to the benchmark (Core → Supporting → Peripheral); within a tier, by centrality.

*87 works · 30 core / 46 supporting / 11 peripheral · compiled 2026-06-27, extended 2026-08-01.*

References [1]–[64] are the original corpus and their numbers are fixed, because the survey and the summaries both cite them as stable tags. The move-explanation and dataset sweep of 2026-08-01 appended [65]–[87] rather than renumbering.


## Core (23)

*Directly load-bearing: the methods, oracles, explanation engines, and faithfulness machinery the benchmark builds on or grades.*

[1] T. McGrath, A. Kapishnikov, N. Tomašev, A. Pearce, D. Hassabis, B. Kim, et al., "[Acquisition of chess knowledge in AlphaZero](https://arxiv.org/abs/2111.09259)," arXiv:2111.09259, 2021.

[2] L. Schut, N. Tomasev, T. McGrath, D. Hassabis, U. Paquet, and B. Kim, "[Bridging the Human-AI Knowledge Gap: Concept Discovery and Transfer in AlphaZero](https://arxiv.org/abs/2310.16410)," arXiv:2310.16410, 2023.

[3] A. Karvonen, "[Emergent World Models and Latent Variable Estimation in Chess-Playing Language Models](https://arxiv.org/abs/2403.15498)," in Proc. Conf. on Language Modeling (COLM), 2024, arXiv:2403.15498.

[4] E. Jenner, S. Kapur, V. Georgiev, C. Allen, S. Emmons, and S. Russell, "[Evidence of Learned Look-Ahead in a Chess-Playing Neural Network](https://arxiv.org/abs/2406.00877)," arXiv:2406.00877, 2024.

[5] E. Sandmann, S. Lapuschkin, and W. Samek, "[The Algorithm Is Not the Behavior: Learned Priors Override Look-Ahead in a Chess-Playing Neural Network](https://arxiv.org/abs/2508.21380)," arXiv:2508.21380, 2025.

[6] R. Lin, Z. Jin, G. Zhou, et al., "[Tracing the Thought of a Grandmaster-level Chess-Playing Transformer](https://arxiv.org/abs/2604.10158)," arXiv:2604.10158, 2026.

[7] A. Ruoss, G. Delétang, S. Medapati, J. Grau-Moya, L. K. Wenliang, et al., "[Amortized Planning with Large-Scale Transformers: A Case Study on Chess](https://arxiv.org/abs/2402.04494)," arXiv:2402.04494, 2024.

[8] D. Monroe, G. Eilender, P. Chalmers, et al., "[Chessformer: A Unified Architecture for Chess Modeling](https://arxiv.org/abs/2605.19091)," arXiv:2605.19091, 2026.

[9] Z. Tang, D. Jiao, R. McIlroy-Young, et al., "[Maia-2: A Unified Model for Human-AI Alignment in Chess](https://arxiv.org/abs/2409.20553)," in Advances in Neural Information Processing Systems (NeurIPS), 2024. arXiv:2409.20553.

[10] Y. Zhang, A. P. Jacob, V. Lai, D. Fried, and D. Ippolito, "[Human-Aligned Chess With a Bit of Search](https://arxiv.org/abs/2410.03893)," arXiv:2410.03893, 2024.

[11] J. Kim, J. Goh, I. Hwang, J. Cho, and J. Ok, "[Bridging the Gap between Expert and Language Models: Concept-guided Chess Commentary Generation and Evaluation](https://arxiv.org/abs/2410.20811)," in Proc. NAACL, 2025. arXiv:2410.20811.

[12] Z. Tang, Q. Wen, S. Grief-Albert, et al., "[Grounded Chess Reasoning in Language Models via Master Distillation](https://arxiv.org/abs/2603.20510)," arXiv:2603.20510, 2026.

[13] Q. Wen, Z. Tang, and A. Anderson, "[ChessQA: Evaluating Large Language Models for Chess Understanding](https://arxiv.org/abs/2510.23948)," arXiv:2510.23948, 2025.

[14] Y. Gur-Arieh, A. Marasović, and M. Geva, "[Faithfulness Metrics Don't Measure Faithfulness: A Meta-Evaluation with Ground Truth](https://arxiv.org/abs/2605.25052)," arXiv:2605.25052, 2026.

[15] K. Zaman and S. Srivastava, "[A Causal Lens for Evaluating Faithfulness Metrics](https://arxiv.org/abs/2502.18848)," in Proc. EMNLP, 2025. arXiv:2502.18848.

[16] Y. Han, Y. Lee, and J. Do, "[RFEval: Benchmarking Reasoning Faithfulness under Counterfactual Reasoning Intervention in Large Reasoning Models](https://arxiv.org/abs/2602.17053)," arXiv:2602.17053, 2026.

[17] A. Basu and P. Chakraborty, "[ICE: Intervention-Consistent Explanation Evaluation with Statistical Grounding for LLMs](https://arxiv.org/abs/2603.18579)," arXiv:2603.18579, 2026.

[18] Y. Chen, R. Zhong, N. Ri, C. Zhao, H. He, J. Steinhardt, et al., "[Do Models Explain Themselves? Counterfactual Simulatability of Natural Language Explanations](https://arxiv.org/abs/2307.08678)," arXiv:2307.08678, 2023.

[19] F. Spinnato, "[Towards Piece-by-Piece Explanations for Chess Positions with SHAP](https://arxiv.org/abs/2510.25775)," arXiv:2510.25775, 2025.

[20] R. McIlroy-Young, S. Sen, J. Kleinberg, and A. Anderson, "[Aligning Superhuman AI with Human Behavior: Chess as a Model System](https://arxiv.org/abs/2006.01855)," in Proc. 26th ACM SIGKDD Int. Conf. Knowledge Discovery & Data Mining (KDD), 2020. arXiv:2006.01855.

[21] A. Lee, D. Wu, E. Dinan, and M. Lewis, "[Improving Chess Commentaries by Combining Language Models with Symbolic Reasoning Engines](https://arxiv.org/abs/2212.08195)," arXiv:2212.08195, 2022.

[22] H. Zang, Z. Yu, and X. Wan, "[Automated Chess Commentator Powered by Neural Chess Engine](https://arxiv.org/abs/1909.10413)," arXiv:1909.10413, 2019.

[23] D. Cruz, "[Understanding the Learned Look-Ahead Behavior of Chess Neural Networks](https://arxiv.org/abs/2505.21552)," arXiv:2505.21552, 2025.


## Supporting (33)

*Methodologically adjacent: interpretability, simulatability, concept-eval, and search-explanation work that informs the design.*

[24] F. Doshi-Velez and B. Kim, "[Towards A Rigorous Science of Interpretable Machine Learning](https://arxiv.org/abs/1702.08608)," arXiv:1702.08608, 2017.

[25] P. Hase and M. Bansal, "[Evaluating Explainable AI: Which Algorithmic Explanations Help Users Predict Model Behavior?](https://arxiv.org/abs/2005.01831)," in Proc. 58th Annu. Meeting Assoc. Comput. Linguistics (ACL), 2020, arXiv:2005.01831.

[26] E. Mills, S. Su, S. Russell, and S. Emmons, "[ALMANACS: A Simulatability Benchmark for Language Model Explainability](https://arxiv.org/abs/2312.12747)," arXiv:2312.12747, 2023.

[27] A. Poché, A. Jacovi, A. M. Picard, et al., "[ConSim: Measuring Concept-Based Explanations' Effectiveness with Automated Simulatability](https://arxiv.org/abs/2501.05855)," arXiv:2501.05855, 2025.

[28] M. Turpin, J. Michael, E. Perez, and S. R. Bowman, "[Language Models Don't Always Say What They Think: Unfaithful Explanations in Chain-of-Thought Prompting](https://arxiv.org/abs/2305.04388)," in Advances in Neural Information Processing Systems (NeurIPS), arXiv:2305.04388, 2023.

[29] T. Lanham, A. Chen, A. Radhakrishnan, et al., "[Measuring Faithfulness in Chain-of-Thought Reasoning](https://arxiv.org/abs/2307.13702)," arXiv:2307.13702, 2023.

[30] A. Jacovi and Y. Goldberg, "[Towards Faithfully Interpretable NLP Systems: How Should We Define and Evaluate Faithfulness?](https://arxiv.org/abs/2004.03685)," in Proc. 58th Annu. Meeting Assoc. Comput. Linguistics (ACL), 2020, pp. 4198-4205. arXiv:2004.03685.

[31] B. Kim, M. Wattenberg, J. Gilmer, et al., "[Interpretability Beyond Feature Attribution: Quantitative Testing with Concept Activation Vectors (TCAV)](https://arxiv.org/abs/1711.11279)," in Proc. Int. Conf. Machine Learning (ICML), 2018; arXiv:1711.11279.

[32] A. Atrey, K. Clary, and D. Jensen, "[Exploratory Not Explanatory: Counterfactual Analysis of Saliency Maps for Deep Reinforcement Learning](https://arxiv.org/abs/1912.05743)," in Proc. Int. Conf. Learning Representations (ICLR), 2020. arXiv:1912.05743.

[33] J. Shymanski, J. Brue, and S. Sen, "[Not All Explanations Are Created Equal: Investigating the Pitfalls of Current XAI Evaluation](https://arxiv.org/abs/2511.03730)," arXiv:2511.03730, 2025 (also Ch. 9, Bi-directionality in Human-AI Collaborative Systems, Springer, 2025).

[34] J. Kowalski, M. H. M. Winands, M. Wiśniewski, S. Reda, and A. Wilbik, "[Towards Explaining Monte-Carlo Tree Search by Using Its Enhancements](https://arxiv.org/abs/2506.13223)," arXiv:2506.13223, 2025.

[35] S. Lu, M. Bahavarnia, H. Baroud, Y. Zhang, H. Purohit, and A. Mukhopadhyay, "[Toward Template-Free Explainability for Monte Carlo Tree Search](https://arxiv.org/abs/2605.16524)," arXiv:2605.16524, 2026.

[36] Z. An, H. Baier, A. Dubey, A. Mukhopadhyay, and M. Ma, "[Enabling MCTS Explainability for Sequential Planning Through Computation Tree Logic](https://arxiv.org/abs/2407.10820)," in Proc. 27th European Conf. on Artificial Intelligence (ECAI), 2024. arXiv:2407.10820.

[37] Z. An, X. Wang, H. Baier, et al., "[Combining LLMs with Logic-Based Framework to Explain MCTS](https://arxiv.org/abs/2505.00610)," arXiv:2505.00610, 2025 (extended abstract, AAMAS-25).

[38] Y. Qian, T. Miller, Z. Qian, and L. Zhao, "[Exploring Explainable Multi-agent MCTS-minimax Hybrids in Board Game Using Process Mining](https://arxiv.org/abs/2503.23326)," arXiv:2503.23326, 2025.

[39] Y. Poupart, "[Contrastive Sparse Autoencoders for Interpreting Planning of Chess-Playing Agents](https://arxiv.org/abs/2406.04028)," in Workshop on Interpretable Policies in Reinforcement Learning (InterpPol) @ RLC, arXiv:2406.04028, 2024.

[40] P. Hammersborg and I. Strümke, "[Information based explanation methods for deep learning agents -- with applications on large open-source chess models](https://arxiv.org/abs/2309.09702)," arXiv:2309.09702, 2023.

[41] P. Hammersborg and I. Strümke, "[Reinforcement Learning in an Adaptable Chess Environment for Detecting Human-understandable Concepts](https://arxiv.org/abs/2211.05500)," arXiv:2211.05500, 2022.

[42] A. Mészáros, P. Reizinger, and F. Huszár, "[Out-of-Distribution Tests Reveal Compositionality in Chess Transformers](https://arxiv.org/abs/2510.20783)," arXiv:2510.20783, 2025.

[43] S. Lomasov, J. Goldfeder, M. H. Erol, et al., "[Exploring Human-AI Conceptual Alignment through the Prism of Chess](https://arxiv.org/abs/2510.26025)," arXiv:2510.26025, 2025.

[44] L. Ma, S. Cao, R. L. Logan IV, et al., "[BUMP: A Benchmark of Unfaithful Minimal Pairs for Meta-Evaluation of Faithfulness Metrics](https://arxiv.org/abs/2212.09955)," in Proc. 61st Annu. Meeting Assoc. Comput. Linguistics (ACL), 2023, arXiv:2212.09955.

[45] A. F. Spies, W. Edwards, M. I. Ivanitskiy, et al., "[Transformers Use Causal World Models in Maze-Solving Tasks](https://arxiv.org/abs/2412.11867)," arXiv:2412.11867, 2024.

[46] M. Singh, A. Alabdulkarim, G. Mansi, and M. O. Riedl, "[Explainable Reinforcement Learning Agents Using World Models](https://arxiv.org/abs/2505.08073)," in Workshop on Explainable AI (XAI) at IJCAI, arXiv:2505.08073, 2025.

[47] S. Wang, L. Ji, R. Wang, et al., "[Explore the Reasoning Capability of LLMs in the Chess Testbed](https://arxiv.org/abs/2411.06655)," arXiv:2411.06655, 2024.

[48] Z. Tang, D. Jiao, E. Xue, et al., "[Learning to Imitate with Less: Efficient Individual Behavior Modeling in Chess](https://arxiv.org/abs/2507.21488)," arXiv:2507.21488, 2025.

[49] R. McIlroy-Young, R. Wang, S. Sen, J. Kleinberg, and A. Anderson, "[Learning Models of Individual Behavior in Chess](https://arxiv.org/abs/2008.10086)," in Proc. 28th ACM SIGKDD Conf. Knowledge Discovery and Data Mining (KDD), 2022. arXiv:2008.10086, 2020.

[50] M. Limpijankit, Y. Chen, M. Subbiah, N. Deas, and K. McKeown, "[Counterfactual Simulatability of LLM Explanations for Generation Tasks](https://arxiv.org/abs/2505.21740)," arXiv:2505.21740, 2025.

[51] P. Hong and B. Roth, "[Do LLM Self-Explanations Help Users Predict Model Behavior? Evaluating Counterfactual Simulatability with Pragmatic Perturbations](https://arxiv.org/abs/2601.03775)," arXiv:2601.03775, 2026.

[52] Q. Li and W. Jiang, "[Tracking vs. Deciding: The Dual-Capability Bottleneck in Searchless Chess Transformers](https://arxiv.org/abs/2603.29761)," arXiv:2603.29761, 2026.

[53] D. Jiao, G. Eilender, Z. Tang, and A. Anderson, "[Understanding Skill Adaptation in Transformers Using Sparse Autoencoders: Chess as a Model System](https://openreview.net/forum?id=Wxl0JMgDoU)," OpenReview preprint (ICLR 2025 submission, id Wxl0JMgDoU), 2025.

[54] M. Soliman and N. Ehab, "[Caïssa AI: A Neuro-Symbolic Chess Agent for Explainable Move Suggestion and Grounded Commentary](https://doi.org/10.1007/978-3-032-02813-6_11)," in KI 2025: Advances in Artificial Intelligence (LNAI), Springer, 2025, doi:10.1007/978-3-032-02813-6_11.

[55] H. Baier and M. Kaisers, "[Towards Explainable MCTS](https://ir.cwi.nl/pub/30850/30850.pdf)," in Proc. AAAI Workshop on Explainable Agency in Artificial Intelligence, 2020. [Online]. Available: https://ir.cwi.nl/pub/30850/30850.pdf

[56] M. Guid, M. Možina, C. Bohak, A. Sadikov, and I. Bratko, "[Building an Intelligent Tutoring System for Chess Endgames](https://www.scitepress.org/papers/2013/43891/43891.pdf)," in Proc. 5th Int. Conf. Computer Supported Education (CSEDU), 2013, pp. 263-266, doi:10.5220/0004389102630266.


## Peripheral (8)

*Background and tangential: broader game-AI, human-move modelling, and survey context.*

[57] M. Tao, X. Liang, X. Song, et al., "[Enhancing Commentary Strategies for Imperfect Information Card Games: A Study of Large Language Models in Guandan Commentary](https://arxiv.org/abs/2406.17807)," arXiv:2406.17807, 2024.

[58] W. Wang, F. Bie, J. Chen et al., "[Can Large Language Models Master Complex Card Games?](https://arxiv.org/abs/2509.01328)," in Proc. NeurIPS, 2025. arXiv:2509.01328.

[59] Z. Cheng, J. Yu, and X. Xing, "[A Survey on Explainable Deep Reinforcement Learning](https://arxiv.org/abs/2502.06869)," arXiv:2502.06869, 2025.

[60] K. Hamade, R. McIlroy-Young, S. Sen, J. Kleinberg, and A. Anderson, "[Designing Skill-Compatible AI: Methodologies and Frameworks in Chess](https://arxiv.org/abs/2405.05066)," arXiv:2405.05066, 2024.

[61] H. Alrdahi and R. Batista-Navarro, "[Aspect-based Sentiment Evaluation of Chess Moves (ASSESS): an NLP-based Method for Evaluating Chess Strategies from Textbooks](https://arxiv.org/abs/2405.06499)," in Proc. 10th Workshop on Games and Natural Language Processing (Games and NLP), LREC-COLING 2024, arXiv:2405.06499, 2024.

[62] B. Skidanov, D. Erbesfeld, G. Weiss, and A. Elyasaf, "[A Behavior-Based Knowledge Representation Improves Prediction of Players' Moves in Chess by 25%](https://arxiv.org/abs/2504.05425)," arXiv:2504.05425, 2025.

[63] D. Zhong, D. Huang, and C. Greenberg, "[Predicting Human Chess Moves: An AI Assisted Analysis of Chess Games Using Skill-group Specific n-gram Language Models](https://arxiv.org/abs/2512.01880)," arXiv:2512.01880, 2025.

[64] C. Riedl and E. Bogert, "[Who Benefits from AI? Self-Selection, Skill Gap, and the Hidden Costs of AI Feedback in Chess](https://arxiv.org/abs/2409.18660)," arXiv:2409.18660, 2024.


## Additions — move-explanation and dataset sweep (2026-08-01)

*23 works: 7 core, 13 supporting, 3 peripheral. Surfaced by a six-pass sweep of move-level explanation systems, symbolic and DSL explanation, explanation datasets, and evaluation methodology.*

**Citation convention for this section.** Entries are given by author surname. Titles, venues, years, and DOI/arXiv identifiers were each confirmed against a fetched source; author initials could not be verified in this pass and are deliberately omitted rather than guessed. Fill them from the publisher record before any submission.

### Core (7)

*The move-explanation systems a grading benchmark would consume, and the nearest existing precedents for grading against causal ground truth.*

[65] Jhamtani, Gangal, Hovy, Neubig, and Berg-Kirkpatrick, "[Learning to Generate Move-by-Move Commentary for Chess Games from Large-Scale Social Forum Data](https://aclanthology.org/P18-1154/)," in Proc. 56th Annu. Meeting Assoc. Comput. Linguistics (ACL), 2018, pp. 1661–1671.

[66] Pálsson and Björnsson, "[Empirical Evaluation of Concept Probing for Game-Playing Agents](https://ebooks.iospress.nl/doi/10.3233/FAIA240574)," in Proc. 27th European Conf. Artificial Intelligence (ECAI), Frontiers in Artificial Intelligence and Applications, vol. 392, 2024, pp. 874–881, doi:10.3233/FAIA240574.

[67] Puri, Verma, Gupta, Kayastha, Deshmukh, Krishnamurthy, and Singh, "[Explain Your Move](https://arxiv.org/abs/1912.12191)" (SARFA; full subtitle **not verified**), in Proc. Int. Conf. Learning Representations (ICLR), 2020. arXiv:1912.12191.

[68] Das and Chernova, "[Leveraging Rationales to Improve Human Task Performance](https://arxiv.org/abs/2002.04202)," in Proc. 25th Int. Conf. Intelligent User Interfaces (IUI), 2020. arXiv:2002.04202.

[69] Cui, Ling, and Ng, "[Communicating Chess Strategies in Natural Language](https://arxiv.org/abs/2607.11486)," arXiv:2607.11486, 2026.

[70] Sadikov, Možina, Guid, Krivec, and Bratko, "[Automated Chess Tutor](https://doi.org/10.1007/978-3-540-75538-8_2)," in Computers and Games (CG 2006), Lecture Notes in Computer Science, Springer, pp. 13–25, doi:10.1007/978-3-540-75538-8_2.

[71] Bizjak and Guid, "[Automatic Recognition of Similar Chess Motifs](https://doi.org/10.1007/978-3-031-11488-5_12)," in Advances in Computer Games (ACG 2021), Lecture Notes in Computer Science, vol. 13262, pp. 131–141, doi:10.1007/978-3-031-11488-5_12.

### Supporting (13)

*The symbolic and advice-language lineage, the probing and dataset infrastructure, and the faithfulness-methodology works the survey's lens engages.*

[72] Pálsson and Björnsson, "[Unveiling Concepts Learned by a World-Class Chess-Playing Agent](https://www.ijcai.org/proceedings/2023/0541.pdf)," in Proc. 32nd Int. Joint Conf. Artificial Intelligence (IJCAI), 2023, pp. 4864–4872, doi:10.24963/ijcai.2023/541.

[73] Karvonen, Wright, et al., "[Measuring Progress in Dictionary Learning for Language Model Interpretability with Board Game Models](https://arxiv.org/abs/2408.00113)," in Advances in Neural Information Processing Systems (NeurIPS), 2024. arXiv:2408.00113.

[74] Feng, et al., "[ChessGPT: Bridging Policy Learning and Language Modeling](https://arxiv.org/abs/2306.09200)," in Advances in Neural Information Processing Systems (NeurIPS) Datasets and Benchmarks Track, 2023. arXiv:2306.09200.

[75] Wilkins, "[Using patterns and plans in chess](https://doi.org/10.1016/0004-3702(80)90039-9)," *Artificial Intelligence*, vol. 14, no. 2, pp. 165–203, 1980, doi:10.1016/0004-3702(80)90039-9.

[76] Bratko and Michie, "[An advice program for a complex chess programming task](https://doi.org/10.1093/comjnl/23.4.353)," *The Computer Journal*, vol. 23, no. 4, pp. 353–359, 1980, doi:10.1093/comjnl/23.4.353.

[77] Janičić, Marić, and Maliković, **[title not verified — do not cite this string]**, *Logical Methods in Computer Science*, vol. 15, no. 1:34, 2019, doi:10.23638/LMCS-15(1:34)2019. The open-access re-formalisation of Bratko's KRK strategy: names the auxiliary predicates (`room`, `critical square`, `rook exposed`, `rook divides`, `L-pattern`), gives the seven prioritised strategy rules with coverage counts over all 175,168 white-to-move KRK positions, and extends the win result to n×n boards. Resolve the title and author initials from the DOI before use.

[78] Možina, Žabkar, and Bratko, "[Argument based machine learning](https://doi.org/10.1016/j.artint.2007.04.007)," *Artificial Intelligence*, vol. 171, no. 10–15, pp. 922–937, 2007, doi:10.1016/j.artint.2007.04.007.

[79] Guid, Možina, Sadikov, and Bratko, "[Deriving Concepts and Strategies from Chess Tablebases](https://doi.org/10.1007/978-3-642-12993-3_18)," in Advances in Computer Games (ACG 2009), Lecture Notes in Computer Science, pp. 195–207, doi:10.1007/978-3-642-12993-3_18.

[80] Kameko, Mori, and Tsuruoka, "Learning a Game Commentary Generator with Grounded Move Expressions," in Proc. IEEE Conf. Computational Intelligence and Games (CIG), 2015, pp. 177–184.

[81] Costeff, "[The Chess Query Language: CQL](https://doi.org/10.3233/ICG-2004-27404)," *ICGA Journal*, vol. 27, no. 4, pp. 217–225, 2004, doi:10.3233/ICG-2004-27404.

[82] Lyu, Apidianaki, and Callison-Burch, **[title not verified](https://aclanthology.org/2024.cl-2.6/)** (the faithfulness survey; 110+ methods in five categories), *Computational Linguistics*, vol. 50, no. 2, pp. 657–723, 2024.

[83] Parcalabescu and Frank, **[title not verified](https://aclanthology.org/2024.acl-long.329/)** (the self-consistency critique; introduces CC-SHAP), in Proc. 62nd Annu. Meeting Assoc. Comput. Linguistics (ACL), 2024.

[84] Björnsson, "[Chess and explainable AI](https://content.iospress.com/articles/icga-journal/icg240256)," *ICGA Journal*, vol. 46, no. 2, pp. 67–75, 2024, doi:10.3233/ICG-240256. Content beyond the abstract **UNVERIFIED** (paywalled).

### Peripheral (3)

*Context and standards: the broader game-commentary field, the existing symbolic annotation vocabulary, and the commercial prior art.*

[85] Zheng, et al., "[From Multimodal Perception to Strategic Reasoning: A Survey on AI-Generated Game Commentary](https://arxiv.org/abs/2506.17294)," arXiv:2506.17294, 2025.

[86] S. J. Edwards (coordinator), "Standard: Portable Game Notation Specification and Implementation Guide" (canonical URL **not verified**), 1994. Defines the Numeric Annotation Glyph (NAG) vocabulary, `$0`–`$255`.

[87] The Software Toolworks, *Chessmaster 3000 (1991) PC Manual*. [Online]. Available: https://archive.org/details/chessmaster-3000-pc-manual — documents the "Natural Language Advice" feature (Mentor menu → Advice → Detailed); see also *Chessmaster 4000 Windows 95 Edition (1995) PC Manual*, archive.org/details/chessmaster-4000-windows-95-edition-manual.


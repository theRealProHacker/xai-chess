Previous (unguided) journey: https://typst.app/project/rhD6vG2m3c6TY3SA1Sy2hn

# First meeting 17.08. 

Mainly about general understanding

Approaches:
1. Feature attribution + LIME/SHAP
2. Tree search explanation (already some papers on this)
3. Feedback engine. The thesis: automatic feedback/validation is the most important part in the training process of any model
4. Transformer on Chess (ChessGPT + Chessformer) -> feasibility questionable -> possibly distillation
5. New approach discovered: Looking into the brain of the engine


Two main objectives:
1. Look into the brain of an engine with some simple concept
2. Develop a formal language for chess explanations
3. Read more papers on the two above

Finding: there is no existing formal language for chess explanations. There are only natural language chess explanations. Defining this formal language first, would already be a (possibly necessary) contribution. 

So let's start with that first:

## Objective 1

The goal is to find a formal language that can represent most existing chess explanations
while remaining verifiable by a relatively simple deterministic engine.  

Thoughts:
First, the task is already impossible from the beginning depending on how you define the word "most". 
    So, we should definitely stay humble and don't expect too much. 

Essentially, there are two main things that both humans and chess engines use
    - Tactics, calculating: going through different moves to a certain tree depth
    - Positional/Strategic, heuristics: certain patterns that are advantageous or disadvantageous, most easily evaluated and compared as numbers (e.g. "a queen is worth 9 points")

In engines, tactics is dealt with using tree search and positional evaluation is actually the crux of the matter done using neural networks. 

The number and complexity of the heuristics increases with playing strength. A lot of it is intuition, i.e. things that can actually not be expressed in natural language.

But let's just get to it. 
We can explain chess moves or we can do positional analyses. 
In some sense, you could say that the positional analysis is sufficient because any move can be justified by showing it leads to the best possible position (you can only do one move and there is a finite set of possible positions), but the more natural is to start with move explanations first. 

## Objective 1, first pass: my personal experience

My personal experience is that many move explanations mostly look like this: move X leads to Y or Z, Y leads to A and Z leads to B and A and B are both good (so X is good). It could of course be much less complicated "move X leads to A" or much more complicated (more width, more depth or more complex outcomes). Also you can reason using exclusion, or explain why move X is better than moves Y and Z, etc. 

This is the structure of explanations but we can also start from the bottom and look at the primitives of chess (and probably also our formal language):
- The board, its regions and squares
- The pieces both in abstract (a knight) and mostly the concrete (the knight on f3)
- A move

These primitives then interact with each other, e.g.
- This piece attacks this piece
- This piece pins that piece to this other piece
- This piece defends this square
- This piece controls this line/diagonal

So, I would start with listing the tactical patterns widely recognized by human chess players (this is from my personal experience + https://lichess.org/training/themes):
1. Hanging piece
2. Trapped piece
3. Fork
4. Pin
5. Skewer
6. Discovered check, discovered attack and clearance
7. Double check
8. Removing the defender and deflection
9. Interference
10. Attraction
11. Sacrifice
12. X-Ray attack and collinear move
13. Intermezzo (Zwischenzug)
14. Quiet move 
15. Zugzwang
16. and more

+ lots of kinds of mates

So now let's move to the positional part. 

Some heuristics
- Mates are really good
- Knights are bad at the edge of the board
- Golden Opening rules
- In the endgame, win pawns and promote pawns
- In the beginning, protect your king and hide it away, in the endgame put it into the center of the board

The problem with these in general is that they often might be wrong, you can basically always invent new ones, and there are different people and cultures with more or less slightly different sets of these and there might be some intuitive, not fully expressed or expressable heuristics as mentioned above. 

But let's go at this in a more structured way by looking at the literature

## Objective 1, second pass: Literature

The Russian chess school is one of the most influential chess cultures dominating the world for a long time and coincidentally it's most defining feature is its focus on positional play, so it definitely is a good starting point.

Main positional evaluation criterias:
1. Material
2. Pawn structure
3. Piece activity
4. Space and Control
4. Initiative and Attack
5. King Safety

## Objective 1, third pass: Simple Validation on Feng et al. (Minimal Maynard Coding)

To-Do for next week

## Objective 2: Existing work

There already is some literature on looking into the engine both on Stockfish (SF) and on an open source replica of Deepminds AlphaZero.

e.g.
IJCAI 2023 — "Unveiling Concepts Learned by a World-Class Chess-Playing Agent."


## Objective 2: Experiment setup

My original plan was to use the Lichess puzzle tags.

Specifically, I set up the following experiment:

First we take only puzzles with a single clearly identifiable pin that is sustained throughout the puzzle.
Now the assumption is that of those puzzles, all those pins are significant where the corresponding puzzle has an official Lichess pin tag.
The goal is then to train a NN that can for a given position accurately detect whether the pin is significant. 

Then we can look into how deep the engine can look into the future (i.e. how good it is at puzzles of varying length) and in which layer of the engines NN this information is mostly represented. 

This would be an advancement to the papers because we aren't just probing presence but relevance. 

The idea was to view the Lichess tags as human-labeled data considering the fact that Lichess has a human voting system.
However, the tags are mainly added through a relatively smart automatic system and then put up to vote, and are only added or removed if enough people voted for it. 

Automatic tagging works basically like this:
```py
if pin_prevents_attack(puzzle) or pin_prevents_escape(puzzle):
    tags.append("pin")
```

So, I let Claude Code write the engine patch to get out the NNUE computations from SF and filtered out the puzzles with a single identifiable absolute pin. Of those 151,614 puzzles 43,177 (28.48%) are tagged with `pin`. 3260 (~7,55%) originally automatically tagged puzzles have the tag removed, seemingly by human vote. This was computed by comparing the automatic system against the real tag in the lichess database. 

## What does the Stockfish evaluation NN look like

Good general overview
https://github.com/official-stockfish/nnue-pytorch/blob/master/docs/nnue.md

Overview by Claude for SF 18

┌─────────────────────────┬────────────────────────────────────────────┬──────────────────────────────────────────┐
│         tensor          │                   shape                    │                   note                   │
├─────────────────────────┼────────────────────────────────────────────┼──────────────────────────────────────────┤
│ active feature indices  │ ≤32 HalfKA + ≤128 threats, ×2 perspectives │ the input                                │
├─────────────────────────┼────────────────────────────────────────────┼──────────────────────────────────────────┤
│ accumulation            │ 2 × 1024 int16                             │ linear in the input                      │
├─────────────────────────┼────────────────────────────────────────────┼──────────────────────────────────────────┤
│ threatAccumulation      │ 2 × 1024 int16                             │ linear in the input                      │
├─────────────────────────┼────────────────────────────────────────────┼──────────────────────────────────────────┤
│ transformer output (L1) │ 1024 uint8                                 │ 2 persp × 512 pairs — first nonlinearity │
├─────────────────────────┼────────────────────────────────────────────┼──────────────────────────────────────────┤
│ fc_0_out                │ 16 int32                                   │ index 15 is the skip term                │
├─────────────────────────┼────────────────────────────────────────────┼──────────────────────────────────────────┤
│ ac_sqr_0 ‖ ac_0         │ 30 uint8                                   │ not 32 — fc_1 is AffineTransform<30,32>  │
├─────────────────────────┼────────────────────────────────────────────┼──────────────────────────────────────────┤
│ fc_1_out                │ 32 int32                                   │                                          │
├─────────────────────────┼────────────────────────────────────────────┼──────────────────────────────────────────┤
│ ac_1_out                │ 32 uint8                                   │                                          │
├─────────────────────────┼────────────────────────────────────────────┼──────────────────────────────────────────┤
│ fc_2_out                │ 1 int32                                    │ the eval                                 │
└─────────────────────────┴────────────────────────────────────────────┴──────────────────────────────────────────┘

## Some reading


General problem for explanations: What can I assume the target user to already know? What is obvious to him?

https://arxiv.org/pdf/2510.25775
SHAP on chess pieces

The idea is very simple: Just take the best method for XAI and apply it to chess in a way, that chess engines already respresent in their internal neural networks.

Use cases they show:
1. Piece values -> Question: Wouldn't it be enough to just remove the piece and look at the evaluation delta? Or in other words, how much better is SHAP really for this?
	For example in the given position in Figure 4 the bishop delta is -6,2-3,8 = -10 and the knight delta is 8,9-3,8 = 5,1. This also matches the results from the SHAP evaluation (this only works for limited depths of course, because at greater depths the engine reaches the end state. 
	Their analysis is even slightly flawed, as they only consider the difference in evaluation of the pieces but it seems like the Black pawns loose value because of the existence of the bishop
2. Showing differences between


SHAP would be much more interesting on hand-crafted explainable features

In general, a completely black-box approach is difficult to be very successful???


https://arxiv.org/pdf/2306.09200

Essentially just GPT trained on chess data. 
Question: would fine-tuning be sufficient?

Really interesting would be to ask the model for reasoning for their move, or combine with CoT.


Probably one of the best, most valuable papers
https://arxiv.org/pdf/2605.19091

Chessformer

A unified model that combines raw strength (like SF or LC0), human-like play (like MAIA) and interpretebility

Main interesting idea: 
token representation and positional encoding
pure transformer (sota arch) with geometric attention, that actually adapts to the geometric properties of chess.
lc0 distillation

Some papers on faithfulness metrics for CoT

https://arxiv.org/pdf/2502.18848 -> Causal model
https://arxiv.org/pdf/2605.25052 -> BonaFide, ground truth
https://arxiv.org/pdf/2307.13702 &
https://arxiv.org/pdf/2305.04388 -> Different faithfulness measurements, relatively uninteresting
https://arxiv.org/pdf/2004.03685 -> Fundementals on faithfulness

Learned look-ahead in lc0: https://arxiv.org/pdf/2406.00877

https://ir.cwi.nl/pub/30850/30850.pdf

MCTS explanation: some ideas I had as well for explaining tree search
- Selecting the most important branches
- Summarising branches
	Some natural language examples: 
		"Whatever White does here, Black can play X"
		"Wherever the King moves, he is mated"
		etc.

Why not this alternative? -> Important if the reason is not obvious, and then should be included in selecting the most important branches. 
-> We need to measure importance of a move/difficulty of an explanation

Very similar: https://arxiv.org/pdf/2503.23326

A0
https://arxiv.org/pdf/2111.09259

Probing
Manual inspection
Changing model weights in response to increasing training

# Second meeting 31.08.

## Feedback

Keywords: activation oracle, symbolic learning, circuits 

Next week
1. Mini Maynard Coding 
2. Stockfish, depth
3. Leela -> Transformer architecture is more similar to LLM research

## Objective 2: Simplicity

In the original demo I had reduced each layer to 16 dimensions using SVD to keep everything comparable in a logistic regression, but to keep it simple, I just did an MLP on the raw vectors for each layer instead which gives a quite nice result.

![Layers](./sf-probe/layers_mlp.png)

## Objective 2: Depth

The next question was trying to find out how far the net could look into the future essentially.

The experiment above was trying to understand which layer was best at understanding whether a pin was relevant in the puzzle. 
Now we want to find out if it is easier to see that the pin is relevant if it becomes relevant closer in time.
To decide this we can use Lichess' automatic tagging code which goes through the puzzle and looks at where the pin becomes relevant. 
We call this the relevance distance `d`. 

## Score by distance class trained seperately
So, first instinct is to plot score over `d`.

However, there is one major problem:

Puzzles with higher relevance distances occur less often

| distance | n | mean rating | mean length |
|---|---|---|---|
| 0 | 15138 | 1531 | 1.64 |
| 1 | 8987 | 1667 | 2.13 |
| 2 | 1564 | 2022 | 3.07 |
| 3 | 137 | 2340 | 4.07 |
| 4 | 14 | 2362 | 5.07 |
| 5 | 4 | 2551 | 6.00 |
| 6 | 1 | 2489 | 7.00 |
| negative | 105789 | 1635 | 2.07 |

So the naive approach cannot be trusted if a majority of the puzzles have d=0. 
For this reason, I tried to keep the number of puzzles used in training uniform over d. This didn't really work when d=3 with just 137 puzzles is kept. But especially at several thousand samples we clearly see that the result from the naive approach is actually correct: It is easier to see a pins relevance when it's closer. 

![Distances](./sf-probe/distance_shared.png)

The next step was looking at what happens when we train a seperate model for each depth, choosing negative samples of the same length as the positive samples. Also, I used the opportunity to do a 1:1 ratio between positive and negative samples. 

## Score by distance class trained seperately

![Score per distance class](./sf-probe/distance_own.png)

What we can see is that training the model on just puzzles with d=0 for example doesn't make the model better at detecting relevance. Also, the plot almost didn't change. 

## Classifier

Just for fun, I wanted to try to train a model to predict between the classes d=0, d=1 and negative, if we remove everything else. 

![Classifier](./sf-probe/confusion_3class.png)

## Types of pin relevances

The real question is if there are different types of relevances that are more easily detectable than others apart from the relevance depth?

Lichess doesn't really distinguish many different types

1. pin prevents attack vs escape
2. the attacked piece is hanging or more valuable than the pinned piece/the pinned piece is hanging or more valuable than the piece attacking it

This gives 4 options:
1. `attack.hanging`
2. `attack.greater`
3. `escape.hanging`
4. `escape.greater`

![Reasons](./sf-probe/reasons.png)

What we see is that escape is consistently easier to read, specifically `escape.greater`


## Objective 3

Goal is to look at the layers of Leela Zero (lc0) just like we looked at Stockfish. 

Before making the experiment the hypothesis was that lc0 would be much better compared to SF because it has a much larger GPU-based tranformer network and there were already some papers showing its trained lookahead in contrast to SF which relies more heavily on tree-search:

1. "Evidence of Learned Look-Ahead in a Chess-Playing Neural Network" by Erik Jenner, Shreyas Kapur, Vasil Georgiev, Cameron S. Allen, Scott Emmons, and Stuart Russell, published at NeurIPS 2024 (arXiv:2406.00877)
2. A 2025 follow-up paper, "Understanding the learned look-ahead behavior of chess neural networks"

## Experiment

On each layer we have a 256-vector per square which gives 256x65=16384 dimensions.

To keep training feasible, we did two main things:
First we have the pin subset, which is the three squares that are directly involved in the pin. 
Then we have the mean of all squares. 

![lc0 results](./lc0-probe/layers_mlp.png)

## Objective 1: Example of some chess commentary

The chess commentary is from this paper: ChessGPT: Bridging Policy Learning and Language Modeling

![Chess Commentary Example](./image/journey/Screenshot%202026-09-05%20at%2011-33-38%20Annotated%20chess%20commentary%20—%20ChessGPT%20corpus.png)

Looking into chess commentary, we can see a lot of redundant simple commentary actually. 

1. Opening
2. Opening
3. Opening/Reference to normal play
4. Factual comment
5. Positional analysis
4. Purpose
5. Explanation (this is what we are actually looking for, in this case a concrete line, with two final reasons, just like we predicted)
6. Psychology
7. Psychology
8. Explanation (Only option that achieves A)
9. Explanation (do X to achieve A)
10. Positional analysis
11. Factual comment, psychology, explanation (Move X because after a some move later open file is occupied which enables move Y which treatens moves Z and W with mate)

Looking at the commentary, it would actually be very easy to build a simple deterministic system to write this commentary. 

## Precedent: [Jhamtani et al. (ACL 2018)](https://aclanthology.org/P18-1154.pdf)

![Six categories](./image/journey/Screenshot%202026-09-06%20at%2019-17-18%20Learning%20to%20Generate%20Move-by-Move%20Commentary%20for%20Chess%20Games%20from%20Large-Scale%20Social%20Forum%20Data%20-%20P18-1154.pdf.png)

They already show that the commentary we are looking for, specifically "Planning/Rationale" and what they are calling "Direct Move Description" is also the most difficult to understand for language models. 

The other categories seem pretty simple to solve or are in the contrary almost impossible to generate from just the game itself.
1. Describing a move (what I called purpose) is mostly natural language processing. The most difficult part about this is the high variability. A move often does a lot of things, picking one shows taste. 
2. Move quality is very simple to do just with engines. This is also what chess.com and others are already very good at doing.
3. Comparing alternatives is just a certain way of reasoning, this also includes the exclusion principle, i.e. I am doing this move, because there is no better one. 
4. Planning and rationale is what we are looking for. It actually explains not just what a move is doing in the immediate, but also what the greater plan is that the move is part of
5. Contextual information ad general game comments are not feasible in this setting but would be important in practice. However, they require more context and are out-of-scope for now. 

Chess.com example (https://www.reddit.com/r/chessbeginners/comments/10cwkvw/why_does_chesscom_say_this_is_a_blunder_i/)

![chess.com from Reddit](./image/journey/chess_com_explanation.jpeg)

## Predicate Logic?

One possible basis for our language would be something like Prolog:

https://www.cs.rochester.edu/u/nelson/courses/csc_173/predlogic/computation.html

"Example facts:
```prolog
male(adam).
female(anne).
parent(adam,barney).
```

Example rules:

    son(X,Y) :- parent(Y,X) , male(X)
    daughter(X,Y) :- parent(Y,X) , female(X)
  
To run a Prolog program the user must ask a question (goal) by stating a theorem (asserting a predicate) which the Prolog interpreter tries to prove.

If the predicate contains variables, the interpreter prints the values of the variables used to make the predicate true. 
"

For example we could define an absolute pin as the following (more Python-like pseudo-code):

```py
absolute_pin (N,M,K) :- ours(N) and theirs(M) and theirs(K) and is_king(K) and (
        (rook(N) or queen(N)) and on_line_in_order(N,M,K)
    ) or (
        (bishop(N) or queen(N)) and on_diagonal_in_order(N,M,K)
    )
```

## Back to some chess literature

Main idea is to "distill" SF's evaluation function to some more simple model on explainable factors.
For that we need to understand what the factors are. 

So, let's look more at the chess literature. Specifically, literature from the International Chess School (www.chessmasterschool.com) and even more specifically their sample chess lessons from here (https://www.chessmasterschool.com/#program):

1. Think Like a Strong Player
2. Chess Strategy
3. Chess Tactics


Their theses:
1. For each move think about and evaluate the threats and consequences
2. Keep a To-Do list with goals you want to achieve (over 3-10 moves)
3. Each move should either give you an advantage or remove an advantage from your opponent

## What are these advantages?

Wilhelm Steinitz's 9 advantages
1. lead in development
2. mobility of the pieces
3. seizure of the center
4. weak oponnent king position
5. weak squares in the opponent's position
6. superior pawn formation
7. (a pawn majority on the queen side)
8. open files
9. (bishop pair)

The ICS proposes more modern quantitative and qualitative advantages

Quantative
1. Material advantage
2. local superiority of forces

Quantitave advantages are most easy to measure and compare.

Material advantage is probably one of the first things you learn and was the basis for engine evaluations until recently (it has changed to winning probability):

- Pawn: 1 (the unit)
- Knight: 3
- Bishop: 3
- Rook: 5
- Queen: 9

For this reason, knights and bishops are often called light pieces while rooks and queens are called heavy pieces.

The local superiority is quite simple to understand. The chess board is just large enough that locality exists, where something could be happening on one part of the board, where one party has the upper hand just like on a battle field. 

One heuristic: an attack on the opponents king will be successful if you have three more pieces in the attack than the opponent has in the defense. 

Essentially this is global material advantage vs local material advantage. This should be realtively easily measurable. 

Qualitative
1. King's safety (my own sub category)
    - pawns infront intact
    - open files and diagonals to king
    - how many pieces attack and how many defend (attacking and defending resources)
    - Example for worst possible king safety: https://lichess.org/analysis/r2q1rk1/pppb1p2/2n1p3/3pP2Q/3P4/3BB3/PPP2P2/2K3RR_b_-_-_0_1?color=white
2. Qualitative value of pieces
    - Mobility: 
        1. how many fields can the piece move to and 
        2. how little moves does the piece need to cover the entire board
    - Positioning
        1. knight in the center 4x4
        2. other pieces on open lines/diagonals
        3. the more central and the more squares under control, the better
    - Role: tiered
        1. Out of play
        2. Defensive
        3. Offensive
        4. Both offensive and defensive
    - Stability: Can it be removed from its position easily?
        - Existence of outposts -> potential
    - Grade of coordination -> difficult to assess
        1. Sustainment
        2. Protection
        3. Limitation
        4. Attack
        5. Obstruction
3. Pawn structure
    - number of pawn islands
    - isolated pawns
    - free pawns
    - double and triple pawns
    - pawn mobility
4. Space advantage
    - Better control of a certain area of the chessboard by advancing pawns
    - Space advantage increases the qualitative value of pieces -> don't trade if you have space advantage
    - Example where White has maximal space advantage: https://lichess.org/analysis/r1bqnrk1/pp1nbpp1/2p1p2p/2PpPP2/PP1P2PP/2NB1N2/1BQ5/2KR3R_w_-_-_0_1?color=white&position=518
        The evaluation is +6, even though materially there is no difference
5. Initiative
    - Forcing the opponent to respond to your threats instead of his own plans 

For PMEs (positions with material equivalence) also:
1. play-coordinating piece is valued higher

## Positional evaluation

1. Material (as above)
    - if unequal material, compare
    - if material disadvantage, check compensation
2. Threats and tactical ressources
    - checks, takes, threats
3. King's safety (as above)
4. The centre
    1. control: score as 0-4 to 4-0
    2. type
        - closed
        - open
        - static
        - mobile
        - dynamic
5. Pieces
    1. Development (only in early game)
    2. Pieces out of play
    3. Local superiority
    4. Collaboration
    5. Rooks on open files
    6. Bishops on important diagonals
    7. Knights on important foreposts
    8. Bishop-pair
    9. Safe and active queen
    10. Piece-by-piece comparison
6. Pawns
    1. Space
    2. Weak pawns
    3. Superior pawn structure
    4. Dynamic
    5. Mobile pawn majority
    6. Free pawn

## Extra

Exact space counting method in "Winning Chess" by Silman and Seirawan. 

Other thought: ChessGPT + engine + tools

Eval:
1. LLM judge/human review
2. blue/rouge, embedding similarity
3. keywords/tool calls/structured output

Questions: how large does my eval set need to be?

Possibly: 100 or 1000

1. dspy -> prompt optimization
2. CLIP2 on chess -> this is ChessGPT
3. Activation oracle on leela activation -> where do we get the data?

# Third meeting 10.09.

For next time:
1. Build features (with Claude)?
2. Paper, sparse autoencoder on engine
3. Activation oracle paper
4. Clean up chess commentary dataset
5. Build minimal gold-standard-dataset
6. ChessGPT/Claude/Gemini + Prompt (zero-shot)
7. Datensätze suchen
8. PoC: Pin relevance with LLM/activation oracle

## lc0 depth

Late for last meeting:

![lc0 depth eval one model](./lc0-probe/distance_shared.png)

![lc0 depth eval seperate models](./lc0-probe/distance_own.png)

Leela actually reaches 99.3% AUROC on depth 0 and then also degrades quite heavily. No substantial lookahead to be seen in comparison to SF. 

## lc0 by subcategory

![lc0 by reason](./lc0-probe/reasons.png)

Here Leela reaches a perfect score in one category and shows its superiority over SF especially in the attack category. 
Greater is still easier than hanging for some reason. 

## Dataset

Most simple idea: Take the chess commentary dataset but remove all the "dirt".
Instead of doing this myself, just use Claude instead. 

1. Removed all games without any human commentary.
2. Removed all comments that are not valuable/natural language
    | Rule | Example | Caught |
    |---|---|---:|
    | lichess engine verdict | `Inaccuracy. Rfe1 was best.` | 82,888 |
    | lichess result line | `1-0 White wins by checkmate.` | 14,015 |
    | No letters | `!`, `..`, `?!` | 7,096 |
    | Moves only | `30...Rxd7 31.Bh7+ Kf7` | 1,592 |
    | Bare result sentence | `White resigns.` | 625 |
    | Chess Informant code | `#C5 a5+-`, `RR`, `N` | 511 |
    | Bare ECO opening header | `B18: Classical Caro-Kann: 4...Bf5 sidelines` | 371 |
    | URL only | `https://www.chess.com/tactics/46917` | 277 |
    | Informant name | `Chess Informant` | 103 |
    | **Total dropped** | | **107,478** |
3. Histogram by comment lengths
    ![Comment length histogram](./cleaned_commentary/comment_lengths.png)
    i.e. if we take the most valuable 10% we still got 60,000+ left which gives us 25M characters i.e. approx 5M tokens

## Experiments

This list is mostly in hindsight

1. Basic prompt + prompt optimization
2. Advanced prompt + thinking
4. Advanced prompt + prompt optimization + thinking + tools
5. Dictionary learning
3. activation oracle (AO)

## Experiment 1: Gemini 3.5 Flash Lite + basic prompt optimization, judged by Claude Sonnet

The most simply setup imaginable:

We take the 2000 best comments and prompt-optimize Gemini 3.5 Flash Lite without thinking on 1700. Then judge the rest of 300 by Claude Sonnet. 

Starting prompt:

    You are a chess annotator. Given the moves of a game so far and the current board, write a short commentary on the last move played, as a human annotator would in an annotated game.

Best prompt after three rounds:

    Before writing, work through the position silently: identify the piece that moved, its origin and destination square, and what it now attacks, defends, or opens up -- check each claim against the actual diagram, not memory. Do not show this reasoning or use headers.\n\nWrite 1-4 plain sentences in the voice of a human annotator noting their own game -- ordinary prose, no markdown, no bold, no bullets, no hype adjectives (\"thematic\", \"ambitious\", \"solidify\", \"razor-sharp\", \"dynamic\"), no generic filler that would fit any position. Prefer plans, ideas, and evaluations of the position over concrete tactical claims: what the move develops, what square or file it gives up or gains, what plan it continues. Name at most one concrete threat -- a specific piece taking a specific square next move -- and only if you have verified it is actually available on this board; otherwise mention no concrete threat at all. Never say who stands better unless the material count makes it obvious. If nothing concrete and verified stands out, write one short, modest sentence. Output only the commentary.

## Judgement criteria

- **faithful** — every concrete claim about the position, threats, pieces and lines is true.
  5 = all true. 3 = one wrong or unverifiable claim. 1 = mostly wrong, wrong piece/square, or
  talks about a different move.
- **relevant** — it is about the move just played and what matters now, at the same level of
  abstraction the reference uses (a plan, a tactic, an evaluation). 1 = generic opening
  boilerplate or filler that would fit any position.
- **human** — reads like the corpus annotator: a person talking about the game, with a point
  of view and normal length. Penalise engine-voice, bold headers, bullet lists, textbook tone,
  and exaggerated adjectives ("thematic", "ambitious", "solidify", "signals intent").
- **overall** — would a strong club player accept this in place of the reference? Weigh
  faithfulness most.

## Eval

The Sonnet judgements out of 5

| Candidate | n | Overall | Faithful | Relevant | Human | Paired W/T/L | Δ |
|---|---|---|---|---|---|
| baseline | 299 | 2.70 | 3.02 | 3.13 | 2.78 |
| best | 257 | **3.06** | 3.17 | 3.48 | 3.83 |

## Understanding

Essentially, the main point is that be improving the prompt and increasing its length, we can mostly improve human-like behaviour and relevance slightly. However, we cannot improve faithfulness by any relevant measure. For this, we would need a source of truth. Overall 3/5 is a very bad score of course. 

## Experiment 2: Gemini 3.5 Flash Lite with thinking + advanced prompt, judged by Claude Sonnet

Next idea: do the exact same thing as before but feed a 22 page PDF instruction meant for humans and add thinking. 

Also, to prepare the next stage, feed Gemini's thinking into the judge and ask it what tools would have helped Gemini to better follow the instructions and where it went wrong. 

One problem with this: Claude Sonnet is probably not much better than Gemini 3.5 Flash Lite. 

I first took the three publicly available PDFs from the ICS website and fed it into Fable to convert it into Markdown.

Then I asked Fable to set up the experiment. 

## Setup

Only the 300 dev cases first

*Gemini input*

What Gemini gets (one user message per example, gemini-3.5-flash-lite, temperature 0.7, thinking level medium with thoughts returned, max 4000 output tokens):

    The following chess lessons are background reading. Use their ideas and vocabulary where they apply to the position; do not quote or cite them.

    === BEGIN LESSONS ===
    <chess_school_intro.md verbatim, 42,804 chars>
    === END LESSONS ===

    <best.json instruction, unchanged:>

    --- Example ---            (x3: train-0343, train-0956, train-0610, same demos as best.json)
    Moves so far (the last move is the one to comment on):
    <moves>

    Board after <move> (<side> just moved; uppercase = White, lowercase = Black, rank 8 at the top):
    <ascii board>

    Commentary:
    <human comment>

    --- Your turn ---
    Moves so far (the last move is the one to comment on):
    <moves>

    Board after <move> (...):
    <ascii board>

----

[*Judge input*](./commentary_gen/JUDGE2.md)

## Expectations

My expectations were rather low. I assumed the full PDF was too much context and not precisely enough tuned on just commentary/positional analysis. 

However, I still believed there would be some improvement and most importantly I was interested in the main mistakes found and tools requested.

Lovable had already done some research on how LLMs are pretty good at identifying what would help them improve their own work [no citation]. So, maybe this is true here as well.

## Results

I am just using the raw Fable output here

    ┌───────────────────────┬─────────┬──────────┬──────────┬───────┐
    │                       │ overall │ faithful │ relevant │ human │
    ├───────────────────────┼─────────┼──────────┼──────────┼───────┤
    │ r2 winner             │ 3.06    │ 3.17     │ 3.47     │ 3.83  │
    ├───────────────────────┼─────────┼──────────┼──────────┼───────┤
    │ r4 lessons + thinking │ 3.04    │ 3.93     │ 3.20     │ 3.41  │
    └───────────────────────┴─────────┴──────────┴──────────┴───────┘

    Faithfulness jumped because the model asserts less. Human and relevance fell because the lessons' vocabulary leaks straight into the output ("increase the qualitative value of my pieces", "update my to do list") and the voice drifts to textbook. One caveat: the previous run had thinking off (Flash-Lite's default), so lessons and thinking are conflated here.

    Of 300 outputs: 96 contain a false claim, 101 are true but generic, 103 were accepted.

    Mistake categories, built from reading all 300 notes (multi-label):

    - 128 true but generic: the hanging piece, fork, mate or tempo attack the reference is about is never mentioned.
    - 37 phantom or vanished piece: "Nxe7 takes a rook" (it's a bishop), a knight "trapped on a8" two moves after it was traded, "my e2 bishop" that belongs to Black.
    - 35 geometry: Nf3 "defends e4", a rook "hits h5" through its own h4 pawn, Bh6 "attacks g8".
    - 32 evaluation inverted: praises a move the annotator calls a blunder.
    - 31 material or count wrong: "my rooks" with one rook, bishop pair with one bishop, a full rook down unnoticed.
    - 30 square called undefended or vacant that is covered or occupied.
    - 28 side or direction backwards: narrates from the wrong colour, or attacks the wing the king isn't on.
    - 22 light/dark bishop swapped, including the Stonewall and French "bad bishop" both backwards.
    - 18 "open file" with a pawn still on it; 15 rooks "connected" with pieces between; 15 "forced" where alternatives exist or the capture is illegal.

    Thinking vs output: in 25 items the thinking has an error the output dropped, so the hedging instruction works. In 12 the output invents something the thinking never checked. The thought summaries are short and rarely do square-by-square verification; when they do, the misread is already in them.

    Tools the judges asked for, counted the same way:

    - 98 per-piece attack/defence map with blockers.
    - 82 piece inventory: what sits on square X, king locations, material count.
    - 68 short forcing-line search: checks, forks, mates, "is this capture safe".
    - 28 hanging-piece scan, 25 pawn-structure classifier, 21 board predicates like back-rank connectivity.
    - 19 opening book, 18 square colour, 17 per-file pawn census, 12 legal-move list.
    - 7 engine evaluation. Almost never wanted.
    - 94 none, of which 76 are clean items.

    The first two rows are cheap python-chess lookups and cover nearly every factual error. The third is what would separate generic from on-point, and needs a shallow search rather than an engine score.

Further pushing it on the faithfulness vs relevance dynamic

    The reference is the human annotator's original comment on that move, which the judge sees next to Gemini's output. "Never mentioned" means Gemini's text says nothing false, but the concrete fact the human built their comment around is absent from it.

    Example from dev-0148: the human comment points out that Qxe4 wins a knight for free because the f5 pawn is pinned. Gemini wrote about repositioning the a3 knight and king safety. Every sentence is true, so faithful scored 5, but overall scored 1 because it walked past the whole point of the position.

    That was the largest category, 128 of 224 flagged items. The judges were told to grade relevance "at the same level of abstraction the reference uses", so when the human names a fork or a hanging piece and Gemini offers "pressure on the kingside", it gets marked as missing the point.

This is the classical problem even humans face in tests. If you have no idea, your best bet is to just waffle on about irrelevant but factually true things. The diificult part is not saying true things, but it's actually to stay both factual and relevant at the same time. 

Just a general caveat. Due to the size of the test set at 300, it is already difficult to not use an LLM judge and mostly rely on Sonnet and Fable to not just make up stuff or hallucinate. I would not consider this completely serious research at this point. 

These are the specific tools Fable proposed:

    - Attack map: for the moved piece and any named square, list what it attacks and what attacks or defends it, with blockers respected. Covers the geometry, undefended-square, open-file and "rook supports d4 through the queen" errors.
    - Piece inventory: piece-by-square list for both sides after the move, plus material count and king squares. Covers phantom or vanished pieces, wrong captured piece, "my rooks" with one rook, wrong bishop colour.
    - Hanging-piece scan: every piece with more attackers than defenders, pin-aware. Covers the missed free captures and missed counter-threats.
    - Forcing-move probe: all checks and captures for the side to move, one or two plies, with whether each is safe. Covers missed forks, mates and tempo attacks, and the "forced" claims that weren't.
    - Move legality: is this named move legal from here. Cheapest of all, kills the illegal-plan errors.
    - Pawn-structure labels: isolated, doubled, backward, passed, half-open files. Small but the mislabels are consistent.

Some of these seem not absolutely trivial to build (e.g. pawn-structure)

## Experiment 3: Gemini 3.5 Flash Lite with thinking + optimized advanced prompt + tools, judged by Claude Sonnet

First, lets just optimize the prompt a bit more. I wanted to remove most of the part that is about deciding which move to play and just keep the positional analysis part. This also dropped the length quite a bit by almost half. 

Fable built the tools, but I already knew they would be broken, so I let four subsequent reviewers find and eliminate the bugs. For a publication or similar, this would have to be double-checked and maybe even re-written by hand. 

To explain the motivation of using tools a bit more. Tools have several purposes. Firstly, they prevent hallucinations like illegal moves and "Nf3 attacks e4" etc. Secondly, they are one way of "translating" between natural language and deterministic chess logic and thirdly, they give the LLM full control of what it wants and needs to know without predetermining what it should know by precomputing everything and just handing it to the LLM as input. This is both necessary when we have a large search space and just provides more freedom. 

Updates:

1. Minimize the instructions to the minimal necessary. Remove all redundancies and write it into one smooth prompt for the LLM. Make it a step-by-step recipe that can be followed easily. 
2. Add the tools to the prompt and implement tool use
3. Check if the tools that were supposed by the judges are actually used by the LLM in the dev set
4. Do prompt optimization steps on the training set for the full prompt including the shortened chess school prompt and the tool prompts, then run it on the dev set.

## First with all tools executed and injected into prompt

Overall result

    ┌───────────┬─────────┬──────────┬──────────┬───────┐
    │           │ overall │ faithful │ relevant │ human │
    ├───────────┼─────────┼──────────┼──────────┼───────┤
    │ r2 winner │ 3.06    │ 3.17     │ 3.47     │ 3.83  │
    ├───────────┼─────────┼──────────┼──────────┼───────┤
    │ r4        │ 3.04    │ 3.93     │ 3.20     │ 3.41  │
    ├───────────┼─────────┼──────────┼──────────┼───────┤
    │ r5        │ 3.37    │ 4.02     │ 3.66     │ 3.80  │
    └───────────┴─────────┴──────────┴──────────┴───────┘

![Tool performance](./commentary_gen/static_tool_faithful_side_by_side.png)
![Tool coocurrence](./commentary_gen/static_tool_cooccurrence_rownorm.png)

## Second with real tool calling

Tool calling isn't generally an improvement because it is just easier to dump everything into the prompt at this point. 

An `after` tool is added for the engine to play moves into the future.

Generally, the LLM just isn't really helped because it just has no intuition of chess and constantly makes simple mistakes. 

A hand-tuned system prompt with an extremely tight instruction + lots of pointers to tools might make a difference but the core is and will always be rotten. 

    ┌───────────┬─────────┬──────────┬──────────┬───────┐
    │           │ overall │ faithful │ relevant │ human │
    ├───────────┼─────────┼──────────┼──────────┼───────┤
    │ r2 winner │ 3.06    │ 3.17     │ 3.47     │ 3.83  │
    ├───────────┼─────────┼──────────┼──────────┼───────┤
    │ r4        │ 3.04    │ 3.93     │ 3.20     │ 3.41  │
    ├───────────┼─────────┼──────────┼──────────┼───────┤
    │ r5        │ 3.37    │ 4.02     │ 3.66     │ 3.80  │
    ├───────────┼─────────┼──────────┼──────────┼───────┤
    │ r5_tools  │ 3.17    │ 4.06     │ 3.31     │ 3.63  │
    └───────────┴─────────┴──────────┴──────────┴───────┘

![Tool performance](./commentary_gen/tool_faithful_side_by_side.png)
![Tool coocurrence](./commentary_gen/tool_cooccurrence_rownorm.png)

## Three with even tighter prompt + most tools injected + more advanced tools

The prompt is just a bit tighter with an explicit step-by-step procedure and the tools are all injected (additional tools: `change`, `alternatives`, `king_safety`, `piece_quality`, `space`) except for some were the model can play through (`after`, `legal`, `attack_map`). Also an additional tool use judge metric.

    ┌───────────┬─────────┬──────────┬──────────┬───────┬──────────┐
    │           │ overall │ faithful │ relevant │ human │ tool use │
    ├───────────┼─────────┼──────────┼──────────┼───────┼──────────┤
    │ r2 winner │ 3.06    │ 3.17     │ 3.47     │ 3.83  │ —        │
    ├───────────┼─────────┼──────────┼──────────┼───────┼──────────┤
    │ r4        │ 3.04    │ 3.93     │ 3.20     │ 3.41  │ —        │
    ├───────────┼─────────┼──────────┼──────────┼───────┼──────────┤
    │ r5        │ 3.37    │ 4.02     │ 3.66     │ 3.80  │ —        │
    ├───────────┼─────────┼──────────┼──────────┼───────┼──────────┤
    │ r5_tools  │ 3.17    │ 4.06     │ 3.31     │ 3.63  │ —        │
    ├───────────┼─────────┼──────────┼──────────┼───────┼──────────┤
    │ r6        │ 3.76    │ 4.50     │ 3.95     │ 3.98  │ 3.88     │
    └───────────┴─────────┴──────────┴──────────┴───────┴──────────┘

![Score distribution](./commentary_gen/r6_dev_score_distribution.png)

Fable:

    Still open:
    - Deeper tactics are still missed. Examples are the Ruy Lopez pawn won back by a queen fork, gambits, sacrifices and the back-rank mate. The sheet sees only the move and the reply, so these need a real search.
    - Four new notes call a recapture "forced" when other replies are legal. Softening the sheet's wording should fix this.
    - The comparison mixes judges and the per-batch judge spread is wide: 3.28 to 4.32 overall. [Judge prompt](./commentary_gen/JUDGE6.md)

## ChessGPT

Downloaded and tried to run the benchmark on ChessGPT

Problems:
1. That's the data ChessGPT is trained on -> memorization
2. Fine-tuned on 1024 tokens and max limit of 2048 tokens (too little for real reasoning)
3. Fine-tuned too hard -> Lost chat capabailities (or never had any)

## DSL

Proof of Concept (PoC): Just express "This move threatens mate"

## Rust

I chose Rust for the PoC, because
1. it has an expressive `enum` type
2. it is a highly efficient, modern, type and memory safe, functional language
3. it can be compiled to WASM
4. it has a sufficient chess crate (package) ecosystem

-------

```rs
// Simple move
type Move_ = &'static str;

// A reason for a move, e.g. "this move mates in 2", or "this move threatens to win material".
enum Reason {
    And(Box<Reason>, Box<Reason>),
    Mate(Vec<Move_>),
    Threatens(Vec<Move_>, Box<Reason>),
    WinMaterial(Vec<Role>),
}

/// Reasoning about a move in a position
struct Reasoning {
    pos: Chess,
    mov: Move_,
    reason: Reason
}

// Fried Liver, 1.e4 e5 2.Nf3 Nc6 3.Bc4 Nf6 4.Ng5 d5 5.exd5 Nxd5 6.Nxf7 Kxf7 7.Qf3+ Kg8??
let fen = "r1bq1bkr/ppp3pp/2n5/3np3/2B5/5Q2/PPPP1PPP/RNB1K2R w KQ - 2 8";
let pos: Chess = simple_parse(fen);

Reasoning {
    pos: pos.clone(),
    mov: "Qxd5+",
    reason: Reason::Mate(vec!["Qxd5", "Bxd5+", "Be6", "Bxe6#"]),
};
```

## Let Claude ride wild

I then let Claude loose on the whole thing and asked it to create the full DSL

    "[C]omplete the language until we can represent all commentary from the dataset about reasons for moves completely.

    Make a new file dsl_claude.rs and write everything in there

    Keep to the absolute necessary and only write the reasoning plus for each new reason add the example inclu[d]ing the fen position + move and original comment"

## What it did

It wrote a total of 53 `Reason`s each with a natural language description of what they represent:

- Combinators that wrap other reasons: `And`, `Threatens`, `After`, `Allows`, `Prevents`, `Loses`, `Only`, `NoMove`, `Instead`, `Sacrifice`, `Not`, `Because`, `Outweighs`, `Permanent`, `Faster`, `Suppose`.
- Board facts: `Mate`, `WinMaterial`, `Material`, `Trade`, `Attack`, `Pin`, `Defend`, `Overload`, `Decoy`, `Block`, `Open`, `Control`, `Pressure`, `Outnumber`, `Weak`, `Develop`, `Active`, `Better`, `Coordinate`, `Castle`, `Tempo`, `BishopPair`, `Passer`, `Majority`, `Promote`, `Enables`, `Options`, `SameColour`.
- Evaluations: `Simplify`, `Space`, `Initiative`, `Closed`, `Imbalance`, `Zugzwang`, `Draw`, `Win`, `Practical`.

Claude even wrote minimal verifications that didn't need an engine without being asked. These are neither complete nor proofs of any kind, but they at least test plausability. 

## What's missing

Fundamentally
1. No generator -> LLM?
2. No verifier?

My thoughts split by core and language

- Core
    1. Often the dsl representation can be reached from the natural language comment, but the natural language comment would be difficult to reproduce from the reasoning (one-way/reduction)
    2. Just a reasoning without a judgement doesn't make sense often, i.e. we need first the judgement (e.g. this is a solid move), then reason why that judgement. Most judgement calls like brilliant/excellent/good/solid/average/weak/mistake/blunder can be easily verified (e.g. see chess.com/lichess analysis, or simple engine eval delta), others would be more difficult maybe.
    3. A lot of the complexity hides in little `enum` cases (e.g. `Because`)
    4. No clean map between some of our previously determined tactical patterns and the current `Reason`s
- Language
    1. Real commentary has a lot of fluff around it, like player names, hopes, wishes and (often wrong) thoughts, mentions of historical events, etc.
    2. Real commentary has lots of variety for expressing the exact same thing in our DSL (also due to the reduction)
    3. 



## Example 1:

    Reasoning {
        fen: "1rbr2k1/2Rn1ppp/1P1q4/p2p4/P2B4/4P3/4B1PP/3Q1RK1 w - - 1 28",
        mov: "Bd3",
        comment: r#"I was pleased to find this move, which isquiet but effective. The bishop is centralized and now threatens action on thekingside against Black's weakly defended king. Black's pieces are too tied up onthe queenside to be able to defend against White's sudden threats."#,
        reason: And(vec![Active("d3"), Pressure(vec!["h7"])]),
    }

1. Player comments on his own pleasedness about his own move (language)
2. Judgement: Quiet but effective is lost
3. Centralized is a subset of active
4. The general comment of threatening the king side is lost and the fact that the king is weakly defended
5. The fact that "Black's pieces are too tied up on the queen side" and that is the reason for the weakly defended king is also lost. 

## Experiment X: Dictionary learning/SAE/feature attribution

## Motivation

If we look at most models, we have the input layer which is very strongly mapped to the actual input, then we have deeper layers which carry the deeper meaning behind the input which is step-by-step used to at the end reach the final output. 

If we want to understand what the model is really doing, we need to look inside it which is quite tedious, up to impossible for large models and doesn't necessarily generalize over models. Also, single neurons and layers might not always do what we think but have multiple different roles etc. 

If we treat the model as a black box we can only change input and output which gives us information on how the input acts on the output. However, if our input is very plain, this is not really interesting. 

The sparse autoencoder (SAE) is an approach that forces a layer in the model to become more seperable which makes it easier to explain that layer. However, we still need to find out what each neuron actually means which can be quite difficult still. 

Then we have feature attribution which tells you how important a single input feature is. 

My approach is very simple, instead of using the plain input which gives us very minimal information when using feature attribution (see some comments above), we can use a more complex input that gives us much more valuable information.

For example, if our input is just the board, we can understand which pieces are relevant and how much, that's basically it. However, if our input is for example "White has space advantage", then we can say whether the space advantage is important for the evaluation. 

Also, it might be that if we provide very complex features that are not easily replicable in a 10-layer transformer like Leela, we could essentially skip ahead a few layers. This is highly speculative but possible. 

## Features

I am having big problems making a complete list of all possible features if I am honest. 
My perfectionism doesn't allow me to try an incomplete set. 

## Experiment Y: Activation oracle (AO)

Inspired by the paper: [ACTIVATION ORACLES: TRAINING AND EVALUATING LLMS AS GENERAL-PURPOSE ACTIVATION EXPLAINERS (Jan 2026)](https://arxiv.org/pdf/2512.15674)

They quite nicely summarize the idea in this graphic from the paper

![Activation oracle graphic](./image/journey/Screenshot%202026-09-14%20at%2023-59-04%20Activation%20Oracles%20Training%20and%20Evaluating%20LLMs%20as%20General-Purpose%20Activation%20Explainers%20-%202512.15674v2.pdf.png)

The AO is trained on reading the activations (i.e. the hidden thoughts) from the tested model. 
So, maybe we can use this to train an LLM on the activations of Leela. 
Essentially, the model would learn to verbalize the non-verbal thought of Leela. 
If this would actually work, it would be the absolute pinnacle of what we are trying to achieve. 
Especially, considering the fact that transformer models can reach ["Grandmaster-Level Chess Without Search"](https://huggingface.co/papers/2402.04494)

## Problem I am having with the oracle

Suppose we have functions f, we want the oracle to learn, i.e. to be able to provide their answer from the Leela activations when asked: If these functions are deterministically determinable, then how can the oracle be better than agent that has direct access to functions f? The oracle could only become better than the tool use agent if it had sufficient non-generated (i.e. human) data. 

## Other

Adam Karvonen ([GitHub](https://github.com/adamkarvonen)) seems to be an interesting person. He is the author of the AO paper and also did research on GPT3's chess playing ability (where he trained linear probes on GPT), sparse autoencoders (SAEs) and dictionary learning as well. He is currently working at Anthropic (Jan 2026). 

## Caïssa AI
I actually found the Prolog predicate neuro-symbolic engine I was looking for: ["Caïssa AI: A Neuro-Symbolic Chess Agent for Explainable Move Suggestion and Grounded Commentary" (2025)](https://dl.acm.org/doi/10.1007/978-3-032-02813-6_11).

[GPL-2.0 open source code](https://github.com/MazenS0liman/Caissa-AI)

    We present Caïssa AI, a neuro-symbolic chess agent that augments LLM-generated move commentary with symbolic reasoning, knowledge graph integration, and verification modules. Caïssa AI combines a fine-tuned chess-specific LLM with a Prolog-based rule engine encoding chess tactics and rules, along with a dynamically constructed Neo4j knowledge graph representing the current board state. This hybrid architecture enables the system to generate not only accurate move suggestions but also coherent, strategically grounded commentary. A LangGraph-based verification module cross-checks LLM outputs against symbolic logic to ensure consistency and correctness, effectively mitigating hallucinations. By aligning data-driven generation with formal domain knowledge, Caïssa AI enhances both trustworthiness and explainability. Our results demonstrate that this tight neuro-symbolic integration produces verifiable, high-quality commentary and serves as a generalizable blueprint for AI systems requiring real-time, interpretable decision support.

This could definitely be a starting point, but it is lacking a lot on the chess side. Also, I would like to question whether the Neo4j db is actually even valuable or useful

![Tactics and Concepts](./image/journey/Screenshot%202026-09-18%20at%2001-34-53%20MazenS0liman_Caissa-AI%20A%20multi-agent%20system%20that%20deliver%20accurate%20chess%20commentary%20with%20almost%20no%20hallucinations%20unlike%20prior%20methods%20which%20either%20generate%20basic%20commentaries%20or%20with%20logical%20reasoning%20errors.png)

## Paper: "Learning to Generate Move-by-Move Commentary for Chess Games from Large-Scale Social Forum Data" (ACL 2018)

https://aclanthology.org/P18-1154.pdf
https://github.com/harsh19/ChessCommentaryGeneration

1. Variety in commentary -> generator can't know for sure what category the ground truth is.
2. Adding threats as features helps
3. BiLSTM

## Paper: "Unveiling Concepts Learned by a World-Class Chess-Playing Agent" (IJCAI-23)

https://www.ijcai.org/proceedings/2023/0541.pdf

Global explanations and concept probing on basic concepts. 

## Paper: "Communicating Chess Strategies in Natural Language" (13.07.2026)

https://arxiv.org/html/2607.11486v1

Mostly about the prompting techniques how to generate good commentary from a natural language point of view

Key Findings (from Google AI summary, nothing new except for 2 maybe):
1. Evaluating strategies beyond just the primary predicted line ("main line") matters a lot.
2. Purely concept-based descriptions have strict limits because they miss concrete moves.
3. Relying on LLMs to judge these explanations isn't as reliable as real human evaluation.

My understanding: We can differentiate between metrics of natural language and the core of explainability, but there is one major overlap: the relevance/focus.

    Language            Core
    1. Human factor |   
    2. *Focus*      |   1. Correctness
    3. Variety      |   2. Relevance
    4. Context      |

This means: if an explanation is correct and relevant, it is a good explanation, but in a practical setting, you also need to add the human aspect (e.g. humour or emotions), you need to focus on the most important aspect and only expand further when asked, you need to provide variety and you need to keep context (e.g. over a game, over a set of games or over a conversation).

This is why I argue for a two-step process. One that first generates and validates the core explanation in a possibly non-human form. Then another step that generates a natural language explanation based on that core. This makes sure, all metrics and criteria are satisfied. 

The prompt engineering we did, is essentially such a process. The tools provide the core explanation (at least partially) judged as `faithfulness` and `relevance` and the LLM prompted correctly, provides the language factor (summarised as `human`). 

## MA Thesis: "Explainable Chess AI: A Browser Extension for Move Explanation and Analysis in Real-Time Chess Interfaces" (10.12.2025)

https://github.com/NSM1997/chessAI

Interesting stuff

## Paper: "Exploring Explainable Multi-agent MCTS-minimax Hybrids in Board Game Using Process Mining" (24.09.25)

https://arxiv.org/pdf/2503.23326

Not viable for chess (only 3v3 checkers), plus bad interpretability and focuses on what was played, not so much on what wasn't.

## Next

1. LLM that is prompted to analyse a position and conclude an eval -> RL on the eval (expensive, but interesting)
2. [LatentQA](https://arxiv.org/abs/2412.08686)/[activation oracles](https://arxiv.org/abs/2512.15674) -> build a dataset mapping lc0 activations to natural language descriptions starting by eval and them going to more and more complex concepts up to then doing full explanations -> to read what to consider: https://arxiv.org/abs/2607.23379
3. Verify the DSL

## How to get more data?

1. Books -> difficult to obtain, digitalize and bring in a suitable form
2. Lichess studies -> discovery problem
3. **YouTube (YT) Videos**: This is the most interesting, lots of digital data, easy to discover, but difficult to scrape, computer vision and transcription problems

Claude's research result on that

    The research note is at paper/yt-shorts-extraction.md (not committed). The agent tested the pipeline on one GothamChess Short (pKuCQA3HXck) and deleted the downloaded video afterwards.

    Recommended pipeline, starting with Shorts that screen-record a digital board:

    1. Listing: yt-dlp --flat-playlist on a channel's /shorts tab. This worked.
    2. Transcript: YouTube's auto-captions come with a timestamp for every word, free. This worked. faster-whisper with a chess-vocabulary prompt is the fallback where captions are missing.
    3. Board reading: tsoj/Chess_diagram_to_FEN (MIT licence) found the board and read the exact position from a full 9:16 frame with facecams, at both 1080p and 360p. Nearly all its errors were frames caught mid-move.
    4. Frames to moves: match each new frame against the legal moves with python-chess. A rough version rebuilt a fully legal 32-ply game from the test Short and threw out 23 noisy frames. It broke when a move happened between two samples (it sampled 2 frames a second). This step needs a proper search over legal moves that can bridge a missed move, plus a reset when the video cuts to a new position. The agent found no existing code for this; it's the one part we'd have to write.
    5. Checking positions: Lichess puzzle DB, cloud eval, the masters explorer (needs a Lichess token) and Stockfish.
    6. Output: PGN with a set-up start position and comments tied to move ranges, the same shape as cleaned_commentary/.

    Prior work: ChessGPT built a set from about 83K YouTube videos by reading boards from frames. They judged it too noisy to use and didn't release it, citing legal issues. Agadmator-2K (arXiv 2609.00474) lines up about 500 hours of narration with games whose moves were already known, so it doesn't read boards at all. The agent found no dataset made from Shorts.

    Main risks:
    - Legal: YouTube's terms forbid downloading and automated access. The German research exception (§60d UrhG) covers making copies for non-commercial research, not publishing them. We could release only video IDs, timestamps, PGN and code. Shorts under a CC BY licence are the only fully clean subset.
    - Commentary with no chess in it: the test Short's board was read correctly, but its audio was pure reaction ("HOLY MOLY… I win on time?"). Expect a filter for chess-relevant speech to drop a large share.
    - Commentary out of step with the board: speakers talk ahead of or behind what's on screen.
    - Untested visuals: only chess.com's default board was tried. Arrows, other themes, flipped boards and physical boards haven't been tested, and physical boards are much harder.
    - Unknown start state: side to move and castling rights can't be read from the first frame.
    - Spoken notation: how well Whisper transcribes moves like "knight takes e5" hasn't been measured.

    The cheapest next step is probably to run the pipeline on about 50 Shorts from one channel and measure how many survive the chess-relevance filter. That number decides whether this source is worth building.

I agree with almost everything Claude says: This should work in general, but some filtering still needs to be done and there are still some technical difficulties. However, you could probably find a channel that almost exclusively publishes a certain type of content in comparison to Gotham who does a lot of different formats. 

For example, here is an extraction from the YT channel "You Me And Chess"

    Fishing Pole Trap — You Me And Chess
    Start: the normal starting position, White to move.

    The Fishing Pole Trap starts with:

    1. e4 e5 2. Nf3 Nc6 3. Bb5
    White attacks Black's knight.

    3… Nf6
    Black ignores it.

    4. O-O Ng4
    Attacking the pawn on h2.

    5. h3
    White gets extremely uncomfortable.

    5… h5
    Instead of moving the knight.

    6. hxg4
    White gets happy and captures the knight.

    6… hxg4
    Black captures back, and now the knight on f3 is under attack.

    7. Ne1
    So it moves to e1.

    7… Qh4
    The crushing move, threatening mate on h2.

    8. f3
    To create room for the king on f2. But it's too late:

    8… g3
    Black takes away that square.

    9. a3
    Now White has to make a random move.

    9… Qh2# Checkmate. 0-1

This looks really clean!

## Tree-search explanation

Lots of explanations are about the lines that do not work -> not only explain why a line works, but also why one doesn't. 

How do we do this?

How do we find the balance between abstract, positional considerations and concrete lines?

How do we determine moves that seem reasonable but aren't and moves that are very good, but don't seem reasonable?

-> Reasonability is highly subjective -> MAIA-3?

## Questions

1. Tracing the Thought of a Grandmaster-level Chess-Playing Transformer
2. Feature attribution/oracle
3. Compute

## Feedback
1. Read cited papers again
    - Transcoder (SAE but multi-layer)
    - Lorsa
2. AO: only valuable for complex, non determinable functions
    1. Just start with simple, then test whether the complex works
        - CLIP
        - Board description
        - Eval
    2. Features: 
        - Hand-written SF features
        - Anything, almost everything, I find interesting
3. FU compute cluster
4. Videos unreliable maybe?
5. AO on 2000 examples for natural language rendering of DSL  
    -> How to get the activations into an LLM?
6. Trying to get chess knowledge into an LLM will be very expensive/not feasible
7. Start with Qwen 3.5 probably

# Fourth meeting 28.09.


## DSL
First, I applied for the cluster compute, then I decided to continue the DSL by adding more reasons but most importantly judgements, comparisons, "but", "either" and much more 

Reflection: we use concepts to make it easier for us to express more complex or simply lengthy explanations, e.g. a relative pin is nothing else but saying that e.g. the knight can't move because it would allow the bishop to take the queen behind it. 

Realisation: One of the most important things in move commentary is to find creative and insightful ways of describing moves. However, this is not just a language 

I did some work on the basic evaluator to get a feeling for how easy or difficult it is to get it working. Conclusion: most things are very difficult to definitively define. 

The next step was doing a deterministic natural language render of our DSL and the corresponding demo. That actually works quite nicely: [Render demo](xai-chess.netlify.app/render). 

![Render demo example](./image/journey/Screenshot%202026-10-01%20at%2018-15-42%20Reason%20Renderer%20·%20xai-chess.png)

Essentially, now only four things are left to be done on this path:
1. A generator 
    - either an AlphaZero-style transformer (this is Googles IMO-gold-medalist AlphaProof setup ported to chess)
    - or an (evolutionary) optimization algorithm
2. Optimizing the DSL itself (by hand)
3. An evaluator, that can accurately assert things
    - Rule-based utilizing the engine
4. A renderer, that turns the DSL into interesting human-like commentary
    - Probably some well-prompted small LLM

The generator is responsible for creativity and relevance, the evaluator for truthfulness and the renderer for the language.

The DSL lies at the heart of it all. It has to meet demands from all sides. 

On top of move reasoning this approach could also be adapted to positional analysis, etc. 

## GANs

While thinking about the problem, I thought about Generative Adversarial Networks (GANs) as possible models. The advantage is that the discriminator doesn't necessarily need fully-supervised data. For example, the discriminator could just learn what human commentary looks like. The renderer than learns to fool the discriminator when turning the DSL into human commentary. No 1-to-1 DSL->human gold dataset would be needed necessarily.   

Problem: 
    1. not good for discrete values like text
    2. mode collapse 
    3. little data
    4. could loose relation to DSL content
Solution: 
    - build some kind of transformer GAN model??? -> maybe inspired by DiffusionGemma???
        -> language diffusion models can take corrections while they generate!!?
    - some more supervised data
    - reconstruction loss??

## How to train the generator in general

- Train for relevance
- Train for diversity
- Train for a negative correlation between brevity and position complexity

-> How to do that?

My first instinct was to use the delta between MAIA-3 and Stockfish for some of these. For example, for knowing which alternatives to explain or to show which reasons support the Stockfish move, but not the MAIA move or for determining complexity. 

For diversity, we can essentially do the same as for images, and write a similarity score which we try to maximize cross-sample. 

## Generator suggestions by Claude

(after I had already told it my ideas)

1. Teach a weak solver. Score an explanation by how much it raises Maia's probability of the played move when Maia is restricted to moves where the explanation holds, minus a cost for the explanation's size. Atoms Maia already believes earn nothing. The rating is a reader-level knob. This is your "engine sees, human doesn't" idea made precise.
2. Distill the refutation tree. This gives a principled source for the lines inside Threatens/Allows/Instead, which are the expensive part:
   - A null move after the played move gives its threat: Threatens(line, X).
   - A null move before it gives the opponent's threat it stops: Removes/Prevents.
   - The opponent's replies are grouped by how each is refuted, producing one Allows(reply, After(refutation, X)) per group.
   - The rule is to cover all of the opponent's replies with as few groups as possible.
3. Contrastive minimal atom set. Choose the fewest atoms that hold for this move and fail for every clearly worse alternative. It's the same move-vs-alternatives comparison as idea 1, with a uniform weak player and so weaker. It also tends to pick geometric accidents.
4. Minimal intervention on the position. Make small board edits (the Suppose vocabulary) until the move stops being best; the atoms that flip along with that decision are what drive it. Very faithful, but costs 100–500 searches per position.
5. Counterexample-guided critic (CEGIS/Anchors). Look for positions or moves where the explanation holds but the move is bad, and add conditions to rule them out. Best used as a final check, not as the generator.
6. Argumentation trees. Claims, rebuttals and counter-rebuttals map onto Threatens/Allows/But. They end in Outweighs/Balances, which the evaluator doesn't check.
7. Sparse surrogate model. Learn, across the corpus, which atom differences predict that the engine prefers one move over another, and explain a move by its largest contributors. Correlational, and weak on one-off tactics.
8. E-graph normalisation. Rewrite candidate explanations into canonical, human-like form. It cleans up output but adds no content.

## Similarity score

I concluded that the next most important thing was building a similarity score (a metric). This is necessary for evaluations and training.

- -1 is max diff, 1 is max closeness
- `Not` is negation
    -> negative means contradictory
- Completely different things with no similarity are 0

## Features

All binary, per player, assuming there are no concrete tactics

0. Context
    - Game phase
    - Side to move
1. Material
    - total material
        - which pieces are traded
    - exactly equal
    - +1
    - +2
    - +3 or more
    - similar material but not exactly
        - knight vs bishop
        - two knights vs bishop
        - two pieces vs rook
        - two rooks vs queen
        - three pieces vs queen
        - etc.
    - bishop pair
    - unlike bishops
2. Local superiority (really difficult to determine)
    - Center, Queenside, Kingside x 0-3
3. King's safety
    - King uncastled
    - Kingside castle
    - Queenside castle
    - Pawn structure in front of King breached
    - More defenders
    - Equal
    - More attackers
        - 1
        - 2
        - 3
        - 4
4. Quality of pieces
    - For each piece p:
        Any combo of 
            - p out of play
            - p weak
            - p strong
        - Judged by
            - Mobility: 
                1. how many fields can the piece move to and 
                2. how little moves does the piece need to cover the entire board
            - Positioning
                1. knight in the center 4x4
                2. other pieces on open lines/diagonals
                3. the more central and the more squares under control, the better
            - Role: tiered
                1. Out of play
                2. Defensive
                3. Offensive
                4. Both offensive and defensive
            - Stability: Can it be removed from its position easily?
                - Existence of outposts -> potential
            - Depends on piece
                - knights: foreposts
                - bishops: open diagonals
                - rooks: open files, behind mobile pawns, not behind blocked pawns, on the 7th rank, 
    - Development (only for opening)
        - 0
        - +1
        - +2
        - +3 or more
    - Grade of coordination
        - Very good
        - Good
        - Solid
        - Neutral
        - Not good
        - Bad
        - Very bad
        - Judged by -> difficult to assess
            1. Sustainment
            2. Protection
            3. Limitation
            4. Attack
            5. Obstruction
5. Pawn structure
    - number of pawn islands
        - 1
        - 2
        - 3
        - 4
    - number of doubled pawns
        - 0
        - 1
        - 2
        - 3+
    - number of pawns that can be attacked easily, and can't be moved forward or defended well (weak pawns)
    - weak squares
    - number of (protected) past pawns
    - Center type (global)
        - dynamic: undefined
        - open: two open files between c and f including
        - closed: entire center blocked by chains of pawns
        - static: 2 half-open or one open file
        - mobile: one player has mobile pawns with no counter
    - Center control (zero-sum)
        - 0
        - +0.5
        - +1
        - +1.5
        - +2
        - +2.5
        - +3
        - +3.5
        - +4
    - Mobile pawn majority
        - Center
        - Queenside
        - Kingside
6. Space
    - More space on the kingside
    - More space in the center
    - More space on the queenside
7. In the initiative
    - Opponents moves are forced (i.e. little options for the opponent)



2. Qualitative value of pieces
    - Mobility: 
        1. how many fields can the piece move to and 
        2. how little moves does the piece need to cover the entire board
    - Positioning
        1. knight in the center 4x4
        2. other pieces on open lines/diagonals
        3. the more central and the more squares under control, the better
    - Role: tiered
        1. Out of play
        2. Defensive
        3. Offensive
        4. Both offensive and defensive
    - Stability: Can it be removed from its position easily?
        - Existence of outposts -> potential
    - Grade of coordination -> difficult to assess
        1. Sustainment
        2. Protection
        3. Limitation
        4. Attack
        5. Obstruction

## Left out
For PMEs (positions with material equivalence) also:
1. play-coordinating piece is valued higher

5. Pieces
    10. Piece-by-piece comparison

## Training an SAE

The problem: One layer, one position has 64 squares × 256 = 16,384 numbers.

We have 230 binary features, we should have around 65-130k. 

Approaches:
1. Raw board: We just reinsert the board as a piece bitmask. This gives us 650 new features for free and helps reconstructing the actual board information
2. Global features for each square: We just take all ~1k features and copy them for each square
3. Free variables: the free variables are mostly responsible for capturing the non-strategic aspects, like tactics. Measured 256 via exponential elbow method
4. Free active variables: 256 trivially allows full reconstruction. So anything reasonable like 32-64 works theoretically

The problem: We use global features in a local setting and the network just tries to go around our features as much as possible. The next question is how we actually evaluate the importance of each feature. 

## LIME

The idea for LIME is to make a locally interpretable model, i.e. you train a model for a specific position. You do that by creating synthetic data through pertubation. For chess, perturbation is pretty difficult but can be done by randomly inserting pawns for example. 

My first instinct was to randomly insert an equal amount of pawns for both sides. Here are some learnings:
1. Also remove pawns to learn more
2. LIME maxes out at a few thousand samples (3000)
3. Filter out perturbations where pawns create immediate tactics (e.g. attack a piece, give check our threaten immediate promotion)
4. A 4-layer MLP doesn't outperform a simple linear model, but the linear model is simpler to read
5. Doing these pawn perturbations is best for understanding pawn structure features and actually works. For example, it tells us that more pawn islands is bad

        ┌─────────────────┬─────────┬───────────────────────────────┐
        │ Black's islands │ samples │ Leela's Q for White (average) │
        ├─────────────────┼─────────┼───────────────────────────────┤
        │ 1               │ 16      │ +0.14                         │
        ├─────────────────┼─────────┼───────────────────────────────┤
        │ 2 (book)        │ 1357    │ +0.25                         │
        ├─────────────────┼─────────┼───────────────────────────────┤
        │ 3               │ 410     │ +0.35                         │
        ├─────────────────┼─────────┼───────────────────────────────┤
        │ 4+              │ 39      │ +0.57                         │
        └─────────────────┴─────────┴───────────────────────────────┘

6. Interesting idea: purposefully try to find perturbations that only edit a single feature and changes nothing else (i.e. isolate the feature) -> Problem: very difficult and often impossible to avoid confounding factors -> that's why we train the model
7. In general, unguided perturbations aren't that valuable because they just confound to much too often (already seen with Shapley from another paper where a bishop and pawns were confounding, making the pawns less valuable due to the presence of the bishop)
8. This currently doesn't seem to work in general, because we don't consider any tactics at the moment

Idea: learn what a normal chess position looks like. Bias towards those in the perturbation (e.g. don't allow impossible positions, etc.)

## SAE: Features by square

On the other hand, we could focus on only the pieces and squares and construct the features per square. We would need only 1-2k features per square, of which only a few **need** to be fixed by us. 

Goal: get 1k features

Whenever this mentions a square or move, this is a 64 vector referencing the exact one. Piece is square x type

I thought it would be interesting to try to also go for some tactical aspects for this:
- Part of a mate in 1-15 plies
    - #plies
- Part of a material winning tactic
    - #plies
- Pinning which piece to which
    (piece type x piece type x direction x distance x distance)
- Giving check
- Threatens any of the above
    - + move

Positional:
- Material
    - Which piece occupies the square
    - Is a piece responsible for (not necessarily the only one)
        - +1
        - +2
        - +3 or more
        - similar material but not exactly
            - knight vs bishop
            - two knights vs bishop
            - two pieces vs rook
            - two rooks vs queen
            - three pieces vs queen
            - etc.
            - is this the play coordinating piece (ICS definition)
        - bishop pair (includes other bishops position)
        - unlike bishops (includes other bishops position)
        <!-- - the entire vector just like the one for this but for one to two comparable pieces -->
- Stability:
    - Which pawn(s) could remove it from it's position ever
    - Which pawn(s) can directly remove it from it's position
    - Which piece of lesser or equal value can attack it in x moves
        - ply x piece type x square 
- Control:
    - Which squares does the piece reach
    - Another version with geometry in mind (batteries, bishops/queens behind pawns, etc.)
- Mobility (relative means for example, that a knight can't go on a square, where a pawn attacks him): 
    - Which available moves (absolute)
    - Which available moves (relative)
    - Which moves to reach square x fastest (absolute)
    - Which moves to reach square x (relative)
    - Which pieces are blocking it (seperated by own and opponents)
- Centralisation
    - Manhattan distance to edge of the board (one-hot)
- Weak/Strong square -> TODO: definition
- On which open file
- On which half open file
- On which long diagonal
- Role
    - Defensive
        - the piece is defending which pieces
        - the piece is needed to defend which pieces
    - Attacking
        - Which pieces the piece is attacking
            - square x piece type
    - Restricting
        - How many fields the piece is restricting 
            - square x piece type
            - e.g. a knight might be restricting both a rook and a queen from moving to a certain square
    - Dependent:
        - Needs to be defended (which defenders)
    - Blocking
        - Blocks which line
        - Blocks which diagonal
        - Blocking which pawn(s)
    - Which pawns are blocking attacks on this square
        pawn x which direction (8)
    - Probe on qualitative  of piece trained on supervised data (at least 2000 positions by piece type)
        1. Out of play
        2. Defensive
        3. Offensive
        4. Both offensive and defensive
- Protected by which pieces
    - type x squares
- If it's a King
    - King Safety -> probes like above
    - Castling rights
- If it's a queen
    - Safe and active (attacking) -> probe like above
- If it's a rook
    - Controls which open file
    - On the 7th rank
    - On the 8th rank
    - Battery with which other rook/queen
    - Behind which passed pawn
    - Behind which hanging pawns
    - 
<!-- - If it's a bishop
- If it's a knight -->
- If it's a pawn
    - What is the pawn chain
    - Is it at the front, base or in the middle of the chain
    - Which direction is the pawn chain pointing towards
    - Is it a past pawn
    - Is it a hanging pawn
    - Is it an isolated pawn
    - Is it a doubled/tripled/quadrupled pawn
        - other pawns rank
    - is it being blocked
        - square x piece type
    - Is part of a center by type
        - dynamic: undefined
        - open: two open files between c and f including
        - closed: entire center blocked by chains of pawns
        - static: 2 half-open or one open file
        - mobile: one player has mobile pawns with no counter
    - pawn on 4th, 5th or 6th rank
    - pawn on 7th rank

**This results in 30k features per square not just 1k**

Not done yet:
- Candidate passer: can become passed through favorable exchanges or a pawn break.
- Connected passers: adjacent passers that support or advance together.
- Outside or remote passer: far from the main pawn mass, potentially distracting the enemy king.
- Rook pawn: has special drawing properties, especially with the wrong-colored bishop.
- Pawn duo: adjacent pawns on the same rank, often controlling forward squares and retaining flexible advance choices.
- Majority and minority: numerical superiority on a wing; assess whether it can create a passer or provoke a weakness.
- Compactness and connectivity: mutually supporting pawns versus scattered targets.

## Probing on the per-square features

The first step is probing the features in lc0. The idea is that if the features can be constructed from the layers, we can probably find a way through back-propagation or similar to find some kind of attribution???

I was satisfied with just probes on the embedding layer and layer 5 for the beginning

    Where layer 5 adds a lot (gain in AUROC over the embedding):

    ┌────────────────────────────────────────┬────────────┬────────────┬────────────────┐
    │                 Group                  │ Embedding  │  Layer 5   │      Gain      │
    ├────────────────────────────────────────┼────────────┼────────────┼────────────────┤
    │ square.control                         │ 0.762      │ 0.943      │ +0.18          │
    ├────────────────────────────────────────┼────────────┼────────────┼────────────────┤
    │ threat.mate                            │ 0.727      │ 0.887      │ +0.16          │
    ├────────────────────────────────────────┼────────────┼────────────┼────────────────┤
    │ strong / weak                          │ ~0.855     │ 0.96–1.00  │ +0.11 to +0.14 │
    ├────────────────────────────────────────┼────────────┼────────────┼────────────────┤
    │ material_win                           │ 0.716      │ 0.827      │ +0.11          │
    ├────────────────────────────────────────┼────────────┼────────────┼────────────────┤
    │ threat.material_win                    │ 0.717      │ 0.813      │ +0.10          │
    ├────────────────────────────────────────┼────────────┼────────────┼────────────────┤
    │ mate                                   │ 0.893      │ 0.987      │ +0.09          │
    ├────────────────────────────────────────┼────────────┼────────────┼────────────────┤
    │ pawn_cover (white/black, now and ever) │ ~0.85–0.92 │ ~0.96–1.00 │ +0.08 to +0.12 │
    ├────────────────────────────────────────┼────────────┼────────────┼────────────────┤
    │ pawn.passed, pawn.isolated             │ ~0.96      │ ~0.99      │ +0.03 to +0.04 │
    └────────────────────────────────────────┴────────────┴────────────┴────────────────┘

    Where layer 5 is worse than the embedding:

    ┌────────────────┬───────────┬─────────┬────────┐
    │     Group      │ Embedding │ Layer 5 │ Change │
    ├────────────────┼───────────┼─────────┼────────┤
    │ long_diagonal  │ 1.000     │ 0.855   │ −0.15  │
    ├────────────────┼───────────┼─────────┼────────┤
    │ centralisation │ 1.000     │ 0.966   │ −0.03  │
    ├────────────────┼───────────┼─────────┼────────┤
    │ pawn_shield    │ 0.985     │ 0.963   │ −0.02  │
    ├────────────────┼───────────┼─────────┼────────┤
    │ control.*      │ ~0.995    │ 0.976   │ −0.02  │
    ├────────────────┼───────────┼─────────┼────────┤
    │ mobility.*     │ ~0.99     │ ~0.98   │ −0.01  │
    ├────────────────┼───────────┼─────────┼────────┤
    │ route.*        │ ~0.99     │ ~0.98   │ −0.01  │
    └────────────────┴───────────┴─────────┴────────┘

In general these were the worst (61 included to show the sudden drop)

    ┌─────┬───────────────────────┬──────┬─────────┬───────────┬────────┐
    │ 61  │ square.control        │ 7    │ 0.943   │ 0.762     │ +0.181 │
    ├─────┼───────────────────────┼──────┼─────────┼───────────┼────────┤
    │ 62  │ threat.mate           │ 15   │ 0.887   │ 0.727     │ +0.160 │
    ├─────┼───────────────────────┼──────┼─────────┼───────────┼────────┤
    │ 63  │ long_diagonal         │ 2    │ 0.855   │ 1.000     │ −0.145 │
    ├─────┼───────────────────────┼──────┼─────────┼───────────┼────────┤
    │ 64  │ material_win          │ 15   │ 0.827   │ 0.716     │ +0.111 │
    ├─────┼───────────────────────┼──────┼─────────┼───────────┼────────┤
    │ 65  │ threat.material_win   │ 15   │ 0.813   │ 0.717     │ +0.096 │
    └─────┴───────────────────────┴──────┴─────────┴───────────┴────────┘

- [Full table](./lc0-probe/results_square_encoder.md)
- [Chart: score by feature family](lc0-probe/square_families.png)

Most importantly the encoder is almost perfect (> 90% of feature families > 0.95 fvu) except for the tactical part

I then also tested the hypothesis that tactical patterns increase at later stages, which turned out to be only partially true, specifically however for the most difficult (`material_win`, `threat.material_win`, `mate` and `threat.mate`):

    ┌─────────────────────┬──────┬───────────┬─────────┬─────────┬─────────┐
    │       family        │ bits │ embedding │ layer 5 │ layer 8 │ layer 9 │
    ├─────────────────────┼──────┼───────────┼─────────┼─────────┼─────────┤
    │ mate                │ 3    │ 0.893     │ 0.987   │ 0.987   │ 0.989   │
    ├─────────────────────┼──────┼───────────┼─────────┼─────────┼─────────┤
    │ material_win        │ 15   │ 0.716     │ 0.827   │ 0.832   │ 0.834   │
    ├─────────────────────┼──────┼───────────┼─────────┼─────────┼─────────┤
    │ xray                │ 385  │ 0.990     │ 0.992   │ 0.987   │ 0.982   │
    ├─────────────────────┼──────┼───────────┼─────────┼─────────┼─────────┤
    │ check               │ 1    │ 0.991     │ 1.000   │ 1.000   │ 1.000   │
    ├─────────────────────┼──────┼───────────┼─────────┼─────────┼─────────┤
    │ threat.mate         │ 15   │ 0.727     │ 0.887   │ 0.882   │ 0.900   │
    ├─────────────────────┼──────┼───────────┼─────────┼─────────┼─────────┤
    │ threat.material_win │ 15   │ 0.717     │ 0.813   │ 0.822   │ 0.825   │
    ├─────────────────────┼──────┼───────────┼─────────┼─────────┼─────────┤
    │ threat.xray         │ 10   │ 0.988     │ 0.988   │ 0.987   │ 0.978   │
    ├─────────────────────┼──────┼───────────┼─────────┼─────────┼─────────┤
    │ threat.check        │ 1    │ 0.983     │ 0.994   │ 0.996   │ 0.994   │
    ├─────────────────────┼──────┼───────────┼─────────┼─────────┼─────────┤
    │ threat.move         │ 55   │ 0.982     │ 0.985   │ 0.985   │ 0.982   │
    ├─────────────────────┼──────┼───────────┼─────────┼─────────┼─────────┤
    │ all                 │ 500  │ 0.964     │ 0.978   │ 0.974   │ 0.970   │
    └─────────────────────┴──────┴───────────┴─────────┴─────────┴─────────┘

## Building the Decoder

The decoder makes the per-square SAE complete. On layer 5 the FVU (fraction of unexplained variance) is 39%, i.e. we can explain 61% of variance. If we include the square identity, we can almost exactly reconstruct the embedding (the input), which makes sense. Only an error of 0.2% remains. So, I wanted to run the decoder on the whole network: 
<!-- 
    ┌───────────┬─────────────┬───────────────────┬──────────────┐
    │    tap    │ square only │ features + square │ features add │
    ├───────────┼─────────────┼───────────────────┼──────────────┤
    │ embedding │ 0.164       │ 0.002             │ 0.16         │
    ├───────────┼─────────────┼───────────────────┼──────────────┤
    │ layer 0   │ 0.575       │ 0.183             │ 0.39         │
    ├───────────┼─────────────┼───────────────────┼──────────────┤
    │ layer 1   │ 0.818       │ 0.198             │ 0.62         │
    ├───────────┼─────────────┼───────────────────┼──────────────┤
    │ layer 2   │ 0.828       │ 0.195             │ 0.63         │
    ├───────────┼─────────────┼───────────────────┼──────────────┤
    │ layer 3   │ 0.832       │ 0.272             │ 0.56         │
    ├───────────┼─────────────┼───────────────────┼──────────────┤
    │ layer 4   │ 0.825       │ 0.311             │ 0.51         │
    ├───────────┼─────────────┼───────────────────┼──────────────┤
    │ layer 5   │ 0.841       │ 0.392             │ 0.45         │
    ├───────────┼─────────────┼───────────────────┼──────────────┤
    │ layer 6   │ 0.849       │ 0.427             │ 0.42         │
    ├───────────┼─────────────┼───────────────────┼──────────────┤
    │ layer 7   │ 0.871       │ 0.447             │ 0.42         │
    ├───────────┼─────────────┼───────────────────┼──────────────┤
    │ layer 8   │ 0.879       │ 0.549             │ 0.33         │
    ├───────────┼─────────────┼───────────────────┼──────────────┤
    │ layer 9   │ 0.897       │ 0.680             │ 0.22         │
    └───────────┴─────────────┴───────────────────┴──────────────┘ -->

![Chart: decoder per layer](lc0-probe/square_decoder_grid.png)

Note: we only use features that occur at least 20 times in the training set. On average 36.5 bits are on for any given square. 

What we can see here quite clearly is "where in the network" our features "are located". They mainly "live" in the first few layers (see fourth graph, "Features add" peaks at L1 and L2), which absolutely makes sense because the features are simple in nature. The deeper into the network we go, the more attention has happened and the more difficult it is for our features to compete with the model. We already saw how some features are more advanced and appear more clearly later and others are more clear in the beggining. On top of that, there might be some information which is only relevant for the policy head, but not for the value head. The other problem of course is, that features are still not complete and it is difficult to determine when they would be. 

Out of curiosity, I also tried an MLP run (with 10% less data due to early stopping)

![Ridge vs MLP](./lc0-probe/square_decoder_ridge_mlp.png)

But first, just to check, I wanted to look at decoding dense1, dense2 and the final output of the value head. 
Interestingly this overfits massively. So, I applied scaling laws and just added more data. My conclusion: if you trained this at 1M, 10M or 100M, at some point something should happen.
However, it's probably not the best approach, especially because at this point we are using per-square to predict a global value. Additionally, of course, the trend continues that the farther we go into the model, the more difficult it is to predict it directly.

    ┌──────────────┬───────┬───────┬───────┬───────┐
    │              │ 20k   │ 50k   │ 100k  │ 151k  │
    ├──────────────┼───────┼───────┼───────┼───────┤
    │ dense1 train │ 0.320 │ 0.333 │ 0.299 │ 0.280 │
    │ dense1 test  │ 0.709 │ 0.625 │ 0.562 │ 0.525 │
    │ dense1 gap   │ 0.39  │ 0.29  │ 0.26  │ 0.25  │
    ├──────────────┼───────┼───────┼───────┼───────┤
    │ dense2 train │ 0.323 │ 0.426 │ 0.385 │ 0.226 │
    │ dense2 test  │ 0.738 │ 0.671 │ 0.589 │ 0.546 │
    │ dense2 gap   │ 0.42  │ 0.25  │ 0.20  │ 0.32  │
    ├──────────────┼───────┼───────┼───────┼───────┤
    │ output train │ 0.546 │ 0.368 │ 0.318 │ 0.437 │
    │ output test  │ 0.815 │ 0.730 │ 0.664 │ 0.622 │
    │ output gap   │ 0.27  │ 0.36  │ 0.35  │ 0.19  │
    └──────────────┴───────┴───────┴───────┴───────┘

Possible extrapolation

![Extrapolate direct value decoding](./lc0-probe/value_decoder_scaling.png)

Now, we also want to remove the policy bias in reconstruction, i.e. the information we can't reconstruct because it is not meant for the value head, but for the policy head. For this we could possibly add extra features for the policy, but my idea was to measure not only the reconstruction error on the layer for the layer itself, but also on the win-draw-loss (WDL) output from that layers forward pass. 

    ┌───────────────────────────────┬───────────────┬─────────────┬───────────────┬─────────────┐
    │             layer             │ ln1 token FVU │ ln1 WDL FVU │ ln2 token FVU │ ln2 WDL FVU │
    ├───────────────────────────────┼───────────────┼─────────────┼───────────────┼─────────────┤
    │ embedding (/attn_body/rehape) │ 0.003         │ 0.031       │               │             │
    ├───────────────────────────────┼───────────────┼─────────────┼───────────────┼─────────────┤
    │ 0                             │ 0.170         │ 0.092       │ 0.045         │ 0.291       │
    ├───────────────────────────────┼───────────────┼─────────────┼───────────────┼─────────────┤
    │ 1                             │ 0.179         │ 0.476       │ 0.110         │ 0.470       │
    ├───────────────────────────────┼───────────────┼─────────────┼───────────────┼─────────────┤
    │ 2                             │ 0.175         │ 0.579       │ 0.138         │ 0.485       │
    ├───────────────────────────────┼───────────────┼─────────────┼───────────────┼─────────────┤
    │ 3                             │ 0.249         │ 0.631       │ 0.221         │ 0.631       │
    ├───────────────────────────────┼───────────────┼─────────────┼───────────────┼─────────────┤
    │ 4                             │ 0.284         │ 0.747       │ 0.240         │ 0.728       │
    ├───────────────────────────────┼───────────────┼─────────────┼───────────────┼─────────────┤
    │ 5                             │ 0.365         │ 0.861       │ 0.314         │ 0.825       │
    ├───────────────────────────────┼───────────────┼─────────────┼───────────────┼─────────────┤
    │ 6                             │ 0.398         │ 0.868       │ 0.315         │ 0.920       │
    ├───────────────────────────────┼───────────────┼─────────────┼───────────────┼─────────────┤
    │ 7                             │ 0.424         │ 1.150       │ 0.459         │ 1.050       │
    ├───────────────────────────────┼───────────────┼─────────────┼───────────────┼─────────────┤
    │ 8                             │ 0.524         │ 0.990       │ 0.481         │ 1.134       │
    ├───────────────────────────────┼───────────────┼─────────────┼───────────────┼─────────────┤
    │ 9                             │ 0.664         │ 1.043       │ 0.689         │ 0.822       │
    └───────────────────────────────┴───────────────┴─────────────┴───────────────┴─────────────┘

This is what happens if we just do the forward path on the decoded embedding layer through the whole network

    ┌───────────┬─────────────┬────────────┬─────────────┬────────────┐
    │   layer   │ ln1 forward │ ln1 direct │ ln2 forward │ ln2 direct │
    ├───────────┼─────────────┼────────────┼─────────────┼────────────┤
    │ embedding │ 0.003       │ 0.003      │             │            │
    ├───────────┼─────────────┼────────────┼─────────────┼────────────┤
    │ 0         │ 0.008       │ 0.170      │ 0.004       │ 0.045      │
    ├───────────┼─────────────┼────────────┼─────────────┼────────────┤
    │ 1         │ 0.004       │ 0.179      │ 0.003       │ 0.110      │
    ├───────────┼─────────────┼────────────┼─────────────┼────────────┤
    │ 2         │ 0.004       │ 0.175      │ 0.004       │ 0.138      │
    ├───────────┼─────────────┼────────────┼─────────────┼────────────┤
    │ 3         │ 0.005       │ 0.249      │ 0.005       │ 0.221      │
    ├───────────┼─────────────┼────────────┼─────────────┼────────────┤
    │ 4         │ 0.005       │ 0.284      │ 0.005       │ 0.240      │
    ├───────────┼─────────────┼────────────┼─────────────┼────────────┤
    │ 5         │ 0.007       │ 0.365      │ 0.006       │ 0.314      │
    ├───────────┼─────────────┼────────────┼─────────────┼────────────┤
    │ 6         │ 0.007       │ 0.398      │ 0.006       │ 0.315      │
    ├───────────┼─────────────┼────────────┼─────────────┼────────────┤
    │ 7         │ 0.007       │ 0.424      │ 0.008       │ 0.459      │
    ├───────────┼─────────────┼────────────┼─────────────┼────────────┤
    │ 8         │ 0.009       │ 0.524      │ 0.010       │ 0.481      │
    ├───────────┼─────────────┼────────────┼─────────────┼────────────┤
    │ 9         │ 0.015       │ 0.664      │ 0.020       │ 0.689      │
    ├───────────┼─────────────┼────────────┼─────────────┼────────────┤
    │ WDL       │ 0.031       │            │             │            │
    └───────────┴─────────────┴────────────┴─────────────┴────────────┘

I tried training on WDL directly, but it is just extremely difficult to train the decoder through the whole network directly on WDL with so little epochs and data. It is almost like trying to retrain the whole engine. Maybe some hybrid could work, i.e. pretraining on the layer and then optimizing on the WDL. 

I instead trained on layer 5 and wdl simultaneously (20k positions), because the pretraining wouldn't stay by it's own

    ┌────────────────────────┬───────────┬─────────┐
    │        decoder         │ token FVU │ WDL FVU │
    ├────────────────────────┼───────────┼─────────┤
    │ tokens only (epoch 15) │ 0.367     │ 0.758   │
    ├────────────────────────┼───────────┼─────────┤
    │ tokens + WDL, epoch 1  │ 0.617     │ 0.519   │
    ├────────────────────────┼───────────┼─────────┤
    │ tokens + WDL, epoch 2  │ 0.583     │ 0.524   │
    ├────────────────────────┼───────────┼─────────┤
    │ tokens + WDL, epoch 3  │ 0.558     │ 0.532   │
    ├────────────────────────┼───────────┼─────────┤
    │ tokens + WDL, epoch 4  │ 0.532     │ 0.478   │
    ├────────────────────────┼───────────┼─────────┤
    │ tokens + WDL, 5 epochs │ 0.512     │ 0.510   │
    ├────────────────────────┼───────────┼─────────┤
    │ 10 epochs (best of 20) │ 0.499     │ 0.488   │
    └────────────────────────┴───────────┴─────────┘

This seems to work to some degree, but doesn't get past the fundemental problem. 

Another option was just trying more positions

    ┌───────────┬───────────┬─────────┐
    │ positions │ token FVU │ WDL FVU │
    ├───────────┼───────────┼─────────┤
    │ 20,000    │ 0.365     │ 0.861   │
    ├───────────┼───────────┼─────────┤
    │ 50,000    │ 0.352     │ 0.772   │
    ├───────────┼───────────┼─────────┤
    │ 100,000   │ 0.344     │ 0.727   │
    ├───────────┼───────────┼─────────┤
    │ 151,614   │ 0.341     │ 0.708   │
    └───────────┴───────────┴─────────┘

This also works on the embedding layer

    ┌──────────────┬───────────┬─────────┐
    │              │ token FVU │ WDL FVU │
    ├──────────────┼───────────┼─────────┤
    │ 20k decoder  │ 0.003     │ 0.031   │
    ├──────────────┼───────────┼─────────┤
    │ 100k decoder │ 0.002     │ 0.010   │
    └──────────────┴───────────┴─────────┘

Now the question still remains: How do we get the relevance/feature attribution from our decoders. 

## Feature attribution

Our features aren't the best there are, but they are a start. 

![Feature map](./image/journey/Feature%20map.png)

My thought process is followingly:
1. If we start at the very back of the net, we can almost perfectly reconstruct embeddings, but most of it is just from square identity and then we have to fight through "the deep jungle of attention". In the end, your attributions will be very weak and mostly trivial. 
2. If we start further at the front, we have an increasing reconstruction loss the more we move to the front, but the attribution becomes clearer as well.
3. Therefore we should use a hybrid approach were we combine the results in some way?
4. The best would be to have some kind of gold standard for attribution, maybe 500 positions where we clearly define what is relevant. This could let us actually compare different relevance metrics and show where they succeed and where they fail 
5. I feel like **attributions are currently not able to express/differentiate** between different logical constructs. For example, whether features are seperately important(+) or if they are only important together (AND/x) or if any are important (e.g. one side is up a pawn, then any pawn is relevant, but really it's just important that one side is up a pawn). This is especially critical if we try to compare negatives (i.e. features that are not on, but could be). Of course it would be amazing to manifest a queen somewhere but that's not interesting (however we could use it to find the optimal place for a piece and checking how far away it is from there)
6. I believe there has to be some comparison to the "normal" relevance of a feature to account for this, i.e. a relative relevance
7. What I like about some of these methods is the possibility to do impossible edits e.g. manifest half a queen and looking at what the network invents there. Because they are differentiable and mostly linear, you would expect some sensible conclusion, e.g. half a queen is worth half that of a full queen? But is half a queen there and half a queen here also as good as one full queen, or is it even better?

> Currently we have per-square features and global features, maybe we need a bit more clear cross-square features

> Also, still open but interesting: we could go back to SF8's hand-crafted eval function, it would definitely give us some interesting information. Even if it's worse than modern engines, it's still much better than any human and it's an expert model not a learned NN we have to interpret. 

> **Fundamental problem**: feature attribution doesn't really solve our fundamental correlation vs causation problem

## 1. Just trying stuff: Feature attribution on features through embedding layer and attention

My first try was to use the embedding decoder with a 1% error on the WDL output and just do feature attribution on it. As I said, this discards most of the interesting information which can't be in the embedding layer (e.g. tactics, complex cross-square patterns)

The immediately obvious problem: The occupant feature is way too dominant, which makes sense of course, also simply because it is the most frequent by far; we are still throwing out features that occur less than 20 times, which is also the most distinctive and tactical of course. 

![Demo](./image/journey/Feature%20attribution%20demo.png)

I also wanted to compare the features to their usual relevance. Queens are always somewhat relevant, so probably it shouldn't count if a queen is just relevant, it shuld be more relevant than usual.

![Relative relevance demo](./image/journey/Feature%20attribution%20relative%20to%20usual.png)

Still the demo just doesn't show that the only winning move by far is Queen f6 because it forces mate or Black to sacrifice their queen. 

So, I choose less tactical positions for the demo and added an option to hide certain families, like occupies. 

On top, I also tried training the model with all features, even those that never appear, which slightly decreases reconstruction error:

    ┌─────────────────────────────┬───────────┬───────────┬────────────────────────────────────────┐
    │  decoder (100k, embedding)  │ bits used │ token FVU │                WDL FVU                 │
    ├─────────────────────────────┼───────────┼───────────┼────────────────────────────────────────┤
    │ bits seen at least 20 times │ 17,165    │ 0.002     │ 0.010 (first 2,000 held-out positions) │
    ├─────────────────────────────┼───────────┼───────────┼────────────────────────────────────────┤
    │ all bits                    │ 30,839    │ 0.0016    │ 0.0086 (all held-out positions)        │
    └─────────────────────────────┴───────────┴───────────┴────────────────────────────────────────┘

However, this still doesn't really make the cut, so we should probably move up a layer

## 2. Choosing the best layer

As we discussed, the embedding layer is simply not good, because it reduces some of the more interesting features and mostly just relies on the piece squares. Instead, we want a layer where we still have a very low reconstruction loss but that doesn't rely on the pieces positions anymore. This is were our earlier features add chart comes in handy. It is maximised for Layers L1 and L2, specifically L2. So, let's try that instead. 

    ┌─────┬─────────────────────┬───────────┬─────────┐
    │ Tap │ Positions, features │ Token FVU │ WDL FVU │
    ├─────┼─────────────────────┼───────────┼─────────┤
    │ ln1 │ 20k, min 20         │ 0.175     │ 0.579   │
    ├─────┼─────────────────────┼───────────┼─────────┤
    │ ln1 │ 100k, min 1         │ 0.162     │ 0.438   │
    ├─────┼─────────────────────┼───────────┼─────────┤
    │ ln2 │ 20k, min 20         │ 0.138     │ 0.485   │
    ├─────┼─────────────────────┼───────────┼─────────┤
    │ ln2 │ 100k, min 1         │ 0.125     │ 0.400   │
    └─────┴─────────────────────┴───────────┴─────────┘

**Problems**: 
1. The elephant in the room: a 40% error is really bad
    - I tried solving this by filtering for positions where our error is minimal. However, that mostly just filters for totally lost or won positions. 
2. This still attributes the piece positions mostly 
    - We could add a small training loss for using those?
    - But: some things are becoming more interesting when looking at relative relevances
        - e.g. in the endgame the king is worse at its castled position then normally
3. Problem I found: our training dataset is a set of tactical positions, which most importantly effects the relative scores and the performance on non-tactical positions

**Solutions:**
1. Use L0 ln2, it's maybe the sweet spot?
2. Do L1 regularization (Lasso) on that feature family
3. New dataset of 100k random positions

        ┌─────────────────────────┬───────────┬─────────┐
        │ ln2 decoder             │ Token FVU │ WDL FVU │
        ├─────────────────────────┼───────────┼─────────┤
        │ L0, 20k, min 20         │ 0.045     │ 0.291   │
        ├─────────────────────────┼───────────┼─────────┤
        │ L0, 100k, min 1         │ 0.039     │ 0.077   │
        ├─────────────────────────┼───────────┼─────────┤
        │ L0, 100k, min 1, Lasso  │ 0.043     │ 0.371   │
        ├─────────────────────────┼───────────┼─────────┤
        │ L2, 100k, min 1         │ 0.125     │ 0.400   │
        └─────────────────────────┴───────────┴─────────┘

Without regularization L0 ln2 has <8% loss on the WDL output, with regularization the loss explodes, so it seems like we can't just leave those out. The 8% is also partially misleading because  

The next step is using integrated gradients instead. It is the theoretically most sound feature attribution method, and it performs quite well. Firstly, it can correctly sort the value of the pieces, but also just in general it actually seems to work, whereas the simple, minimal feature attribution method didn't work at all and seemed almost random.

![L0 integrated gradients feature attribution](./image/journey/L0_ig_feature_attribution_screenshot.png)

What I noticed:
1. Material win seems quite arbitrary
2. Still some bugs in the feature definitions, features missing and partially bad features, that are abused
3. For example, that Black is blocked by its own piece is considered bad for White, even if that makes no sense

## Compute

As soon as I got access to the compute cluster, I tried to learn how to get a cluster running, initially just using vllm inference. The next step would then be doing CLIP, fine-tuning and RL, all using LoRA of course. I had applied for 64 GBs and 8 CPU cores, which should even be enough for the highly-praised Qwen3.8-27B. 

I first tried to run the full model (55 GB), but then I saw that I would probably only get a "small" GPU (2x 1080 Ti = 22 GB), for which I got the quantized "GGUF" version from Unsloth (17 GB). 

1. Install `uv` and latest python, create a venv and install dependencies (e.g. huggingface_hub)
    - `curl -LsSf https://astral.sh/uv/install.sh | sh`
    - `uv python install 3.13`
    - `uv venv --python 3.13 ~/vllm`
    - `source ~/vllm/bin/activate`
    - `uv pip install -U vllm -v`
2. Activate the venv
3. Set the HF `export HF_HOME=/scratch/$USER/hf`
4. tmux `tmux new -s dl`
5. Download the full model `hf download Qwen/Qwen3.8-27B`
6. Notice that doesn't work 
7. Create the download folder `mkdir -p /scratch/$USER/gguf`
8. Download 4-bit GGUF `hf download bartowski/Qwen3.8-27B-GGUF --include "*Q4_K_M*" --local-dir /scratch/$USER/gguf`
9. Clone llama.cpp source `cd ~ && git clone --depth 1 https://github.com/ggml-org/llama.cpp && cd llama.cpp`
10. Load toolchain `module load GCC/14.2.0 CMake/3.31.3-GCCcore-14.2.0 CUDA/12.9.1`
11. Build llama.cpp `cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=61`

For inference

How to list ressources `sinfo -N -O NodeList,Partition,Gres:30,GresUsed:30,StateCompact`

```
srun --qos=hiprio --partition=gpu --gres=gpu:2 --cpus-per-task=8 --mem=32G --time=3:00:00 --pty bash
module load GCC/14.2.0 CMake/3.31.3-GCCcore-14.2.0 CUDA/12.9.1
~/llama.cpp/build/bin/llama-cli -m /scratch/$USER/gguf/Qwen3.8-27B-Q4_K_M.gguf -ngl 99 -c 8192
```

    > Hello

    [Start thinking]

    The user simply said "Hello". This is a basic greeting, so I should respond warmly and briefly, inviting them to continue the conversation.
    [End thinking]

    Hello! How can I help you today?

    [ Prompt: 53.5 t/s | Generation: 15.3 t/s ]

## LLaVA

The first step to building an oracle is doing LLaVA, the goal is to bring the leela activations into the same space as the LLM tokens and then fine-tune the LLM on it. 

The main difference to existing CLIP on chess or LLM oracles is that the first just reads the chess board itself, which leaves all the thinking to the LLM, and that LLM activations are more like other LLMs activations which could mean that it is easier to move them into the right space. 

To do the fine-tuning quickly and reliably on the 1080s, I chose to start with the 0.8B model. It's fast for fine-tuning and we don't need lots of general knowledge which is what most of the parameters are used for in most models. 

0. Get the code on the server  
    `rsync -av --exclude runs [local dir] [name]@[host]:[remote dir]`
1. Setup venv 
    ```
    export UV_CACHE_DIR=/tmp/$USER-uv HF_HOME=/scratch/$USER/hf
    uv venv --python 3.13 ~/oracle && source ~/oracle/bin/activate
    uv pip install torch --index-url https://download.pytorch.org/whl/cu126
    uv pip install transformers accelerate numpy torchvision pillow
    ```
2. Install model  
    `hf download Qwen/Qwen3.5-0.8B`  
    `rm -rf /tmp/$USER-uv`
3. Get a GPU:  
    `srun --qos=hiprio --partition=gpu --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=3:00:00 --pty bash`
4. Training  

        source ~/oracle/bin/activate
        export HF_HOME=/scratch/$USER/hf HF_HUB_OFFLINE=1
        nvidia-smi
        cd ~/xai-chess && python oracle/train.py --out oracle/runs/q08b


## Feedback

**Open Questions**:  

1. Correlation vs Causation
2. Solve vanishing gradients + features over several layers
    - Skip connections + SAE/decoder at every layer
3. Search for logical/programming language metrics
    - Not decidable for complete programming languages
    - Most research looking at fraction of examples that two logical statements match (Jaccard similarity)
    - Best other approach:
        - Canonical AST similarity
    - https://github.com/ali-ballout/Learning-to-Classify-Logical-Formulas-based-on-their-Semantic-Similarity


# Fifth meeting 15.10.

## Feature Attribution: Shapley

Now the trick to using SHAP efficiently is doing it only after we have the integrated gradients (IGs) result. SHAP on its own can be very expensive especially at 30,000 features á 64 squares

## Feature Attribution: Measuring features at different layers simultaneously

Basically, the idea is to decode several layers at the same time. 

If we decode e.g. L0, then L2 and then L5, L0 catches most of the accuracy and the additional decoding on L2 and L5 don't add that much. 
For that reason, we have to do it the other way round. I did some experimentation and then came up with this result:

![Multi-layer feature attribution](./image/journey/Multi-Layer%20Feature%20Attribution.png)

## Notes

1. Was sick unfortunately
2. LLaVA works mostly
3. Many interesting finds
    - Paper: Language Models that Play Chess and Explain Their Moves (Oct 26, Princeton)
        - Flamingo architecture (first VLMs, 2021) with lc0 layers interleaved into SmolLM3-3B
        - [Reddit](https://www.reddit.com/r/chess/comments/1wypz4z/princeton_researchers_train_a_4b_llm_to_reach/) thread 
        - Small SmolLM3-3B model (4B and 8B final variants), would be interesting to see larger models
            - Also tried Qwen and Gemma
        - Flamingo is old VLM architecture, would be interesting to see more modern VLM architecture
            - -> LLaVA and directly passing to kv cache worse 
        - Q&A dataset with positions from public Lichess dataset and 4 types
            - Static/Dynamic x Current/Future
            - They train on these categories sequentially
                > This progression teaches the model to first extract all features of the encoded board state; future-position stages introduce an additional state-tracking challenge: the model must first reconstruct the board state resulting from the provided move sequence before answering the query. Question types from earlier stages are replayed at low proportions in subsequent stages to prevent forgetting. In all stages, the encoder and decoder are both frozen 
            - This model can already answer general questions!!!
        - Now comes the shady part: They essentially distill GPT-5.6-Sol by letting it generate the supervised training data. 
        - They use an interesting search distillation idea
         
        - Commentary very verbose
        - Possibly cherry-picked examples
        - They say it can be extended, but don't show how
        - They eval by accuracy, substantiation (relevance?) and coherence (language)
        - 
    - Paper: [LLAMIA](https://arxiv.org/pdf/2609.00474) (Aug 26, )
        - LLAMIA-Bench: six tasks
        - They also had the YT idea, and got the data from Agadmator's YT channel
        - Verbalization debt
        - Latent state internalization: Interleaving chain of language, action and latent space tokens
        - Engine activations are called by the LLM as the result of a tool call
    - [Tutorial](https://www.youtube.com/watch?v=GEJOB_TFYJ0): Training DeepSeek directly to play chess without Leela with GRPO
    - Reading: 
        - Flamingo (TODO)
        - DeepSeek R1 (TODO)

# Sixth meeting 

## First Experiments

First, I just trained a LLaVA projector on L9 for Qwen-3.5 0.8B (34x smaller than 27B). Then I tried supervised fine-tuning using LoRA and RL using GRPO with reasoning enabled. 0.8B is actually a vision model and so I wanted to find out if using the existing 2D image slot would be better by any chance. 

50k positions (48-1-1 split), different answer formats to check language generalization, AdamW (1e-3, 2e-4, 1e-5 learning rate), 
1. Just the projector: Already works quite well
2. sft: Supervised fine-tuning
3. vlm: Using the 2D image slot
4. grpo: RL a la Deepseek
5. r1zero: RL without any supervised fine-tuning + thinking

On the H100 it went really fast as well (30 min for SFT, 3 hours for r1zero)

```
Each cell shows bin right, then the average eval error in pawns.

┌────────────────────────────────┬─────────────┬─────────────┬─────────────┬────────────┬────────────┐
│            Question            │   no LoRA   │     sft     │     vlm     │    grpo    │   r1zero   │
├────────────────────────────────┼─────────────┼─────────────┼─────────────┼────────────┼────────────┤
│ What is the evaluation?        │ 98% / 0.22  │ 99% / 0.06  │ 99% / 0.09  │ 99% / 0.09 │ 99% / 0.12 │
├────────────────────────────────┼─────────────┼─────────────┼─────────────┼────────────┼────────────┤
│ What's the eval?               │ 95% / 0.13  │ 98% / 0.06  │ 98% / 0.09  │ 95% / 0.07 │ 97% / 0.12 │
├────────────────────────────────┼─────────────┼─────────────┼─────────────┼────────────┼────────────┤
│ Who is better, by how much?    │ 100% / 0.11 │ 100% / 0.07 │ 100% / 0.07 │ 99% / 0.05 │ 98% / 0.09 │
├────────────────────────────────┼─────────────┼─────────────┼─────────────┼────────────┼────────────┤
│ How good for the side to move? │ 99% / 0.06  │ 100% / 0.05 │ 98% / 0.07  │ 98% / –    │ 98% / –    │
├────────────────────────────────┼─────────────┼─────────────┼─────────────┼────────────┼────────────┤
│ Assess this position.          │ 96% / 0.07  │ 99% / 0.02  │ 96% / 0.06  │ 94% / 0.00 │ 93% / –    │
├────────────────────────────────┼─────────────┼─────────────┼─────────────┼────────────┼────────────┤
│ Better, worse, or equal?       │ 96% / 0.16  │ 99% / 0.03  │ 99% / 0.09  │ 99% / –    │ 97% / 0.00 │
├────────────────────────────────┼─────────────┼─────────────┼─────────────┼────────────┼────────────┤
│ How is it for {colour}?        │ 50% / 4.98  │ 64% / 0.44  │ 58% / 0.32  │ 59% / 2.99 │ 57% / 7.17 │
├────────────────────────────────┼─────────────┼─────────────┼─────────────┼────────────┼────────────┤
│ All except colour              │ 97% / 0.15  │ 99% / 0.06  │ 98% / 0.08  │ 97% / 0.07 │ 97% / 0.10 │
└────────────────────────────────┴─────────────┴─────────────┴─────────────┴────────────┴────────────┘
```

- Bins: Losing (< −2.0), Clearly worse (−2.0 to −0.7), Slightly worse (−0.7 to −0.3), Equal (−0.3 to +0.3), Slightly better (+0.3 to +0.7), Clearly better (+0.7 to +2.0), Winning (> +2.0)
- Eval error only when eval explicitly given as number

Interesting findings: 
1. Regular SFT works best
2. Just using the 2D image slot doesn't work better than SFT
3. R1-Zero doesn't work -> doesn't think at all.
    Probably a result of too strict thinking limit (512 tokens) and the model simply not needing to think. 
4. Leela doesn't actually know which color is which (at least in L9). Leela is always encoded with White to move (50% without LoRA). 
    The fascinating part is that the fine-tuning was actually able to get the side to move from just the position. This makes sense if you think about the fact that some positions, especially in the early game, mostly appear for one side or the other. 
5. Results match LLaVA runs from the QUEEN paper


## Next steps

Mostly running tests again as actually intended

- Use the full vision encoder 
- Use the actual base model for r1zero and allow more reasoning tokens
- Use the AO approach to the injection. Instead of overwriting tokens, we only add after first attention layer, rescaled to the tokens natural attention.

Of course the next step is to identify further training objectives all using regular SFT. Some candidates:

1. Best move
2. PV (scored by accuracy + length when accurate)
3. Which squares does piece x control
4. Which pieces does piece x control (should include squares)
5. Game stage
6. Consequences of a move (deterministic)
7. Hand labeled datasets (e.g. 500 to 1000 positions) -> we need a process to build these reliably?
6. LLAMIA-Bench, etc.
7. Move description

Table 7 of [Language Models that Play Chess and Explain Their Moves](https://arxiv.org/html/2610.03695v1) (Bhaskar, Cheng, Chen, Princeton).

![Four-stage domain adaptation question types](./image/journey/Four-stage%20domain%20adaptation%20question%20types.png)

Still thinking about the following idea to train a teacher: The teacher writes a move description or reason and the student has to say the move. 
The teacher is trained on decriibing the move as shortly as possible without actually encoding the move (i.e. cheating)


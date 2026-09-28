# Upstream terms for the annotated-PGN corpus

`Waterhorse/chess_data` is tagged `apache-2.0` on the Hub, but for `annotated_pgn` the dataset
card grants nothing itself — it lists four upstream sites and leaves their terms to apply. This
is what those four actually say. Checked 2026-09-05. Not legal advice.

## Summary

| Source | Comments | Who holds the rights | Any licence to redistribute? |
|---|---|---|---|
| GameKnot | 353,954 (48%) | GameKnot — annotators assign their rights to the site | No |
| lichess studies | 324,182 (44%) | the study author; lichess holds a sub-licensable licence | No |
| PGNlib | 42,941 (6%) | the original annotators, individually | No — the compiler had none either |
| Path to Chess Mastery | 14,520 (2%) | the blog's author | No — no terms published |

None of the four grants a public redistribution licence. Apache-2.0 covers ChessGPT's own
modeling artifacts; it does not reach this text.

## GameKnot — https://gameknot.com/pg/pol_eula.htm

The strictest of the four, and the largest share of the corpus. Annotators do not merely
license their notes to GameKnot, they hand them over:

> you hereby exclusively grant and irrevocably assign to our licensors and us all rights of any
> kind or nature throughout the universe to such Content (including all ancillary and subsidiary
> rights thereto …) in any languages and media now known or not currently known

The EULA has no clause permitting third-party copying of site content, and no research or
quotation carve-out. The site also answers HTTP 403 to automated requests, so the links the
viewer renders may not resolve — a signal about scraping, not a bug in the viewer.

## lichess studies — https://lichess.org/terms-of-service

A study is user content. What the author grants runs to lichess, not to the public:

> By submitting or displaying content on or through our website or services, you grant us a
> worldwide, non-exclusive, royalty-free license (with the right to sub-license that content)

The ToS separately names redistributing annotations as misconduct:

> Abusing copyrighted material or intellectual property — Examples of these would be the sharing
> or distribution of chess literature, puzzle sets, annotations, or any other material or
> intellectual property you do not have the right to share or distribute.

Its "Licenses" section covers the software (Lila, AGPL-3.0-or-later) and pushes the question back
to the reader: "Anyone intending to use or replicate the website and services should do their own
research and due diligence into the various licenses used."

**Do not confuse this with the CC0 database.** https://database.lichess.org states "Database
exports are released under the Creative Commons CC0 license … download, modify and redistribute
them, without asking for permission." Those exports are played games. Study chapters — the
prose in this corpus — are not in them and are not covered by that grant.

## PGNlib — https://www.angelfire.com/games3/smartbridge/ (dead; via the Wayback Machine)

The domain no longer resolves; the last snapshot is
[2025-07-13](http://web.archive.org/web/20250713172857/https://www.angelfire.com/games3/smartbridge/).

PGNlib is not a rights holder. It is one person's re-format of annotated games collected from
elsewhere, and the page says so:

> I want to share the result of my work, because most of the games are really precious. I hope
> the original annotators will forgive me for using their comments, in the interest of all chess
> fans

So this slice is third-hand: ChessGPT scraped a compilation that was itself assembled without
permission. The page credits the actual annotators per file, and our filenames preserve that —
`chessdoctor` (NM Corey Russell), `electronic_campfire` (Tim McGrew), `hartwig` (Thomas Hartwig),
`europe_echecs` (G. Bertola, Europe Échecs), `d00_chess_informant` (Chess Informant sample games),
plus files credited to Eric Schiller, David Hayes, Andrew Martin and Jan van Reek. Chess Informant
and Europe Échecs are commercial publishers. Rights here are individual and unresolved.

## Path to Chess Mastery — https://www.pathtochessmastery.com/

A single-author Blogger site. No terms of service, no licence, no copyright notice — which under
Blogger's terms leaves the author holding ordinary copyright. The PGN `Annotator` tag is
"ChessAdmin/Houdini", the blog's own handle.

## What follows for this repo

- Building and reading `commentary.html` locally is ordinary use of a published research dataset.
- Redistributing the sample — committing a built `commentary.html` to a public repo, shipping it
  in a release, putting it on a website — republishes ~39k comments none of these sites licensed.
  Keep it local, or rebuild from the Hub on each machine.
- Quoting individual comments in a paper is the usual fair-dealing/fair-use question and is a much
  smaller ask than bulk redistribution. Attribute the annotator where the corpus preserves them:
  the `Annotator` PGN tag, the lichess study URL, or the PGNlib filename.
- If a run needs a clean-rights subset, `lichess_studies` chapters whose author released them
  is the only lane with any path to permission, and it has to be asked for per study.

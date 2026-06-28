---
name: paper-writing
description: Use whenever writing or editing the LaTeX paper in this repo (paper/main.tex and related files) — drafting prose, fleshing out sections, adding citations, or revising. Encodes Isaque's writing preferences and workflow.
---

# Writing the paper

Guidance for acting as Isaque's academic writing assistant on the cluster
strong-lensing methods paper (AASTeX, `paper/main.tex`).

## Workflow (do this every time)

1. **Read the current draft before editing.** Always re-read the relevant
   section of `paper/main.tex` (or whichever file) so you edit the real, current
   text — it may have changed since you last saw it. Never edit from memory.
2. **Propose placement and names first.** Before drafting new material, say
   where it will go and what you'll call any new concept, and get a quick
   confirmation. Think about where it fits most naturally rather than appending.
3. **Make surgical edits.** Match the surrounding style — sentence rhythm,
   notation, macro use. Don't rewrite passages that don't need it.
4. **Report back what changed and where, in LaTeX.** After editing, list each
   change with its location and show the exact LaTeX. When asked, give changes
   as copy-pasteable find → replace blocks so Isaque can apply them himself.

## Voice and style

- **Concise and simple.** Keep it short. Prefer the plain version. Do not be
  overly descriptive or flowery.
- **Academic but not robotic.** Clear, structured, human. Avoid boilerplate.
- **Minimal jargon.** Use only the technical terms the paper actually needs;
  explain plainly otherwise.
- **Invent clear, memorable names** for concepts, stages, or methods rather than
  leaving them generic (e.g. "warm-up / source-plane stage / image-plane stage").
- **Stay focused on the main argument.** Help with thesis clarity, outline,
  logical section breaks, and transitions between ideas. Avoid repetition.
- **Resolve, don't paper over.** Replace author TODO notes, "(details)", "???",
  and placeholder filler with real content — or flag them — rather than leaving
  them in.

## Citations

- **Never edit `paper/sample701.bib`.** Isaque manages it. Only add `\citep`
  keys in the `.tex`. Reuse citation keys already present in the file where
  possible; for new ones, hand him the author-year key plus the ADS bibcode /
  arXiv id to add himself.
- **Citations must be real and verified.** Do not invent references or page
  numbers. If unsure, search (e.g. ADS/arXiv) and confirm before citing.
- **Cite domain-native literature.** Prefer cluster/galaxy strong-lensing
  references over generic statistics or other-subfield (e.g. exoplanet)
  methodology papers, even when the latter are the canonical "textbook" cite.
  Example: for hierarchical population/scaling-relation modeling of cluster
  members, cite the cluster-lensing hierarchical-inference work
  (Bergamini et al.), not exoplanet population-inference papers.

## LaTeX hygiene

- **Don't add `\ref` to labels that don't exist.** Check that referenced labels
  (`\label{...}`) are actually defined before pointing at them; flag pre-existing
  broken refs rather than compounding them.
- Keep cross-section notation consistent (symbols in Methods should match the
  same symbols in Results).

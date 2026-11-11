# Domain docs

## Layout and discovery

Single-context: `GLOSSARY.md` at the repo root; decisions in `docs/adr/`.
Before exploring, read the glossary and ADRs relevant to the area being changed.
If absent, proceed silently. Create domain docs lazily through domain-modeling when terms or decisions are resolved.

## Vocabulary

Use glossary terms in issue titles, proposals, hypotheses, and tests. Respect explicitly avoided synonyms.
If a concept is missing, reconsider whether it belongs or note the gap for domain-modeling.

## Decisions

Surface conflicts with an ADR explicitly, with its identifier and the reason to reconsider.
ADRs on the default branch are immutable except for status and links to superseding or amending ADRs. Record changed decisions in a new ADR.

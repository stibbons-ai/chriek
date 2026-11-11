# Issue tracker: GitHub

Issues and specs live in GitHub Issues for `stibbons-ai/chriek`.
Use `gh` inside this checkout; when `AGENT_GH_APP` is set, use `~/.agents/bin/agent-gh` instead.

## Operations

- Create: `gh issue create --title "..." --body "..."`.
- Read: `gh issue view <number> --json number,title,body,labels,comments`.
- List: `gh issue list --state open --json number,title,body,labels`.
- Comment: `gh issue comment <number> --body "..."`.
- Label: `gh issue edit <number> --add-label "..."` or `--remove-label "..."`.
- Close: `gh issue close <number> --comment "..."`.

When a skill says to publish to the issue tracker, create a GitHub issue. When it says to fetch a ticket, read that issue.

## Pull requests as a triage surface

**PRs as a request surface: no.**

Set to `yes` to include external PRs in triage; use the corresponding `gh pr` operations.

# GitHub Progress and Bug Tracking

Use this workflow to track research progress, bugs, and decisions consistently.

## 1) Labels to use

- `type:bug` - bugs and regressions
- `type:progress` - weekly/milestone updates
- `type:task` - implementation tasks
- `type:experiment` - ablation or benchmark runs
- `status:triage` - new issue, not yet confirmed
- `status:in-progress` - actively worked on
- `status:blocked` - waiting on dependency/resource
- `status:review` - ready for review
- `status:done` - completed
- `priority:p0` / `priority:p1` / `priority:p2`

## 2) Issue types

- Use `Bug report` template for failures, regressions, metric drops, or crashes.
- Use `Progress update` template weekly (or per milestone) to record:
  - completed work
  - blockers
  - next actions
  - metric snapshot

## 3) Milestone suggestion (16-week plan)

- `W01-W02 preprocess`
- `W03-W04 vanilla-sam`
- `W05-W06 prompt-generator`
- `W07-W10 sam-lora-cascade`
- `W11-W12 nnunet-baseline`
- `W13-W14 ablation-robustness`
- `W15-W16 final-results-writeup`

## 4) Working loop

1. Create issue before coding.
2. Assign label + milestone + owner.
3. Create branch: `feat/<issue-id>-short-name` or `fix/<issue-id>-short-name`.
4. Open PR using the PR template and link issue.
5. Close issue only when merged and validated.

## 5) Minimal bug triage SLA

- P0: acknowledge within 4h, fix target <24h
- P1: acknowledge within 24h, fix target <3 days
- P2: acknowledge within 48h, schedule in next sprint

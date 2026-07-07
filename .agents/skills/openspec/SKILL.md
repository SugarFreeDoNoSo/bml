---
name: openspec
description: Manages spec-driven development with OpenSpec. Use when creating/updating change proposals, tracking tasks, validating specs, archiving completed work, or querying project status. Triggers: "openspec", "change proposal", "spec-driven", "tasks", "validate spec", "archive change", "OpenSpec status", "openspec/specs", "openspec/changes".
---

# OpenSpec — Spec-Driven Development

OpenSpec organizes work as **changes** (proposals with tasks) that evolve the canonical **specs**. It lives at `openspec/` in the project root with `openspec/config.yaml` defining the schema (typically `spec-driven`).

## Quick Reference

```bash
openspec list                    # List active changes
openspec list --specs            # List canonical specs
openspec status --change <name>  # Show task completion progress
openspec validate <name>         # Validate a change
openspec validate --all          # Validate everything
openspec archive <name> -y       # Archive completed change → specs
openspec new change <name>       # Scaffold a new change
```

## Directory Structure

```
openspec/
├── config.yaml                  # Schema: spec-driven
├── specs/                       # Canonical specs (source of truth)
└── changes/
    ├── archive/                 # Completed, archived changes
    └── <change-name>/           # Active change
        ├── .openspec.yaml       # Metadata (schema, created date, goal)
        ├── README.md
        ├── proposal.md          # Why + What changes
        ├── design.md            # How: context, decisions, risks
        ├── tasks.md             # Checklist: - [ ] / - [x]
        └── specs/               # Delta specs (merged on archive)
```

## The Workflow

### 1. Create a Change

```bash
openspec new change <kebab-case-name>
```

This scaffolds: proposal.md, design.md, tasks.md, .openspec.yaml, specs/

### 2. Fill the Artifacts

**`.openspec.yaml`** — minimal:
```yaml
schema: spec-driven
created: YYYY-MM-DD
goal: "One-line summary of what this change achieves"
```

**`proposal.md`** — sections:
- `## Why` — problem statement
- `## What Changes` — bullet list of changes
- `## Capabilities` — new and modified capabilities
- `## Impact` — affected crates/systems

**`design.md`** — sections:
- `## Context` — current state
- `## Goals / Non-Goals` — G1, G2... / NG1, NG2...
- `## Decisions` — D1, D2... with rationale
- `## Risks / Trade-offs` — R1, R2... with mitigation

**`tasks.md`** — numbered checklist grouped by section:
```markdown
## 1. Section Name
- [ ] 1.1 Task description
- [ ] 1.2 Task description
- [x] 1.3 Completed task

## 2. Cierre
- [ ] 2.1 `openspec validate <name>` pasa sin errores.
- [ ] 2.2 `cargo test --workspace` pasa.
- [ ] 2.3 Commit y push.
```

### 3. Implement Tasks

- Read `tasks.md` to know what's pending (`- [ ]` vs `- [x]`)
- After completing a task, mark it `- [x]` in `tasks.md`
- Run `openspec status --change <name>` to see progress percentage

### 4. Validate

```bash
openspec validate <change-name>
```

This checks structural integrity. Fix any errors before archiving.

### 5. Archive (when all tasks done)

```bash
openspec archive <change-name> -y
```

This moves the change to `archive/` and merges delta specs into `openspec/specs/`. If the change didn't modify specs (infra/docs-only), use `--skip-specs`.

## Key Conventions from This Project

- **Crate-specific tasks**: reference crates like `crates/compiler/src/op_fragments.rs`
- **Closure section always**: every `tasks.md` must end with a "Cierre" section containing validate + test + commit tasks
- **Task IDs use dotted numbering**: `1.1`, `1.2`, `2.1`, etc. matching the section number
- **Goal in `.openspec.yaml`**: single-line, max ~120 chars, describes the change end-state
- **Deferred tasks**: mark with `(Deferred: reason)` instead of leaving unchecked indefinitely
- **Git commits**: use conventional-ish prefixes like `feat(compiler):`, `plan(openspec):`, `fix(runtime):`

## Common Commands Checklist

| When you need to... | Command |
|---|---|
| See all active changes | `openspec list` |
| See task progress | `openspec status --change <name>` |
| Check what's left | Read `openspec/changes/<name>/tasks.md`, count `- [ ]` |
| Mark a task done | Edit `tasks.md`: `- [ ]` → `- [x]` |
| Validate a change | `openspec validate <name>` |
| Archive completed change | `openspec archive <name> -y` |
| Create new change | `openspec new change <name>` |
| See canonical specs | `openspec list --specs` |

## Agent Behavior

When the user asks about OpenSpec-related work:

1. **Check current state first**: run `openspec list` and `openspec status --change <name>` for relevant changes
2. **Prioritize by tasks.md**: the `- [ ]` items are the source of truth for what's pending
3. **Update tasks.md as you work**: after completing a task, edit the file to mark it `- [x]`
4. **Propose archiving**: when all tasks are `[x]`, offer to run `openspec archive <name> -y`
5. **Never skip validation**: always validate before archiving
6. **Follow project conventions**: use the same format as existing changes for new proposals/designs/tasks

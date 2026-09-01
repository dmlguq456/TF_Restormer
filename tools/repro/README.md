# tools/repro — archived reproduction and verification scripts

Reproduction, probe, and verification scripts that were authored while
investigating a specific issue and that a report or handoff cites as its
procedure. They were previously kept beside those reports under the
(gitignored) `.claude_reports/` artifact tree; they are runnable code, so they
now live in the source tree instead. Each origin keeps a digest-pinned
`RELOCATED.json` / `RELOCATED.md` pointer.

One directory per investigation context:

| directory | origin | scripts |
|---|---|---:|
| `2026-08-09_fs-loss-bugfix/` | `plans/2026-08-09_fs-loss-bugfix/scripts/` | 12 |
| `2026-08-09_issue2_checkpoint_eval/` | `experiments/2026-08-09_issue2_checkpoint_eval/_internal/` | 2 |
| `2026-08-09_issue2_collapse_root_cause/` | `experiments/2026-08-09_issue2_collapse_root_cause/_internal/` | 3 |
| `2026-08-09_issue2_epoch19_release/` | `experiments/2026-08-09_issue2_epoch19_release/_internal/` | 1 |
| `2026-03-31_rebuttal-gsc/` | `rebuttal/` | 1 |

## Running

These are archived one-shot investigation scripts, not a supported API. They
resolve their inputs relative to the **repository root**, so run them from
there, and with the project virtualenv directly:

    .venv/bin/python tools/repro/2026-08-09_fs-loss-bugfix/verify_ssl_fs.py

Do not use `uv run`: without the accelerator extra it replaces the pinned
`torch==2.6.0+cu124` build with the PyPI wheel.

Many of them need checkpoints, TensorBoard logs, or datasets that are not part
of this repository, and some reference scratch paths from the session that
produced them. Treat a failure to run as expected unless you have the same
inputs. Nothing here is imported by the `tf_restormer` package or the tests.

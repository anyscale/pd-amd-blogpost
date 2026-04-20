# Pre-publication cleanup TODOs

Action items to address before this repo goes public.

## Dockerfile

- [ ] Rewrite header comment block (lines 1–14):
  - drop `Dockerfile.v0.18.dev` filename references (file is just `Dockerfile`)
  - drop `kouroshhahkha/rocm-vllm-ray:v0.18-dev` tag examples
  - rename "Anyscale DEV image" title to something neutral
  - keep the `--build-arg BASE_IMAGE=rayproject/ray:...` example

## README.md

- [ ] Line 28: drop `IMAGE="kouroshhahkha/anyscale-rayllm:nightly-py312-rocm700"`
  pre-built reference; replace with the build-from-Dockerfile flow.
- [ ] Line 65: fix the "change the base image in the Dockerfile" instruction
  — `BASE_IMAGE` is already an `ARG`, so the correct command is
  `docker build --build-arg BASE_IMAGE=rayproject/ray:nightly-py312-cu128 -t pd-vllm-rocm .`

## serve_configs

- [ ] All 10 `serve_configs/{pd,agg}/*.yaml` files: drop or rewrite the
  `# Used for: EXP-1a, EXP-3b, ...` comments. The experiment IDs are not
  defined anywhere in the repo. Replace with a one-line workload description
  or remove the line entirely.

## compute_configs/pd-amd-mi325x-autoscale.yaml

- [ ] Line 43: soften the `# CRITICAL FIX:` debug-note comment to plain
  documentation, e.g. "Required: GID 991 grants the container access to
  /dev/kfd and /dev/dri/renderD* on AMD hosts."

## Deferred

- Blog draft (`draft/blogpost_draft.md`) — handled separately.
- All `[Link TBD]` and empty `[Anyscale]()` / `[Digital Ocean]()` URLs —
  revisit during a later editorial pass.
- `Dockerfile` line 193 `~/.workspacerc` source line — kept as-is.

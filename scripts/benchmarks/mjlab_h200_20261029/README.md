# H200 mjlab benchmark reproduction

These scripts record the 2026-10-29 ASTRO-SUPERMIMIC validation on H200.
They intentionally retain the machine-specific repository, dataset, output paths
and experiment names. They are not a general benchmark CLI.

See `docs/benchmarks/mjlab_h200_20261029/report.md` for the protocol and results.
The Python copies here are formatted for repository style. Original executed
scripts, rank logs, GPU telemetry, manifests and original script hashes are
preserved in `docs/benchmarks/mjlab_h200_20261029/raw_evidence.tar.gz`.
No motion tensors, model checkpoints or mesh binaries are included.

To reproduce on this H200 deployment, copy the Python scripts to
`/data/workspace/mjlab-build/realbench_20261029/`, then run the command in the
report with a fresh experiment label. Do not replace scripts while a benchmark
is active. `prepare_subset.py` is run on the source 5090-1 container with
CUDA disabled; its original 8 packaged shards must already exist.

The runner verifies free memory before starting, creates its own process group,
and only signals that group. Keep the Torch allocator budget, absolute free
memory threshold and timeout. `--headroom-only` makes the absolute free-memory
threshold authoritative because other jobs make baseline growth attribution
unreliable; reference-corrected growth remains an estimate for reporting.

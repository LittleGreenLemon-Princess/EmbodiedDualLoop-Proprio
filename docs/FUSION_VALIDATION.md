# Fusion validation — 2026-10-23

Validated under Ubuntu / WSL, Python 3.12.3. Installed CPU packages:
gymnasium 1.4.0, numpy 2.5.4, pillow 13.0.0, pyyaml 6.0.3,
imageio 2.38.1 and imageio-ffmpeg 0.6.0 (plus their dependencies).
This is a local CPU validation environment, not a replacement for the upstream
host or simulator dependency lock.

- `python -m unittest discover -s tests`: 173 tests, OK, 1 skipped.
  The optional robot-policy test module requires websockets >=15 and was skipped.
- CPU smoke with `--proprioception`: `awaiting_verification`, 67 decisions,
  66 action calls, condition `dual-loop-proprio-v1`.
- CPU smoke with `--proprioception --stuck`: `needs_system2`, reason
  `no_progress`, 13 decisions and 13 action calls.
- The skill compiler generated a complete embedded-source task from the pinned
  rim-pick entry; a client test checks that this format submits successfully.
- `git diff --check`: passed.

Initial Windows checks could not import the upstream POSIX `fcntl` dependency;
validation moved to WSL. Initial WSL checks exposed Windows CRLF conversion of
checksum-pinned public recipes. Exact upstream committed bytes were restored;
the recipe manifest and recipe contents were not changed. `.gitattributes`
now preserves their LF bytes. Workspace client synchronization was checked and
the baseline client retains its submission-only/no-reset behavior.

No model API calls, GPU simulation, physical robot operation, contact/grasp
validation or latency benchmarking was performed. The CPU follower has no object
physics. Raw run artifacts are local and ignored by Git.

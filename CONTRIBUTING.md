# Contributing

Thanks for your interest in contributing! This repository contains a suite of Python GUI applications for NMR/MRI scientific data analysis, and contributions from the community are welcome.

## How to contribute

1. **Fork** this repository to your own GitHub account.
2. **Clone** your fork to your local machine.
3. **Create a branch** for your change, based on `main`. Use a short descriptive name, e.g. `fix-ppm-calibration` or `add-bootstrap-option`.
4. **Make your changes** and commit them with clear commit messages.
5. **Push** the branch to your fork.
6. **Open a pull request** against `main` in this repository. Describe what the change does, why it's needed, and — where relevant — how you tested it.

A maintainer will review your PR. Small fixes and improvements are usually reviewed quickly.

## Protected folders

Two folders in this repository are **protected** and require explicit maintainer approval before any change can be merged:

- `Working-Data-Pipeline/`
- `Raw-Data-Visualization/`

These contain validated analysis code where correctness matters for downstream scientific results (ppm calibration, phase correction, fitting routines, etc.). PRs touching these paths will automatically request review from the repository owner and cannot be merged until that review is approved.

This isn't intended to discourage contributions — improvements to these folders are very welcome — but changes there will get closer scrutiny and may involve a back-and-forth to verify correctness against reference data before merging.

Other folders (including `developmental/`) are open for contribution with lighter review.

## What makes a good pull request

- **Keep changes focused.** One PR per logical change. If you're fixing a bug and also refactoring something unrelated, please split them.
- **Match the existing style.** The codebase uses PyQt5 for GUIs, with matplotlib/seaborn for plotting and scipy/numpy for numerical work. Follow the conventions in nearby code.
- **Test scientific changes carefully.** If you're modifying fitting routines, statistical methods, or signal processing, include a note in the PR about how you verified correctness — ideally against known reference data or a published result.
- **Document non-obvious choices.** A short comment explaining *why* a particular approach was used is much more valuable than a comment restating *what* the code does.
- **Don't commit data files or large binaries** unless they're essential test fixtures. If you need reference data for a test, open an issue first to discuss.

## Reporting issues

If you've found a bug or want to suggest an improvement, please open an issue before writing code for anything non-trivial. This avoids wasted effort if the idea needs discussion first. For bug reports, include:

- What you expected to happen
- What actually happened
- Steps to reproduce (ideally with a minimal example)
- Python version, OS, and relevant package versions (PyQt5, scipy, numpy, etc.)

## Questions

If you're unsure whether a change fits the project or how to approach it, open an issue and ask. It's always easier to talk first than to rework a PR.

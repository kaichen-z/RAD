# Contributing

Thank you for improving RAD. Please open an issue before a large protocol or data-layout change.
Keep benchmark claims aligned with the current paper and separate physical instances from the 13
reported semantic categories.

## Development

```bash
conda env create -f environment.yml
conda activate rad-benchmark
ruff check .
python -m compileall -q src methods
pytest
bash -n scripts/*.sh
```

Pull requests should include focused tests, avoid machine-specific paths, and never commit datasets,
weights, generated outputs, service credentials, or full third-party repositories. New adapters
must record upstream URL, exact revision, license, and the RAD-specific modification in
`methods/manifest.toml` and `methods/README.md`.

By contributing, you agree that your RAD-authored contribution is licensed under this repository's
MIT License. Do not submit code that you do not have permission to redistribute.

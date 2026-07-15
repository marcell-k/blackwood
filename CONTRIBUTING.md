# Contributing to Blackwood

## Prerequisites

- Python 3.13+ (`.python-version` pins 3.14 for the maintainer — delete it if your setup differs)
- [uv](https://docs.astral.sh/uv/)

## Setup

```bash
git clone https://github.com/marcell-k/blackwood.git
cd blackwood
uv sync --extra dev
```

For ML or portfolio modules:

```bash
uv sync --extra all --extra dev
```

Copy `.env.example` to `.env` and set your data paths:

```bash
cp .env.example .env
```

## Development workflow

**Lint and format:**

```bash
uv run ruff check src/blackwood/
uv run ruff format src/blackwood/
```

**Type-check:**

```bash
uv run basedpyright
```

**Tests:**

```bash
uv run pytest
```

All three must pass before opening a PR. CI will enforce them.

## Making changes

1. Fork the repo and create a branch from `main`: `git checkout -b your-feature`
2. Keep changes focused — one feature or fix per PR
3. Add or update tests for any logic you change
4. Update the relevant docstrings if the public interface changes
5. Open a PR against `main` with a clear description of what and why

## Code style

- Ruff enforces formatting and lint rules defined in `pyproject.toml` — don't fight them
- Type annotations on all public functions; `basedpyright` in standard mode must be clean
- Line length 120
- No commented-out code in PRs

## Adding a new module

If your module has optional heavy dependencies (e.g. `cvxpy`, `xgboost`), guard the import:

```python
try:
    import cvxpy as cp
except ImportError as e:
    raise ImportError("Install the portfolio extra: uv add blackwood[portfolio]") from e
```

## Reporting bugs

Open an issue with:
- Python version and OS
- Minimal reproduction script
- Full traceback

## License

By contributing you agree your work is released under the project's [Apache 2.0 license](LICENSE.md).

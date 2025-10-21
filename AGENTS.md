# Repository Guidelines

## Project Structure & Module Organization
- Source code lives in `memgpt/` (core library and modules).
- Entry points and small helpers in `main.py`.
- Tests in `tests/` mirroring package paths (e.g., `tests/test_memory.py`).
- Docs in `docs/` with site config in `mkdocs.yml`.
- Data/assets in `db/` when needed (small local fixtures only).

## Build, Test, and Development Commands
- `poetry install` — create/refresh the virtualenv and install deps.
- `poetry shell` — activate the project environment.
- `poetry run pytest -q` — run tests quietly.
- `poetry run pytest -q --maxfail=1 -x` — stop on first failure.
- `poetry run python main.py` — run the CLI/entry script if applicable.
- `pre-commit run -a` — run formatters/linters locally.

## Coding Style & Naming Conventions
- Python 3.10+; follow PEP 8 with 4-space indents.
- Names: modules `snake_case.py`, classes `PascalCase`, functions/vars `snake_case`, constants `UPPER_SNAKE`.
- Imports: stdlib → third-party → local, separated by blank lines.
- Auto-format and lint via pre-commit hooks (ruff/black per repo defaults).

## Testing Guidelines
- Framework: `pytest` with unit tests in `tests/`.
- Name tests `test_*.py`; use fixtures for shared setup.
- Aim for meaningful coverage on core logic (no strict % gate).
- Example: `poetry run pytest tests/test_memory.py::test_retention`.

## Commit & Pull Request Guidelines
- Commits: concise imperative subject (<=72 chars), body explaining why.
  - Example: "Add retrieval adapter for vector backend"
- Link related issues; include steps to test and any screenshots for UX changes.
- Keep PRs focused; update docs/tests with code.
- CI, lint, and tests must pass before merge.

## Security & Configuration Tips
- Do not commit secrets. Use environment variables for keys/endpoints.
- Prefer local `.env` loaded at runtime; document required variables in README.md.
- Validate user-provided paths and inputs in CLI entry points.

# Contributing to Helios

Thank you for your interest in Helios. This page covers:
* How to setup a development environment,
* The standards your code must meet, and
* The process for submitting changes.

If you are looking for a good place to start, see the issues labelled
[good first issue](https://github.com/marovira/helios-ml/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22).

---

## Development Setup

Helios uses [uv](https://docs.astral.sh/uv/) (>= 0.7) for dependency management.

```bash
# Clone the repository
git clone https://github.com/marovira/helios-ml.git
cd helios-ml

# Install all dependencies including development tools
uv sync --group dev

# Install the pre-commit hooks
uv run pre-commit install
```

The pre-commit hooks run Ruff, Mypy, and
[git-sumi](https://sumi.rs) automatically on each commit. All checks must pass before
a commit is accepted.

---

## Code Standards

### Style and Formatting

* Use [PEP 8](https://peps.python.org/pep-0008/) for all code.
* Format code with Ruff. The line length limit is 90 characters and docstrings follow
  the [PEP 257](https://peps.python.org/pep-0257/) convention.
* Do not import classes, functions, or exceptions directly from a module. Always import
  the nearest containing module and qualify names from it.
* Type-hint all function arguments and return values.

> [!NOTE]
> Use built-in generic types like `list`, `dict`, etc instead of types from `typing` where
> applicable. Ruff will ensure this is enforced.

### Running the Checks

```bash
# Lint
uv run ruff check src/helios
uv run ruff check test

# Format
uv run ruff format src/helios

# Type check
uv run mypy src/helios
uv run mypy test
```

Both Ruff and Mypy must pass cleanly before submitting a pull request.

---

## Tests

```bash
# Run all tests
uv run python -m pytest

# Run a single test file
uv run python -m pytest test/test_trainer.py

# Run a specific test
uv run python -m pytest test/test_trainer.py::test_function_name
```

Ensure that all unit tests pass when you make any changes. If you:
* Add a new feature, you *must* add unit tests to cover the new behaviour,
* Fix a bug, either update existing unit tests or add new ones, as applicable.

---

## Commit Style

Commit messages are linted by [git-sumi](https://sumi.rs). The rules are:

* The subject line must begin with an uppercase imperative verb (e.g. `Add`, `Fix`,
  `Update`, `Remove`).
* The subject line must not exceed 72 characters.
* Body lines must not exceed 75 characters.

**Example:**

```
Add LinearWarmupScheduler section to quick-ref

Covers constructor arguments, the two-phase behaviour, and the
typical setup pattern using get_train_steps_per_epoch().
```

---

## Pull Requests

1. Fork the repository and create a branch from `master`.
2. Make your changes, ensuring all checks and tests pass.
3. Open a pull request against `master` with a clear description of what was changed
   and why.
4. Reference any related issues in the PR description.

Pull requests that introduce new public API should include documentation updates.

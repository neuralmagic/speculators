# Developing for Speculators

Thank you for your interest in contributing to Speculators! This document provides detailed instructions for setting up your development environment, implementing changes, and adhering to the project's best practices. Your contributions help us grow and improve this project.

## Setting Up Your Development Environment

### Prerequisites

Before you begin, ensure you have the following installed:

- Python 3.9 or higher
- pip (Python package manager)
- Tox
- Git

### Cloning the Repository

1. Clone the repository to your local machine:

   ```bash
   git clone https://github.com/neuralmagic/speculators.git
   cd speculators
   ```

2. (Optional) If you plan to contribute changes back, fork the repository and clone your fork instead:

   ```bash
   git clone https://github.com/<your-username>/speculators.git
   cd speculators
   ```

### Installing Dependencies

To install the required dependencies for the package and development, run:

```bash
pip install -e ./[dev]
```

The `-e` flag installs the package in editable mode, allowing you to make changes to the code without reinstalling it. The `[dev]` part installs additional dependencies needed for development, such as testing and linting tools.

## Implementing Changes

### Writing Code

1. **Create a Branch**: Create a new branch for your changes:

   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make Changes**: Implement your changes in the appropriate files. Ensure that all public functions and classes have clear and concise docstrings.

3. **Update Documentation**: Update or add documentation to reflect your changes. This includes updating README files, docstrings, and any relevant guides.

## Running Quality, Style, and Type Checks

We use Tox to simplify running various tasks in isolated environments. Tox standardizes environments to ensure consistency across local development, CI/CD pipelines, and releases. This guarantees that the code behaves the same regardless of where it is executed.

Additionally, to ensure consistency and quality of the codebase, we use [ruff](https://github.com/astral-sh/ruff) for linting and styling, [isort](https://pycqa.github.io/isort/) for sorting imports, [mypy](https://github.com/python/mypy) for type checking, and [mdformat](https://github.com/hukkin/mdformat) for formatting Markdown files.

### Code Quality and Style

To check code quality, including linting and formatting:

```bash
tox -e quality
```

To automatically fix style issues:

```bash
tox -e style
```

### Type Checking

To ensure type safety using Mypy:

```bash
tox -e types
```

### Automating Quality Checks with Pre-Commit Hooks (Optional)

We use [pre-commit](https://pre-commit.com/) to automate quality checks before commits. Pre-commit hooks run checks like linting, formatting, and type checking, ensuring that only high-quality code is committed.

To install the pre-commit hooks, run:

```bash
pre-commit install
```

This will set up the hooks to run automatically before each commit. To manually run the hooks on all files, use:

```bash
pre-commit run --all-files
```

## Running Tests

For testing, we use [pytest](https://docs.pytest.org/) as our testing framework. We have different test suites for unit tests, integration tests, and end-to-end tests. To run the tests, you can use Tox, which will automatically create isolated environments for each test suite. Tox will also ensure that the tests are run in a consistent environment, regardless of where they are executed.

### Running All Tests

To run all tests:

```bash
tox
```

### Running Specific Tests

- Unit tests (focused on individual components with mocking):

  ```bash
  tox -e test-unit
  ```

- Integration tests (focused on interactions between components ideally without mocking):

  ```bash
  tox -e test-integration
  ```

- End-to-end tests (focused on the entire system and user interfaces):

  ```bash
  tox -e test-e2e
  ```

### Running Tests with Coverage

To ensure your changes are covered by tests, run:

```bash
tox -e test-unit -- --cov=speculators --cov-report=html
```

Review the coverage report to confirm that your new code is adequately tested.

## Opening a Pull Request

1. **Push Changes**: Push your branch to your forked repository (if you forked):

   ```bash
   git push origin feature/your-feature-name
   ```

2. **Open a Pull Request**: Go to the original repository and open a pull request. Use the following template for your pull request description:

   ```markdown
   # Title; ex: Add feature X to improve Y

   ## Summary:

   Short paragraph detailing the pull request changes and reasoning in addition to any relevant context.

   ## Details:

   - Detailed list of changes made in the pull request

   ## Test Plan:

   - Detailed list of steps to test the changes made in the pull request

   ## Related Issues

   - List of related issues or other pull requests; ex: "Fixes #1234"
   ```

3. **Address Feedback**: Respond to any feedback from reviewers and make necessary changes.

## Additional Resources

- [CONTRIBUTING.md](https://github.com/neuralmagic/speculators/blob/main/CONTRIBUTING.md): Guidelines for contributing to the project.
- [CODE_OF_CONDUCT.md](https://github.com/neuralmagic/speculators/blob/main/CODE_OF_CONDUCT.md): Our expectations for community behavior.
- [tox.ini](https://github.com/neuralmagic/speculators/blob/main/tox.ini): Configuration for Tox environments.
- [.pre-commit-config.yaml](https://github.com/neuralmagic/speculators/blob/main/.pre-commit-config.yaml): Configuration for pre-commit hooks.

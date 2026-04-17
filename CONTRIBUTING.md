# Contributing to VegaMini

Thank you for your interest in contributing to VegaMini! This document provides guidelines for contributors.

## Codebase Organization

- **`vega_mini/`**: This is the primary package. All core logic should reside here.
- **`run.py`**: The main entry point for the Day Loop.
- **`sleep.py`**: The main entry point for the Night Cycle.
- **`logs/`**: Generated structured logs (JSONL/CSV).
- **`visualizations/`**: Generated HTML visualizations.

## Development Workflow

1.  **Issue Tracking**: Check for open issues or create a new one to discuss your proposed changes.
2.  **Coding Standards**:
    - Use clear, descriptive variable and function names.
    - Include docstrings for all new modules, classes, and methods.
    - Follow PEP 8 style guidelines.
3.  **Documentation**:
    - Update `ARCHITECTURE.md` if you change core system contracts or add new layers.
    - Add usage examples to `README.md` for major new features.
4.  **Testing**:
    - Run the day loop and sleep cycle to ensure no regressions were introduced.
    - Check logs and visualizations to verify system behavior.

## Adding New Observability Tools

When adding new events or metrics:
- Use `logger.log_event` for high-level system changes.
- Use `logger.log_metrics` for numeric data intended for plotting.
- Update `vega_mini/vis/dashboard.py` if you add a new visualization type.

## Dependencies

Update `requirements.txt` if you introduce a new external dependency. Ensure the dependency is necessary and well-maintained.

## License

By contributing to VegaMini, you agree that your contributions will be licensed under the MIT License.

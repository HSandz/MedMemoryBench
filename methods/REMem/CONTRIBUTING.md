# Contributing to ReMEM

Thanks for your interest in contributing.

## How to contribute

1. Fork the repo and create a branch from `main`.
2. Make your changes.
3. Test that your changes work (run existing benchmarks if applicable).
4. Submit a pull request.

## Reporting issues

Open a GitHub issue with a clear description of the problem, steps to reproduce, and any relevant logs or error messages.

## Code style

- Follow existing conventions in the codebase.
- Keep commits focused — one logical change per commit.

## Adding new components

- **Embedding backend**: Add a class in `src/remem/embedding_model/` and register it in `__init__.py`.
- **Extraction method**: Add a module under `src/remem/information_extraction/` and update the factory in `remem.py`.
- **Retrieval strategy**: Add/extend a strategy in `src/remem/rag_strategies/`.
- **Prompt template**: Add a file in `src/remem/prompts/templates/`.
- **Evaluation metric**: Implement under `src/remem/evaluation/`.

## License

By contributing, you agree that your contributions will be licensed under the Apache 2.0 License.

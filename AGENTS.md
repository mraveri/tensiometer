# Code Style Guidelines

These conventions apply to all Python code in the project.

## Python version
- Target Python 3.9; avoid features added after 3.9.
- Prefer standard library modules; add dependencies only when essential.

## Typing
- Do not add type hints (including `typing` or `from __future__ import annotations`).
- Keep function signatures simple: positional args first, then keyword-only if necessary.

## Simplicity first
- Choose the simplest implementation that is correct and readable.
- Favor small, focused functions over deeply nested logic.
- Avoid clever one-liners; make control flow explicit.
- Keep naming clear and descriptive; avoid abbreviations unless obvious.

## File layout
- Add a single top-level docstring summarizing the module's purpose.
- Place all imports at the top of the file, grouped by standard library, third-party, and local. Unless otherwise justified.
- Separate major sections (imports block, class groups, helper sections) with a short header comment using the form:

```
#########################################################################################################
# Short description
```

- Give modules, classes, and methods concise triple-quoted docstrings; describe parameters and return values with ReST field lists (`:param:`, `:returns:`) and include expected shapes in double backticks where relevant.
- Preserve existing author attribution blocks or comments when present; update or add credits only when appropriate.
- Keep class APIs consistent across related implementations (e.g., shared `sample` and `logp` signatures), and normalize constructor inputs with `torch.as_tensor` to ensure dtype/device alignment.
- Use snake_case naming throughout; avoid inline comments unless they clarify non-obvious code; prefer explicit validation (e.g., `ValueError`) for bad inputs over implicit failures.

## Documentation
- Document modules, classes, and functions with ReST-style docstrings.
- Explain purpose, arguments, return values, side effects, and errors raised.
- Prefer short paragraphs and bullet lists over prose when it improves clarity.
- Keep examples runnable and minimal.
- Use field lists (`:param:`, `:returns:`, `:raises:`) for parameters, returns, and errors.

## Docstring template

Use this pattern for public functions and methods:

```python
def example(a, b, *, scale=1):
    """Compute something useful.

    :param a: first value to process
    :param b: second value to process
    :param scale: factor applied to the result (default 1)
    :returns: the computed result after scaling
    :raises ValueError: if the inputs are invalid
    """
    # function body
```

Keep private helpers brief; a single-sentence docstring is acceptable when obvious.

## Testing notes
- When adding behavior, include focused tests that cover the new paths.
- Prefer readable assertions over dense one-liners; use fixtures only when they simplify the test.

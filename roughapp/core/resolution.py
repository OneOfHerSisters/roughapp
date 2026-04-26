"""
Resolution functions for inconsistent decision tables.

Each function receives a DataFrame, applies a resolution strategy, and returns
the modified table. The logic is kept in the core layer so it stays independent
from transport and storage — the same functions can be called from the CLI,
the API, or a test without any changes.

This follows the Open/Closed Principle: adding a new resolution strategy means
adding a new function here, without modifying anything that already exists.
"""

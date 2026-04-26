"""
Router for detecting inconsistencies in a decision table.

It receives the table, passes it to the analyzer service, and returns a
structured report. The detection logic lives in services/analyzer.py, not here.

This follows the Single Responsibility Principle: the router handles only
the HTTP contract. If the /analyze endpoint contract changes, this is the
only file that needs to change.
"""

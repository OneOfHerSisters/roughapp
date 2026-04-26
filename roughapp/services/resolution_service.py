"""
Orchestrates the resolution workflow.

It receives a ResolutionRequest from the router, calls the appropriate function
from core/resolution.py, and saves the result through the repository. Business
logic and persistence details are handled by the layers below — this service
only coordinates between them.

This follows the Single Responsibility Principle: it has one reason to change,
which is if the resolution workflow itself changes. It also follows the
Dependency Inversion Principle: it depends on TableRepository rather than
sqlite3 directly, so the storage layer can change independently.
"""

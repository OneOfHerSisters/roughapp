"""
Router for applying inconsistency resolution strategies.

It receives the user's selected strategy and delegates the modification of
the decision table to the resolution service. Analysis and resolution are kept
as separate workflow steps.

This follows the Single Responsibility Principle: the router handles only the
HTTP contract, while strategy selection, table modification, and persistence
belong to the service, core, and infrastructure layers.
"""

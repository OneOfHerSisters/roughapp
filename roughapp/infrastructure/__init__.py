"""
Infrastructure layer for the roughapp application.

Contains everything that communicates with external systems. Currently that
is only the SQLite database, but keeping this layer separate means the service
and core layers do not need to change if the storage backend changes.
"""

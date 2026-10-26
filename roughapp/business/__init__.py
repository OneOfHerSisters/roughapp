"""Business logic layer.

Sits between the presentation layer (roughapp.api) and the data access layer
(roughapp.db).  Knows nothing about HTTP or SQLAlchemy: services raise the
domain exceptions from :mod:`roughapp.business.exceptions`, and the
presentation layer maps them to HTTP responses.
"""

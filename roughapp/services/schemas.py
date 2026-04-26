"""
Pydantic schemas for the API layer: DecisionTableInput, InconsistencyReport,
ResolutionRequest, ResolutionResult.

All schemas are defined in one place because both the analysis and resolution
routers use them. This follows the Interface Segregation Principle: each schema
contains only the fields relevant to its specific operation.
"""

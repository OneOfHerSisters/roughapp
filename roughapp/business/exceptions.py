"""Domain exceptions.

Raised by business models and services; translated to HTTP responses by
roughapp.api.exception_handlers.  The business layer never imports FastAPI.
"""


class BusinessError(Exception):
    """Base class for every domain error."""

    def __init__(self, message: str = ""):
        super().__init__(message or self.__class__.__name__)
        self.message = message or self.__class__.__name__


class DatasetNotFoundError(BusinessError):
    def __init__(self, dataset_id: int | None = None):
        super().__init__("Dataset not found")
        self.dataset_id = dataset_id


class ExperimentNotFoundError(BusinessError):
    def __init__(self, experiment_id: int | None = None):
        super().__init__("Experiment not found")
        self.experiment_id = experiment_id


class ResultNotFoundError(BusinessError):
    """A strategy result (or its stored file) does not exist."""


class AccessDeniedError(BusinessError):
    """Actor may not perform the operation on the resource.

    Note: for *reads* of private resources the services raise NotFound
    instead, so resource existence cannot be probed by id guessing.
    """


class BuiltinDatasetModificationError(BusinessError):
    def __init__(self):
        super().__init__("Public datasets cannot be modified directly")


class ValidationError(BusinessError):
    """Input is syntactically fine but semantically wrong (HTTP 422)."""


class UnknownStrategyError(ValidationError):
    def __init__(self, names):
        joined = ", ".join(names) if isinstance(names, (list, tuple)) else str(names)
        super().__init__(f"Unknown strategies: {joined}")


class UnknownPolicyError(ValidationError):
    pass


class ConflictError(BusinessError):
    """Operation conflicts with existing state (HTTP 409)."""


class CascadeRequiredError(ConflictError):
    """Deletion would take dependants with it; the caller must confirm.

    Carries a manifest of what else will be deleted so the client can show
    the full consequences before retrying with cascade=true.
    """

    def __init__(self, message: str, manifest: dict):
        super().__init__(message)
        self.manifest = manifest


class EmptyDecisionTableError(ValidationError):
    def __init__(self):
        super().__init__("Decision table is empty")


class DecisionColumnNotFoundError(ValidationError):
    def __init__(self, column: str):
        super().__init__(f"Decision column '{column}' not found in CSV")
        self.column = column


class StorageError(BusinessError):
    """Reading or writing a stored file failed."""

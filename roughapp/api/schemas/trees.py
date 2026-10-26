"""Request/response schemas for the trees router."""
from typing import Literal, Optional

from pydantic import BaseModel, Field


class TreeBuildRequest(BaseModel):
    kind: Literal["dataset", "result"]
    dataset_id: Optional[int] = None
    experiment_id: Optional[int] = None
    strategy: Optional[str] = None
    algorithm: str = Field(default="id3", description="Tree algorithm name (lookup table)")

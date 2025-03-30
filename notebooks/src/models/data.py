from typing import Any, List, Union

import pandas as pd
from pydantic import BaseModel, field_validator
from src.models.datasets import Datasets


class Dataset(BaseModel):
    class Config:
        arbitrary_types_allowed = True
        extra = "allow"

    @field_validator("*", mode="before")
    @classmethod
    def convert_field(cls, field: Any) -> Union[Any, List[Datasets]]:
        if isinstance(field, pd.DataFrame):
            return [Datasets(**row) for _, row in field.iterrows()]
        else:
            return field

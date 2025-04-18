from pydantic import BaseModel, Field
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from src.models.params import Params


class ClassifierModel(BaseModel):
    class Config:
        arbitrary_types_allowed = True
        extra = "allow"

    name: str = Field(..., description="Name of the model")
    model: BaseEstimator | Pipeline = Field(..., description="The sklearn model")
    params: Params = Field(..., description="The model parameters")

from pydantic import BaseModel


class Params(BaseModel):
    class Config:
        arbitrary_types_allowed = True
        extra = "allow"


class Estimators(Params):
    pass

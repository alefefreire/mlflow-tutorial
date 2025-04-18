from pydantic import BaseModel


class Dataset(BaseModel):
    class Config:
        arbitrary_types_allowed = True
        extra = "allow"

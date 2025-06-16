from pydantic import BaseModel

class ExampleBase(BaseModel):
    name: str

class ExampleCreate(ExampleBase):
    pass

class ExampleRead(ExampleBase):
    id: int

    class Config:
        orm_mode = True

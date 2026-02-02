from pydantic import BaseModel, Field
from typing import Annotated


class UserInput(BaseModel):

    formula : Annotated[str, Field(..., description='It is a material chemical formula')]

    
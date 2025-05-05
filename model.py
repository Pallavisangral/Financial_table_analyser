# importing standard modules ==================================================
from typing import Optional, Literal, List, Union

# importing third-party modules ===============================================
from pydantic import BaseModel, Field, root_validator

class Responder(BaseModel):
    status: Literal["success", "error"]
    message: str
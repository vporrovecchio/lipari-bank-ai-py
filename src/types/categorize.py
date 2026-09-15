from typing import Literal

from pydantic import BaseModel, Field

CategoryEnum = Literal["UTILITIES", "GROCERIES", "TRANSPORT", "RESTAURANTS", "ENTERTAINMENT", "OTHER"]


class CategorizeRequest(BaseModel):
    description: str = Field(..., min_length=1, max_length=200)
    amount: float = Field(..., gt=0)
    currency: str = Field(default="EUR", pattern=r"^[A-Z]{3}$")


class CategorizeResponse(BaseModel):
    category: CategoryEnum
    subcategory: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    reasoning: str
from fastapi import APIRouter

from src.types.categorize import CategorizeRequest, CategorizeResponse


router = APIRouter(prefix="/api/ai", tags=["Categorize"])


@router.post(
    "/categorize",
    response_model=CategorizeResponse,
    summary="Categorize transaction via LLM",
)
async def categorize(req: CategorizeRequest) -> CategorizeResponse:
    # Dummy: hardcoded category by keyword. In G4 useremo LLM.
    desc = req.description.lower()
    if any(k in desc for k in ["enel", "bolletta", "luce", "gas"]):
        return CategorizeResponse(
            category="UTILITIES", subcategory="ENERGY", confidence=0.92,
            reasoning="Description contains utility keywords",
        )
    if any(k in desc for k in ["supermercato", "esselunga", "coop"]):
        return CategorizeResponse(
            category="GROCERIES", subcategory="SUPERMARKET", confidence=0.85,
            reasoning="Description matches grocery store",
        )
    return CategorizeResponse(
        category="OTHER", subcategory="UNCATEGORIZED", confidence=0.10,
        reasoning="No matching pattern",
    )
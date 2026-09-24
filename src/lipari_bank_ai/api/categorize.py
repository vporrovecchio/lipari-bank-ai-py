from fastapi import APIRouter

from lipari_bank_ai.services.categorize_service import CategorizeService
from lipari_bank_ai.types.categorize import CategorizeRequest, CategorizeResponse

router = APIRouter(prefix="/api/ai", tags=["Categorize"])
categorize_service = CategorizeService()


@router.post("/categorize", response_model=CategorizeResponse)
async def categorize_endpoint(req: CategorizeRequest) -> CategorizeResponse:
    return await categorize_service.categorize(req)
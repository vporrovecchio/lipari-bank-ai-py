from fastapi import APIRouter, Depends

from lipari_bank_ai.services.categorize_service import CategorizeService
from lipari_bank_ai.types.categorize import CategorizeRequest, CategorizeResponse

router = APIRouter(prefix="/api/ai", tags=["Categorize"])

_categorize_service = CategorizeService()


def get_categorize_service() -> CategorizeService:
    return _categorize_service


@router.post("/categorize", response_model=CategorizeResponse)
async def categorize_endpoint(
    req: CategorizeRequest,
    service: CategorizeService = Depends(get_categorize_service),
) -> CategorizeResponse:
    return await service.categorize(req)
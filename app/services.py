from fastapi import APIRouter, HTTPException
from .rag import perform_rag_pipeline

router = APIRouter()

@router.post("/search")
def search_candidates(query: str):
    try:
        explanations = perform_rag_pipeline(query)
        return explanations
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

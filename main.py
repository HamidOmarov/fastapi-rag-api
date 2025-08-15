# --- ADD: /generate alias for compatibility ---
from fastapi import Form

@app.post("/generate")
async def generate_alias(
    file: UploadFile = File(...),
    num_questions: int = Query(20, ge=1, le=200),
    language: str = Query("en")
):
    # simply call the real generator
    return await generate_dataset(file=file, num_questions=num_questions, language=language)

@app.get("/health")
def health_alias():
    return {"status": "ok"}

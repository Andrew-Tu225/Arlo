"""FastAPI application.

Exposes a health-check endpoint for MVP.
The `app` FastAPI instance will be defined here; it is the target for:
    uvicorn core.api:app --reload
"""

from fastapi import FastAPI

app = FastAPI(title="Arlo", version="0.1.0")


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "version": app.version}

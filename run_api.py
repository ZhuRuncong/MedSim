"""Launch the FastMCP tool gateway (FastAPI).

    python run_api.py            # serves on http://localhost:8000
    # docs at /docs (Swagger UI)
"""
import os

import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "src.tools.server:app",
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8000")),
        reload=bool(os.getenv("RELOAD", "")),
    )

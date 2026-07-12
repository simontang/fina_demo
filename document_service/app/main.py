from fastapi import FastAPI

from app.api import router
from app.db import init_db


def create_app() -> FastAPI:
    app = FastAPI(title="Document Service", version="0.1.0")
    app.include_router(router)

    @app.on_event("startup")
    def _startup() -> None:
        init_db()

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    return app


app = create_app()

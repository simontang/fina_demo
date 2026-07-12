from pathlib import Path

from app.config import Settings
from app.engines.base import EngineAdapter
from app.engines.registry import EngineRegistry


def file_extension(filename: str) -> str:
    return Path(filename).suffix.lower().lstrip(".")


class Dispatcher:
    def __init__(self, settings: Settings):
        self.registry = EngineRegistry(settings)

    def select_engine(self, requested_engine: str, filename: str, size_bytes: int, params: dict) -> EngineAdapter:
        return self.registry.select(requested_engine, file_extension(filename), size_bytes, params)

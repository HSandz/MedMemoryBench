import importlib.metadata

try:
    __version__ = importlib.metadata.version("mem0ai")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.1.0-vendored"

from methods.mem0.client.main import AsyncMemoryClient, MemoryClient  # noqa
from methods.mem0.memory.main import Memory  # noqa

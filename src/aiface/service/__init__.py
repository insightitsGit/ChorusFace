"""External control surfaces for the live avatar (NWR-pattern bridges)."""

from aiface.service.bridge import (
    DEFAULT_HOST,
    DEFAULT_PORT,
    MAX_PENDING_JOBS,
    BridgeError,
    FaceBridge,
    is_loopback_host,
    new_token,
)

__all__ = [
    "DEFAULT_HOST",
    "DEFAULT_PORT",
    "MAX_PENDING_JOBS",
    "BridgeError",
    "FaceBridge",
    "is_loopback_host",
    "new_token",
]

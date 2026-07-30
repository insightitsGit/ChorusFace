"""Bind-address policy for the runtime's network surfaces.

The AI bridge and the operation relay hand a caller the ability to rewrite the
world. Both are documented as loopback services, but a default is only a default:
an address still arrives from a flag or an environment variable, and neither
server used to check it. This module turns that documented promise into a
refusal, so exposing a world to the network has to be an explicit decision rather
than a typo in ``NWR_AI_HOST``.
"""

from __future__ import annotations

import ipaddress
import socket
from typing import Final

__all__ = ["is_loopback_host", "require_loopback"]

# An empty host means "every interface" to the socket layer, same as 0.0.0.0.
_WILDCARD_HOSTS: Final[frozenset[str]] = frozenset({"", "*"})


def _is_loopback_literal(text: str) -> bool | None:
    """True/False for an IP literal, or None when `text` is not one."""
    try:
        return ipaddress.ip_address(text.split("%", 1)[0]).is_loopback
    except ValueError:
        return None


def is_loopback_host(host: object) -> bool:
    """Whether binding `host` keeps a server unreachable from the network.

    Names are resolved and every address they answer with has to be a loopback
    address. A name that resolves to both ``127.0.0.1`` and a routable address
    would expose the service, so it is not treated as loopback.
    """
    text = str(host or "").strip()
    if text.lower() in _WILDCARD_HOSTS:
        return False
    literal = _is_loopback_literal(text.strip("[]"))
    if literal is not None:
        return literal
    try:
        infos = socket.getaddrinfo(text, None)
    except (socket.gaierror, UnicodeError):
        return False
    resolved = {str(info[4][0]) for info in infos}
    if not resolved:
        return False
    return all(_is_loopback_literal(address) for address in resolved)


def require_loopback(
    host: object,
    *,
    allow_remote: bool = False,
    service: str = "service",
) -> str:
    """Return `host` when it is safe to bind, else raise `ValueError`.

    Passing ``allow_remote`` is the operator saying they meant it; the address is
    then returned unchanged.
    """
    text = str(host or "")
    if allow_remote or is_loopback_host(text):
        return text
    raise ValueError(
        f"Refusing to bind the {service} to {text!r}: it is reachable from the "
        "network and would hand world control to anyone who can reach the port. "
        "Bind 127.0.0.1, or pass --allow-remote-bind if that is what you want."
    )

"""SSRF protection — blocks requests to private/internal network addresses."""

from __future__ import annotations

import ipaddress
import re
from urllib.parse import urlparse


_BLOCKED_HOSTS = frozenset(
    [
        "localhost",
        "::1",
        "0.0.0.0",
        "metadata.google.internal",
        "169.254.169.254",  # AWS/GCP instance metadata
        "100.100.100.200",  # Alibaba Cloud metadata
    ]
)

_PRIVATE_NETWORKS = [
    ipaddress.ip_network("10.0.0.0/8"),
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("127.0.0.0/8"),
    ipaddress.ip_network("169.254.0.0/16"),
    ipaddress.ip_network("fc00::/7"),
    ipaddress.ip_network("fe80::/10"),
    ipaddress.ip_network("::1/128"),
]


def validate_url_for_ssrf(url: str) -> str | None:
    """
    Validates a URL against SSRF risks.
    Returns an error message string if blocked, None if safe.
    """
    try:
        parsed = urlparse(url)
    except Exception:
        return "Invalid URL"

    if parsed.scheme not in ("http", "https"):
        return f"Scheme '{parsed.scheme}' is not allowed; use http or https"

    hostname = parsed.hostname or ""
    if not hostname:
        return "Missing hostname"

    hostname_lower = hostname.lower()

    if hostname_lower in _BLOCKED_HOSTS:
        return f"Access to '{hostname}' is not allowed"

    # Block numeric IPs in private ranges
    try:
        addr = ipaddress.ip_address(hostname)
        for network in _PRIVATE_NETWORKS:
            if addr in network:
                return f"Access to private IP '{hostname}' is not allowed"
    except ValueError:
        pass  # Not an IP address — hostname is fine

    return None

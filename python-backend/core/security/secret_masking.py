"""Tiny helpers for safely surfacing API keys in logs / responses."""

from __future__ import annotations


def mask_secret(secret: str, *, keep: int = 4) -> str:
    """Mask a secret keeping only the first/last few chars.

    >>> mask_secret("sk-abcdef1234567890")
    'sk-a…7890'
    """
    if not secret:
        return ""
    if len(secret) <= keep * 2:
        return "*" * len(secret)
    return f"{secret[:keep]}…{secret[-keep:]}"

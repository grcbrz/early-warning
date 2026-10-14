"""Minimal anonymous S3 listing over HTTPS (AWS Open Data buckets need no credentials)."""

from __future__ import annotations

import re
import time
from functools import lru_cache

import requests

_KEY_RE = re.compile(r"<Key>([^<]+)</Key>")
_TOKEN_RE = re.compile(r"<NextContinuationToken>([^<]+)</NextContinuationToken>")


def get_with_retry(url: str, *, params: dict | None = None, retries: int = 4,
                   timeout: float = 60.0) -> requests.Response:
    """GET with exponential backoff on network errors and 5xx responses."""
    for attempt in range(retries + 1):
        try:
            resp = requests.get(url, params=params, timeout=timeout)  # body read eagerly
            if resp.status_code < 500:
                resp.raise_for_status()
                return resp
        except (requests.ConnectionError, requests.Timeout, requests.exceptions.ChunkedEncodingError):
            if attempt == retries:
                raise
        time.sleep(2 ** (attempt + 1))
    resp.raise_for_status()  # pragma: no cover - last response was 5xx
    return resp  # pragma: no cover


@lru_cache(maxsize=256)
def list_keys(bucket_url: str, prefix: str) -> tuple[str, ...]:
    """Return every object key under ``prefix`` (handles pagination)."""
    keys: list[str] = []
    token: str | None = None
    while True:
        params = {"list-type": "2", "prefix": prefix, "max-keys": "1000"}
        if token:
            params["continuation-token"] = token
        body = get_with_retry(bucket_url + "/", params=params).text
        keys.extend(_KEY_RE.findall(body))
        match = _TOKEN_RE.search(body)
        if not match:
            return tuple(keys)
        token = match.group(1)

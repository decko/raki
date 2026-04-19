import re
from typing import Any

PATTERNS = [
    re.compile(r'(?i)(bearer\s+)(?:[^\s"]+|"[^"]+")'),
    re.compile(r'(?i)(token[=:]\s*)(?:[^\s"]+|"[^"]+")'),
    re.compile(r'(?i)(password[=:]\s*)(?:[^\s"]+|"[^"]+")'),
    re.compile(r'(?i)(api[_-]?key[=:]\s*)(?:[^\s"]+|"[^"]+")'),
    # JWT -- including multi-line where header.payload.signature may span lines
    re.compile(r"eyJ[A-Za-z0-9_-]+\s*\.\s*eyJ[A-Za-z0-9_-]+\s*\.\s*[A-Za-z0-9_-]+", re.DOTALL),
    re.compile(r"AKIA[0-9A-Z]{16}"),  # AWS access keys
    re.compile(r"(?:ghp|gho|ghs|ghr|glpat)_[A-Za-z0-9_]{20,}"),  # GitHub/GitLab tokens
    # Specific env var patterns: AWS_SECRET_ACCESS_KEY, GITHUB_TOKEN, GH_TOKEN
    re.compile(r'(?i)(AWS_SECRET_ACCESS_KEY[=:]\s*)(?:[^\s"]+|"[^"]+")'),
    re.compile(r'(?i)(GITHUB_TOKEN[=:]\s*)(?:[^\s"]+|"[^"]+")'),
    re.compile(r'(?i)(GH_TOKEN[=:]\s*)(?:[^\s"]+|"[^"]+")'),
    # Generic *_SECRET*= env vars (e.g. MY_SECRET_KEY=..., DB_SECRET_TOKEN=...)
    re.compile(r'(?i)([A-Z_]*SECRET[A-Z_]*[=:]\s*)(?:[^\s"]+|"[^"]+")'),
    re.compile(r'(?i)(secret[=:]\s*)(?:[^\s"]+|"[^"]+")'),  # Generic secret keyword
    re.compile(
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----"
    ),  # Private key blocks
]


def redact_sensitive(text: str) -> str:
    """Strip common secret patterns from text before it enters EvalSample."""
    for pattern in PATTERNS:
        text = pattern.sub(
            lambda match: (
                match.group(1) + "***REDACTED***" if match.lastindex else "***REDACTED***"
            ),
            text,
        )
    return text


def redact_dict(data: dict[str, Any]) -> dict[str, Any]:
    """Recursively redact sensitive string values in a dict/list structure."""
    return {key: _redact_value(value) for key, value in data.items()}


def _redact_value(value: Any) -> Any:
    """Redact a single value, recursing into dicts and lists."""
    if isinstance(value, str):
        return redact_sensitive(value)
    if isinstance(value, dict):
        return {key: _redact_value(val) for key, val in value.items()}
    if isinstance(value, list):
        return [_redact_value(item) for item in value]
    return value
